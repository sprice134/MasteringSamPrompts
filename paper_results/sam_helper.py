import numpy as np
import cv2
import torch
import torch.nn.functional as F
from segment_anything import sam_model_registry, SamPredictor

############################################################################
#   1) Sub-algorithms used by select_point_placement
############################################################################

import random
import time
from itertools import combinations

random.seed(42)
np.random.seed(42)

def naiveMaximization(white_cells, num_points):
    """
    Check all combinations of 'num_points' cells and pick the set with the largest sum of pairwise distances.
    """
    max_distance = 0
    furthest_set = None
    num_checks = 0

    # Simple centroid-based short circuit for num_points = 1
    if num_points == 1:
        centroid_x = np.mean([cell[0] for cell in white_cells])
        centroid_y = np.mean([cell[1] for cell in white_cells])
        closest = min(white_cells, key=lambda c: (c[0]-centroid_x)**2 + (c[1]-centroid_y)**2)
        return [closest]

    for point_set in combinations(white_cells, num_points):
        num_checks += 1
        aggregate_distance = 0
        for i in range(num_points):
            for j in range(i+1, num_points):
                dist_ij = np.sqrt((point_set[i][0] - point_set[j][0])**2 + 
                                  (point_set[i][1] - point_set[j][1])**2)
                aggregate_distance += dist_ij
        if aggregate_distance > max_distance:
            max_distance = aggregate_distance
            furthest_set = point_set

    return furthest_set if furthest_set else []

def simulatedAnnealingMaximization(
    white_cells, num_points, initial_temp=1000, cooling_rate=0.995, max_iterations=1000, patience=300
):
    """
    Simulated Annealing to find a set of points that maximizes the sum of pairwise distances.
    """
    def calculate_total_distance(points):
        total_dist = 0
        for i in range(num_points):
            for j in range(i+1, num_points):
                dist = np.sqrt((points[i][0] - points[j][0])**2 + 
                               (points[i][1] - points[j][1])**2)
                total_dist += dist
        return total_dist

    # Initial random selection
    current_points = random.sample(white_cells, num_points)
    current_distance = calculate_total_distance(current_points)
    best_points = current_points[:]
    best_distance = current_distance
    temperature = initial_temp
    no_improvement_counter = 0

    for iteration in range(max_iterations):
        new_points = current_points[:]
        swap_index = random.randint(0, num_points - 1)
        new_points[swap_index] = random.choice(white_cells)
        new_dist = calculate_total_distance(new_points)

        if new_dist > current_distance or np.exp((new_dist - current_distance)/temperature) > random.random():
            current_points = new_points
            current_distance = new_dist
            if new_dist > best_distance:
                best_points = new_points[:]
                best_distance = new_dist
                no_improvement_counter = 0
            else:
                no_improvement_counter += 1
        else:
            no_improvement_counter += 1

        if no_improvement_counter >= patience:
            break

        temperature *= cooling_rate
        if temperature < 1e-10:
            break

    return best_points

def hillClimbingMaximization(white_cells, num_points, max_iterations=1000):
    """
    Hill Climbing to maximize total pairwise distance.
    """
    def calculate_total_distance(points):
        total_dist = 0
        for i in range(num_points):
            for j in range(i+1, num_points):
                dist = np.sqrt((points[i][0] - points[j][0])**2 + 
                               (points[i][1] - points[j][1])**2)
                total_dist += dist
        return total_dist

    current_points = random.sample(white_cells, num_points)
    current_dist = calculate_total_distance(current_points)
    best_points = current_points[:]
    best_distance = current_dist

    for iteration in range(max_iterations):
        improved = False
        for swap_index in range(num_points):
            for new_point in white_cells:
                if new_point != current_points[swap_index]:
                    test_points = current_points[:]
                    test_points[swap_index] = new_point
                    test_dist = calculate_total_distance(test_points)
                    if test_dist > current_dist:
                        current_points = test_points
                        current_dist = test_dist
                        improved = True
                        break
            if improved:
                break

        if current_dist > best_distance:
            best_points = current_points[:]
            best_distance = current_dist
        if not improved:
            break

    return best_points

def clusterInitialization(white_cells, num_points):
    """
    Greedily pick points that maximize distance from previously selected points.
    """
    selected_points = [random.choice(white_cells)]
    while len(selected_points) < num_points:
        max_dist = 0
        next_pt = None
        for pt in white_cells:
            min_dist_to_selected = min(
                np.sqrt((pt[0]-sp[0])**2 + (pt[1]-sp[1])**2) for sp in selected_points
            )
            if min_dist_to_selected > max_dist:
                max_dist = min_dist_to_selected
                next_pt = pt
        selected_points.append(next_pt)
    return selected_points

def randomSelection(white_cells, num_points):
    """
    Randomly select a set of points from white_cells.
    """
    if len(white_cells) < num_points:
        return white_cells
    return random.sample(white_cells, num_points)

def voronoi_optimization_from_coords(coords, num_points, iterations=50):
    """
    Perform Voronoi-based optimization to select points from a list of coordinates.
    """
    coords = np.array(coords)
    # 1) Initialize points by random sampling
    initial_indices = np.random.choice(len(coords), num_points, replace=False)
    points = coords[initial_indices]

    def voronoi_partition(coords, points):
        """Assign each coordinate to the nearest point."""
        dist_matrix = np.linalg.norm(coords[:, None] - points[None, :], axis=2)
        return np.argmin(dist_matrix, axis=1)

    for _ in range(iterations):
        region_assignment = voronoi_partition(coords, points)
        new_points = []
        for i in range(num_points):
            region_coords = coords[region_assignment == i]
            if len(region_coords) > 0:
                new_points.append(region_coords.mean(axis=0))
            else:
                new_points.append(coords[np.random.choice(len(coords))])
        new_points = np.array(new_points)
        if np.allclose(points, new_points, atol=1e-2):
            break
        points = new_points

    return points.tolist()

############################################################################
#   2) The main function: select_point_placement
############################################################################

from skimage.measure import label, regionprops
from scipy.ndimage import binary_erosion

def select_point_placement(
    mask, num_points, dropout_percentage=0, ignore_border_percentage=0,
    algorithm="Voronoi", select_perimeter=False
):
    """
    Selects N furthest points from a binary mask using a chosen algorithm.
    """
    rows, cols = np.where(mask > 0)
    white_cells = list(zip(rows, cols))

    # Keep only the largest connected component
    labeled_mask = label(mask)
    regions = regionprops(labeled_mask)
    if len(regions) > 1:
        largest_region = max(regions, key=lambda r: r.area)
        white_cells = [(r, c) for (r, c) in white_cells if labeled_mask[r, c] == largest_region.label]

    # If selecting only perimeter
    if select_perimeter:
        eroded_mask = binary_erosion(mask)
        perimeter_mask = mask & ~eroded_mask
        rows, cols = np.where(perimeter_mask > 0)
        white_cells = list(zip(rows, cols))

    # Apply border filtering
    if ignore_border_percentage > 0 and white_cells:
        min_r = min(r for r, _ in white_cells)
        max_r = max(r for r, _ in white_cells)
        min_c = min(c for _, c in white_cells)
        max_c = max(c for _, c in white_cells)
        height = max_r - min_r + 1
        width = max_c - min_c + 1
        ignore_h = int(height * ignore_border_percentage / 100)
        ignore_w = int(width * ignore_border_percentage / 100)
        inner_r_start = min_r + ignore_h
        inner_r_end = max_r - ignore_h
        inner_c_start = min_c + ignore_w
        inner_c_end = max_c - ignore_w
        white_cells = [
            (r, c) for (r, c) in white_cells 
            if inner_r_start <= r <= inner_r_end and inner_c_start <= c <= inner_c_end
        ]

    # Apply dropout
    if len(white_cells) > 0:
        keep_count = int(len(white_cells) * (1 - dropout_percentage/100))
        if keep_count < len(white_cells):
            white_cells = random.sample(white_cells, keep_count)

    # Dispatch to the chosen algorithm
    algo_map = {
        "Naive": naiveMaximization,
        "Simulated Annealing": simulatedAnnealingMaximization,
        "Hill Climbing": hillClimbingMaximization,
        "Cluster Initialization": clusterInitialization,
        "Random": randomSelection,
        "Voronoi": voronoi_optimization_from_coords,
    }
    white_cells_normalized = [(r / mask.shape[0], c / mask.shape[1]) for (r, c) in white_cells]
    start_time = time.time()
    selected_points = algo_map[algorithm](white_cells_normalized, num_points)
    run_time = time.time() - start_time

    # Convert normalized coords back to pixel
    selected_points_pixel = [
        (
            max(0, min(int(p[0]*mask.shape[0]), mask.shape[0]-1)),
            max(0, min(int(p[1]*mask.shape[1]), mask.shape[1]-1))
        )
        for p in selected_points
    ]

    # We won't track 'aggregate_distance' here; your existing logic can handle that if needed
    return selected_points_pixel, 0, run_time


############################################################################
#   3) The SAM predictor setup + bounding-box-based inference
############################################################################

def load_sam_model(sam_checkpoint, model_type="vit_l", device="cuda"):
    """
    Initialize and return a SAM predictor.
    """
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)
    return predictor


def expand_bbox_within_border(x1, y1, x2, y2, width, height, expansion_rate=0.0):
    """
    Expands or shrinks the bounding box by expansion_rate, recentered about its original center.
    E.g., expansion_rate=1.1 → 110% size; expansion_rate=0.9 → 90% size.
    """
    if expansion_rate <= 0:
        return [x1, y1, x2, y2]

    original_w = x2 - x1
    original_h = y2 - y1

    new_w = original_w * expansion_rate
    new_h = original_h * expansion_rate

    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2

    new_x1 = max(0, center_x - new_w / 2)
    new_y1 = max(0, center_y - new_h / 2)
    new_x2 = min(width, center_x + new_w / 2)
    new_y2 = min(height, center_y + new_h / 2)

    return [new_x1, new_y1, new_x2, new_y2]

def adjust_mask_area(mask, target_percentage, max_iterations=50, kernel_size=(3,3)):
    """
    Erodes or dilates 'mask' to reach a target area = (target_percentage / 100) * original_area.
    """
    if target_percentage == 0 or target_percentage == 100:
        return mask

    binary_mask = (mask > 0).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)

    original_area = np.sum(binary_mask)
    target_area = original_area * (target_percentage / 100.0)

    if target_percentage < 100:
        operation = 'erode'
    else:
        operation = 'dilate'

    new_mask = binary_mask.copy()
    for _ in range(max_iterations):
        current_area = np.sum(new_mask)
        # Check if we've met the area criteria
        if operation == 'erode' and current_area <= target_area:
            break
        if operation == 'dilate' and current_area >= target_area:
            break

        if operation == 'erode':
            new_mask = cv2.erode(new_mask, kernel, iterations=1)
        else:  # dilate
            new_mask = cv2.dilate(new_mask, kernel, iterations=1)

    return new_mask

def prepare_mask_for_sam(mask, target_size=(256, 256)):
    """
    Resize a 2D binary mask (H,W) so it can be fed into SAM's predictor as mask_input.
    """
    if len(mask.shape) != 2:
        raise ValueError("Mask must be a 2D array (H, W).")

    mask_float = mask.astype(np.float32)
    # Normalizing to [0,1] if needed
    if mask_float.max() > 1:
        mask_float /= 255.0

    mask_tensor = torch.tensor(mask_float, dtype=torch.float32)[None, None, :, :]  # shape (1,1,H,W)
    mask_resized = F.interpolate(
        mask_tensor,
        size=target_size,
        mode="bilinear",
        align_corners=False
    ).squeeze(0)  # -> shape (1, target_size[0], target_size[1])

    return mask_resized

def run_sam_inference(
    predictor,
    loop_image,
    listOfPolygons,
    listOfBoxes,
    listOfMasks,
    image_width,
    image_height,
    num_points=4,
    dropout_percentage=0,
    ignore_border_percentage=5,
    algorithm="Voronoi",
    use_box_input=True,   # <--- NEW PARAM
    use_mask_input=False,
    box_expansion_rate=0.0,
    mask_expansion_rate=0.0

):
    """
    Runs SAM segmentation refinement on predicted masks.
    Optionally pass an initial mask to SAM (mask_input) or expand bounding boxes.
    Returns a list of SAM-refined masks.
    """

    sam_masks_list = []
    if algorithm=="Distance Max":
        algorithm = 'Hill Climbing'

    def expand_bbox_within_border(x1, y1, x2, y2, width, height, expansion_rate=0.0):
        if expansion_rate <= 0:
            return [x1, y1, x2, y2]
        # Calculate original width and height
        original_w = x2 - x1
        original_h = y2 - y1
        
        # Calculate new width and height based on scale
        new_w = original_w * expansion_rate
        new_h = original_h * expansion_rate
        
        # Center the bounding box and calculate new coordinates
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        
        new_x1 = max(0, center_x - new_w / 2)
        new_y1 = max(0, center_y - new_h / 2)
        new_x2 = min(width, center_x + new_w / 2)
        new_y2 = min(height, center_y + new_h / 2)
    
        return [new_x1, new_y1, new_x2, new_y2]
    
    def adjust_mask_area(mask, target_percentage, max_iterations=50, kernel_size=(3,3)):
        if target_percentage == 0 or target_percentage == 100:
            return mask
        if target_percentage < 100:
            operation = 'erode'
        else:
            operation = 'dilate'
        # Ensure mask is binary
        binary_mask = (mask > 0).astype(np.uint8)

        # Create structuring element (e.g., elliptical kernel)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)

        # Calculate target area based on percentage
        original_area = np.sum(binary_mask)
        target_area = original_area * (target_percentage / 100.0)

        new_mask = binary_mask.copy()
        for i in range(max_iterations):
            current_area = np.sum(new_mask)
            # Check if we've met the area criteria.
            if operation == 'erode' and current_area <= target_area:
                break
            if operation == 'dilate' and current_area >= target_area:
                break
            # Apply morphological operation
            if operation == 'erode':
                new_mask = cv2.erode(new_mask, kernel, iterations=1)
            elif operation == 'dilate':
                new_mask = cv2.dilate(new_mask, kernel, iterations=1)
            else:
                raise ValueError("Operation must be 'erode' or 'dilate'.")
        return new_mask

    for index in range(len(listOfPolygons)):
        box = listOfBoxes[index]
        box = box.cpu().numpy() if hasattr(box, 'cpu') else np.array(box)

        # If use_box_input=False, skip bounding box usage entirely
        if use_box_input:
            x1, y1, x2, y2 = expand_bbox_within_border(
                box[0], box[1], box[2], box[3],
                image_width, image_height,
                expansion_rate=box_expansion_rate
            )
            box = np.array([x1, y1, x2, y2])
        else:
            box = None

        mask = listOfMasks[index]
        if np.sum(mask) > num_points * num_points:
            # Pick sample points from the existing mask
            if algorithm == 'Hill Climbing' and num_points != 1:
                # print('Selecting Points from perimeter')
                # print(num_points)
                # print(algorithm)
                try:
                    selected_points, _, _ = select_point_placement(
                        mask=mask,
                        num_points=num_points,
                        dropout_percentage=dropout_percentage,
                        ignore_border_percentage=ignore_border_percentage,
                        algorithm=algorithm,
                        select_perimeter=True
                    )
                except Exception as e:
                    print(f"Error selecting points: {e}")
                    print(f"Mask sum: {np.sum(mask)}")
                    continue  # Skip to the next iteration
            else:
                # print('Selecting Points from all')
                try:
                    selected_points, _, _ = select_point_placement(
                        mask=mask,
                        num_points=num_points,
                        dropout_percentage=dropout_percentage,
                        ignore_border_percentage=ignore_border_percentage,
                        algorithm=algorithm,
                        select_perimeter=False
                    )
                except Exception as e:
                    print(f"Error selecting points: {e}")
                    print(f"Mask sum: {np.sum(mask)}")
                    continue  # Skip to the next iteration

            # Modify mask after point selection to be independent from buffer
            mask = adjust_mask_area(mask, mask_expansion_rate)
            
            op_y, op_x = zip(*selected_points)
            predictor.set_image(loop_image)
            input_point = np.array(list(zip(op_x, op_y)))
            input_label = np.array([1] * len(input_point))

            # Optionally feed the mask to SAM
            mask_input = prepare_mask_for_sam(mask) if use_mask_input else None

            predict_kwargs = {
                'point_coords': input_point,
                'point_labels': input_label,
                'multimask_output': True
            }

            # Only pass bounding box if use_box_input=True and it's not degenerate
            if use_box_input and box is not None:
                x1, y1, x2, y2 = box
                if (box_expansion_rate > 0.0) or (x2 - x1 > 0 and y2 - y1 > 0):
                    predict_kwargs['box'] = box[None, :]

            if mask_input is not None:
                predict_kwargs['mask_input'] = mask_input

            try:
                masks, scores, logits = predictor.predict(**predict_kwargs)
                sam_masks_list.append(masks[0])  # Keep the first mask for simplicity
            except Exception as e:
                print(f"Error during prediction: {e}")
                continue  # Optionally skip to the next iteration if prediction fails
        else:
            print(f'Skipping Mask Because Area ({np.sum(mask)}) Smaller than number of points^2 ({num_points * num_points})')

    return sam_masks_list




def combine_masks(masks_list, output_mask_path):
    """
    Merge a list of boolean masks into a single labeled mask and save to disk.
    """
    if not masks_list:
        return None

    height, width = masks_list[0].shape
    combined_mask = np.zeros((height, width), dtype=np.uint8)

    for idx, mask in enumerate(masks_list, start=1):
        combined_mask[mask > 0] = idx

    cv2.imwrite(output_mask_path, combined_mask)
    print(f"Combined mask saved to {output_mask_path}")

    return combined_mask
