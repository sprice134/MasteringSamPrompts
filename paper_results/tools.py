import os
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pycocotools import mask as maskUtils
from pycocotools.cocoeval import COCOeval
from pycocotools.coco import COCO
import sys
from PIL import Image
import uuid




def visualize_segmentations(image_path, mask_path, output_dir, title="Segmentation Overlay"):
    """
    Visualize segmentation masks by overlaying them on the original image.
    Saves a PNG file with the overlay in the output directory.
    """

    original_image = Image.open(image_path).convert("RGBA")
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    if original_image is None or mask is None:
        raise FileNotFoundError(
            f"Image or mask not found: {image_path}, {mask_path}"
        )

    # Extract unique object IDs (excluding 0 which is background)
    unique_ids = np.unique(mask)
    unique_ids = unique_ids[unique_ids > 0]

    # Assign random colors (RGBA) for each object ID
    colors = [
        np.append(np.random.randint(0, 256, size=3), 180)  # RGB with alpha channel
        for _ in unique_ids
    ]

    # Create the composite mask
    composite_mask = np.zeros((*mask.shape, 4), dtype=np.uint8)
    for obj_id, color in zip(unique_ids, colors):
        binary_mask = (mask == obj_id)
        composite_mask[binary_mask] = color

    # Convert composite mask to an image
    composite_mask_image = Image.fromarray(composite_mask, mode="RGBA")

    # Overlay the mask onto the original image
    final_image = Image.alpha_composite(original_image, composite_mask_image)

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Save the final image
    output_overlay_path = os.path.join(output_dir, f"{title.replace(' ', '_')}.png")
    final_image.save(output_overlay_path, "PNG")



def mask_to_coco_format(binary_mask, image_id, category_id=1, annotation_id=1):
    """
    Convert a binary mask (NumPy array) to COCO RLE format.
    """
    rle = maskUtils.encode(np.asfortranarray(binary_mask.astype(np.uint8)))
    # pycocotools uses bytes for 'counts', so we decode for JSON serialization
    rle['counts'] = rle['counts'].decode('utf-8')
    return {
        "id": annotation_id,
        "image_id": image_id,
        "category_id": category_id,
        "segmentation": rle,
        "area": float(maskUtils.area(rle)),
        "bbox": maskUtils.toBbox(rle).tolist(),
        "iscrowd": 0
    }

def generate_coco_annotations_from_multi_instance_masks(
    gt_mask_path,
    pred_mask_path,
    image_path,
    image_id=1,
    category_id=1
    ):
    """
    Generate COCO-style annotations for both ground truth and predicted masks.
    """
    gt_mask = cv2.imread(gt_mask_path, cv2.IMREAD_GRAYSCALE)
    pred_mask = cv2.imread(pred_mask_path, cv2.IMREAD_GRAYSCALE)

    if gt_mask is None or pred_mask is None:
        raise FileNotFoundError("One of the mask files was not found.")

    # Identify unique object IDs
    gt_ids = np.unique(gt_mask)
    pred_ids = np.unique(pred_mask)
    gt_ids = gt_ids[gt_ids > 0]
    pred_ids = pred_ids[pred_ids > 0]

    gt_annotations = []
    pred_annotations = []

    # Create GT annotations
    for annotation_id, obj_id in enumerate(gt_ids, start=1):
        binary_mask = (gt_mask == obj_id).astype(np.uint8)
        gt_annotations.append(
            mask_to_coco_format(binary_mask, image_id, category_id, annotation_id)
        )

    # Create prediction annotations
    for annotation_id, obj_id in enumerate(pred_ids, start=1):
        binary_mask = (pred_mask == obj_id).astype(np.uint8)
        pred_annotations.append(
            mask_to_coco_format(binary_mask, image_id, category_id, annotation_id)
        )

    # COCO GT data structure
    gt_data = {
        "images": [{
            "id": image_id,
            "width": gt_mask.shape[1],
            "height": gt_mask.shape[0],
            "file_name": os.path.basename(image_path)
        }],
        "annotations": gt_annotations,
        "categories": [{"id": category_id, "name": "object"}]
    }

    return gt_data, pred_annotations

def compute_specific_metrics(coco_eval, max_dets=200):
    # Adjust parameters as needed
    coco_eval.params.maxDets = [max_dets, max_dets, max_dets]
    # Set specific IoU thresholds
    coco_eval.params.iouThrs = [0.5, 0.75, 0.95]

    # Suppress output during evaluation
    saved_stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w')
    try:
        # Run evaluation
        coco_eval.evaluate()
        coco_eval.accumulate()
    finally:
        # Restore standard output
        sys.stdout.close()
        sys.stdout = saved_stdout

    # Extract precision and recall arrays after accumulation
    precision = coco_eval.eval['precision']  # shape: [T, R, K, A, M]
    recall = coco_eval.eval['recall']        # shape: [T, K, A, M]

    # Find indices for specific IoU thresholds
    iou_thr_indices = {
        0.5: np.where(np.isclose(coco_eval.params.iouThrs, 0.5))[0][0],
        0.75: np.where(np.isclose(coco_eval.params.iouThrs, 0.75))[0][0],
        0.95: np.where(np.isclose(coco_eval.params.iouThrs, 0.95))[0][0],
    }
    max_det_index = 0  # because we set maxDets list to have identical values

    # Compute AP for each IoU threshold over all areas
    ap_values = {}
    for thr, idx in iou_thr_indices.items():
        valid_precisions = precision[idx, :, :, :, max_det_index]
        valid_precisions = valid_precisions[valid_precisions != -1]  # Exclude invalid entries
        if valid_precisions.size > 0:
            ap = np.mean(valid_precisions)
        else:
            ap = float('nan')  # No valid precision values
        ap_values[f"AP@{int(thr*100)}"] = ap

    # Compute AP across IoU thresholds for all areas (AP@50:95)
    all_valid_precisions = precision[:, :, :, :, max_det_index]
    all_valid_precisions = all_valid_precisions[all_valid_precisions != -1]
    if all_valid_precisions.size > 0:
        ap_values["AP@50:95"] = np.mean(all_valid_precisions)
    else:
        ap_values["AP@50:95"] = float('nan')

    # Compute AR for each IoU threshold over all areas
    ar_values = {}
    for thr, idx in iou_thr_indices.items():
        valid_recalls = recall[idx, :, :, max_det_index]
        if valid_recalls.size > 0:
            ar = np.mean(valid_recalls)  # Recall values are valid; no need to filter
        else:
            ar = float('nan')
        ar_values[f"AR@{int(thr*100)}"] = ar

    # Compute AR across IoU thresholds for all areas (AR@50:95)
    all_valid_recalls = recall[:, :, :, max_det_index]
    if all_valid_recalls.size > 0:
        ar_values["AR@50:95"] = np.mean(all_valid_recalls)
    else:
        ar_values["AR@50:95"] = float('nan')

    # Combine AP and AR metrics into a single dictionary to return
    metrics = {}
    metrics.update(ap_values)
    metrics.update(ar_values)
    metrics["maxDets"] = max_dets  # include maxDets for reference if needed

    return metrics

def evaluate_coco_metrics(gt_data, pred_data, iou_type="segm", max_dets=200):
    """
    Evaluate COCO metrics (AP, AR) given ground truth data and predicted data
    in COCO format. Returns a dictionary of metrics.
    """
    # Generate a unique ID for the temporary file names
    unique_id = uuid.uuid4().hex
    gt_filename = f"temp_gt_{unique_id}.json"
    pred_filename = f"temp_pred_{unique_id}.json"
    
    # Write the ground truth data to a unique temporary file
    with open(gt_filename, "w") as gt_file:
        json.dump(gt_data, gt_file)

    # Convert predictions to COCO format and write them to a unique temporary file
    pred_coco_format = []
    for pred in pred_data:
        # Ensure each prediction has a score; adjust for bbox if necessary
        pred_entry = {
            "image_id": pred["image_id"],
            "category_id": pred["category_id"],
            "score": pred.get("score", 1.0)  # default score 1.0 if missing
        }
        # Add segmentation or bbox depending on iou_type
        if iou_type == "segm":
            pred_entry["segmentation"] = pred["segmentation"]
        elif iou_type == "bbox":
            pred_entry["bbox"] = pred["bbox"]
        pred_coco_format.append(pred_entry)

    with open(pred_filename, "w") as pred_file:
        json.dump(pred_coco_format, pred_file)

    # Load ground truth and predictions into COCO API
    coco_gt = COCO(gt_filename)
    coco_pred = coco_gt.loadRes(pred_filename)
    coco_eval = COCOeval(coco_gt, coco_pred, iouType=iou_type)

    # Compute and retrieve metrics
    metrics = compute_specific_metrics(coco_eval, max_dets=max_dets)

    # Clean up the temporary files
    os.remove(gt_filename)
    os.remove(pred_filename)

    return metrics


def combine_masks_16bit(list_of_binary_masks, output_path=None, return_array=False):
    """
    Combine a list of 0/1 or 0/255 binary masks into a single 16-bit instance mask.
    Each mask is assigned a unique ID (1, 2, 3, ...).

    Parameters:
    - list_of_binary_masks (List[np.ndarray]): List of 0/1 or 0/255 binary masks (numpy arrays).
    - output_path (str, optional): Path to save the combined 16-bit PNG file.
    - return_array (bool, optional): If True, returns the combined 16-bit mask array.

    Returns:
    - combined_16bit (np.ndarray, optional): The combined 16-bit instance mask array if return_array is True.
    """
    if not list_of_binary_masks:
        print("[WARNING] No masks provided to combine.")
        return None

    # Verify that all masks have the same shape
    first_shape = list_of_binary_masks[0].shape
    for idx, mask in enumerate(list_of_binary_masks):
        if mask.shape != first_shape:
            raise ValueError(f"All masks must have the same shape. Mask at index {idx} has shape {mask.shape}, expected {first_shape}.")

    height, width = first_shape
    combined_16bit = np.zeros((height, width), dtype=np.uint16)

    for idx, mask in enumerate(list_of_binary_masks, start=1):
        # Debug: Print unique values in the mask
        unique_vals = set(np.unique(mask))
        # print(f"Mask {idx-1} unique values: {unique_vals}")

        # Ensure mask is binary
        if mask.dtype != bool and not (unique_vals <= {0, 1} or unique_vals <= {0, 255}):
            raise ValueError(f"Mask at index {idx-1} has unexpected unique values {unique_vals}. Please provide 0/1 or 0/255 masks.")

        # Convert mask to boolean
        mask_bool = mask.astype(bool)

        # Assign unique ID to each mask
        combined_16bit[mask_bool] = idx

    # Save the combined mask as a 16-bit PNG if output_path is provided
    if output_path:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # OpenCV expects the image in the correct format
        # Ensure that combined_16bit is in uint16
        if combined_16bit.dtype != np.uint16:
            combined_16bit = combined_16bit.astype(np.uint16)

        success = cv2.imwrite(output_path, combined_16bit)
        if not success:
            raise IOError(f"Failed to write the combined mask to {output_path}")

    if return_array:
        return combined_16bit

    return None


def generate_coco_annotations_from_multi_instance_masks_16bit(
    gt_mask_path,
    pred_mask_path,
    image_path,
    image_id=1,
    category_id=1
):
    """
    Generate COCO-style annotations for both ground truth and predicted masks.
    """
    # Read masks preserving the original bit depth (e.g., 16-bit)
    gt_mask = cv2.imread(gt_mask_path, cv2.IMREAD_UNCHANGED)
    pred_mask = cv2.imread(pred_mask_path, cv2.IMREAD_UNCHANGED)

    if gt_mask is None or pred_mask is None:
        raise FileNotFoundError("One of the mask files was not found.")

    # Identify unique object IDs (works correctly for >255 values)
    gt_ids = np.unique(gt_mask)
    pred_ids = np.unique(pred_mask)
    gt_ids = gt_ids[gt_ids > 0]
    pred_ids = pred_ids[pred_ids > 0]

    gt_annotations = []
    pred_annotations = []

    # Create GT annotations
    for annotation_id, obj_id in enumerate(gt_ids, start=1):
        binary_mask = (gt_mask == obj_id).astype(np.uint8)
        gt_annotations.append(
            mask_to_coco_format(binary_mask, image_id, category_id, annotation_id)
        )

    # Create prediction annotations
    for annotation_id, obj_id in enumerate(pred_ids, start=1):
        binary_mask = (pred_mask == obj_id).astype(np.uint8)
        pred_annotations.append(
            mask_to_coco_format(binary_mask, image_id, category_id, annotation_id)
        )

    # COCO GT data structure
    gt_data = {
        "info": {
            "description": "auto-generated dataset",
            "version": "1.0"
        },
        "images": [{
            "id": image_id,
            "width": int(gt_mask.shape[1]),
            "height": int(gt_mask.shape[0]),
            "file_name": os.path.basename(image_path)
        }],
        "annotations": gt_annotations,
        "categories": [{"id": category_id, "name": "object"}]
    }

    return gt_data, pred_annotations

def compute_pixel_precision_recall(gt_path, pred_path):
    """
    Compute pixel-level precision and recall by comparing a predicted mask
    with the ground truth mask.
    """

    # Load images as grayscale
    gt = np.array(Image.open(gt_path).convert("L"))
    pred = np.array(Image.open(pred_path).convert("L"))

    # Convert to binary masks: non-zero indicates "Not Background"
    gt_binary = (gt > 127)
    pred_binary = (pred > 127)

    TP = np.sum(np.logical_and(pred_binary, gt_binary))
    FP = np.sum(np.logical_and(pred_binary, np.logical_not(gt_binary)))
    FN = np.sum(np.logical_and(np.logical_not(pred_binary), gt_binary))

    precision = TP / (TP + FP) if (TP + FP) > 0 else None
    recall = TP / (TP + FN) if (TP + FN) > 0 else None
    return precision, recall
