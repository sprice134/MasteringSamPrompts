import os
import cv2
import glob
import pickle
import torch
import argparse
import numpy as np
import sys
# Replace these imports with your actual modules
from models import load_trained_model, get_inference_predictions

"""
    python model_inference.py \
    --model "yolov8n-seg" \
    --model-type "yolo" \
    --model-path "powder_yolov8n-seg.pt" \
    --image-dir "../datasets/powder/test" \
    --output-file "inference_outputs/powder_yolov8n_inference.pkl" \
    --device "cuda"

"""


def _draw_visualisation(image_bgr, boxes, polygons):
    """Draws bounding‑boxes and polygons on a copy of the image and returns it."""
    vis_img = image_bgr.copy()

    # Draw boxes (green)
    if boxes is not None and len(boxes):
        if torch.is_tensor(boxes):
            boxes = boxes.cpu().numpy()
        for box in boxes:
            x1, y1, x2, y2 = map(int, box[:4])
            cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Draw polygons (red)
    if polygons:
        for poly in polygons:
            pts = np.array(poly, dtype=np.int32).reshape(-1, 2)
            cv2.polylines(vis_img, [pts], isClosed=True, color=(0, 0, 255), thickness=2)
    return vis_img


def run_inference_for_model(model_type, model_name, model_path, image_dir, output_file, device="cuda"):
    """
    Runs inference for a single model on all .jpg/.png images in `image_dir`,
    then saves a pickle with bounding boxes, polygons, and masks.
    Additionally, for the **first image processed**, a visualisation with
    detections over‑laid is saved to `output.png` in the current directory.
    """

    # 1) Gather images
    image_paths = glob.glob(os.path.join(image_dir, "*.jpg")) + glob.glob(os.path.join(image_dir, "*.png"))
    if not image_paths:
        print(f"[ERROR] No .jpg/.png images found in {image_dir}")
        return

    # 2) Load the model
    print(f"[INFO] Loading model: {model_name} ({model_type}) from {model_path}")
    model = load_trained_model(model_type, model_path, device=device)

    # 3) Prepare dictionary to store results
    inference_data = {}

    # Flag to control saving exactly one visualisation
    saved_vis = False

    for img_path in image_paths:
        img_name = os.path.basename(img_path)
        image_bgr = cv2.imread(img_path)
        if image_bgr is None:
            print(f"[WARNING] Could not read {img_path}. Skipping...")
            continue

        # 4) Run inference
        listOfPolygons, listOfBoxes, listOfMasks = get_inference_predictions(
            model=model,
            model_type=model_type,
            image_path=img_path,
            device=device
        )

        # Ensure boxes on CPU & NumPy for pickling / logging convenience
        if torch.is_tensor(listOfBoxes):
            listOfBoxes = listOfBoxes.cpu().numpy()

        # 5) Optionally save a visualisation for the very first image
        if not saved_vis:
            vis_img = _draw_visualisation(image_bgr, listOfBoxes, listOfPolygons)
            cv2.imwrite("output.png", vis_img)
            print("[INFO] Saved visualisation of first image to output.png")
            saved_vis = True

        # 6) Save prediction tensors/masks/polygons for current image
        inference_data[img_name] = {
            "polygons": listOfPolygons,
            "boxes": listOfBoxes,
            "masks": listOfMasks
        }
        print("Number of detections:", len(listOfBoxes))

    # 7) Persist pickle to disk
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "wb") as f:
        pickle.dump(inference_data, f)

    print(f"[INFO] Inference completed for {model_name}. Saved results to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference for a single model and save pickle + visualisation.")
    parser.add_argument("--model", type=str, required=True, help="Model name (e.g. 'yolov8n')")
    parser.add_argument("--model-type", type=str, default="yolo", help="Model type (e.g. 'yolo' or 'maskrcnn')")
    parser.add_argument("--model-path", type=str, required=True, help="Path to model weights")
    parser.add_argument("--image-dir", type=str, required=True, help="Directory of .jpg/.png images")
    parser.add_argument("--output-file", type=str, default="inference_outputs/model_inference.pkl",
                        help="Where to save the pickle file")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run inference on")

    args = parser.parse_args()

    run_inference_for_model(
        model_type=args.model_type,
        model_name=args.model,
        model_path=args.model_path,
        image_dir=args.image_dir,
        output_file=args.output_file,
        device=args.device
    )
