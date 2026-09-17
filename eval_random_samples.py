import argparse
import os
import random
import torch
from torch.utils.data import DataLoader, Subset
from PIL import Image, ImageDraw, ImageFont

from detection.visualize import save_detection_results, draw_detections, VOC_CLASSES
from coco_dataset import build_coco_datasets, coco_collate, COCO_CLASSES
from voc_dataset import build_voc_datasets, detection_collate
from bdd100k_dataset import build_bdd100k_datasets, bdd100k_collate, BDD100K_CLASSES
from detection.maskrcnn_detector import FastViTMaskRCNN
from detection.fastvit_detector import FastViTDetector
from detection.gradcam import GradCAM, overlay_gradcam


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate on random samples with GT, Prediction, and Grad-CAM visualization"
    )
    parser.add_argument(
        "--dataset", type=str, default="coco", choices=["voc", "coco", "bdd100k"]
    )
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument(
        "--arch", type=str, default="fastvit", choices=["fastvit", "maskrcnn"]
    )
    parser.add_argument("--model", type=str, default="fastvit_sa12")
    parser.add_argument("--checkpoint", type=str, default="best.pth")
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument(
        "--score-thresh",
        type=float,
        default=0.3,
        help="Threshold for visualization and prediction",
    )
    parser.add_argument("--output-dir", type=str, default="./output/random_eval")
    args = parser.parse_args()

    # Default data directory per dataset
    if args.data_dir is None:
        if args.dataset == "coco":
            args.data_dir = "./data/coco"
        elif args.dataset == "bdd100k":
            args.data_dir = "./data/bdd100k"
        else:
            args.data_dir = "./data"

    # Load dataset
    print(f"Loading {args.dataset.upper()} dataset from {args.data_dir}...")
    if args.dataset == "coco":
        _, val_dataset = build_coco_datasets(data_dir=args.data_dir, img_size=800)
        collate_fn = coco_collate
        num_classes = 80
        class_names = COCO_CLASSES
    elif args.dataset == "bdd100k":
        _, val_dataset = build_bdd100k_datasets(data_dir=args.data_dir, img_size=800)
        collate_fn = bdd100k_collate
        num_classes = 10
        class_names = BDD100K_CLASSES
    else:
        _, val_dataset = build_voc_datasets(data_dir=args.data_dir, img_size=800)
        collate_fn = detection_collate
        num_classes = 20
        class_names = VOC_CLASSES

    # Select random indices
    if args.num_samples > len(val_dataset):
        args.num_samples = len(val_dataset)
    indices = random.sample(range(len(val_dataset)), args.num_samples)
    subset = Subset(val_dataset, indices)
    dataloader = DataLoader(subset, batch_size=1, shuffle=False, collate_fn=collate_fn)

    # Load model
    print(f"Building model ({args.arch} / {args.model})...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.arch == "maskrcnn":
        model = FastViTMaskRCNN(
            num_classes=num_classes + 1,
        )
    else:
        model = FastViTDetector(
            model_name=args.model,
            num_classes=num_classes,
        )

    if os.path.exists(args.checkpoint):
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint

        # Scrub incompatible keys
        model_dict = model.state_dict()
        filtered = {
            k: v
            for k, v in state_dict.items()
            if k in model_dict and v.shape == model_dict[k].shape
        }
        model.load_state_dict(filtered, strict=False)
        print(
            f"Loaded checkpoint from {args.checkpoint} ({len(filtered)}/{len(model_dict)} keys matched)"
        )
    else:
        print(
            f"Warning: Checkpoint {args.checkpoint} not found. Using untrained weights."
        )

    model.to(device)
    model.eval()

    # Output directories
    gt_dir = os.path.join(args.output_dir, "ground_truth")
    pred_dir = os.path.join(args.output_dir, "predicts")
    cam_dir = os.path.join(args.output_dir, "gradcam")
    os.makedirs(gt_dir, exist_ok=True)
    os.makedirs(pred_dir, exist_ok=True)
    os.makedirs(cam_dir, exist_ok=True)

    # Setup Grad-CAM generator
    gradcam_generator = GradCAM(model)

    print(
        f"Evaluating {args.num_samples} random samples with score threshold {args.score_thresh}..."
    )

    for i, (images, targets) in enumerate(dataloader):
        images = images.to(device)

        # 1. Predict
        with torch.inference_mode():
            predictions = model.predict(
                images,
                score_thresh=args.score_thresh,
                nms_thresh=0.5,
                max_detections=100,
            )

        # 2. Generate Grad-CAM Heatmap
        heatmap_rgb = gradcam_generator.generate_heatmap(images)
        cam_overlay_pil = overlay_gradcam(images[0], heatmap_rgb, alpha=0.5)

        # Save Grad-CAM image
        cam_save_path = os.path.join(cam_dir, f"sample_{i+1}.jpg")
        cam_overlay_pil.save(cam_save_path)

        # Convert predictions to CPU for visualization
        for pred in predictions:
            for k, v in pred.items():
                if isinstance(v, torch.Tensor):
                    pred[k] = v.cpu()

        # Prepare ground truth for visualization
        gt_list = []
        for t in targets:
            gt_list.append({"boxes": t["boxes"].cpu(), "labels": t["labels"].cpu()})

        # Save prediction visualization
        save_detection_results(
            images,
            predictions,
            pred_dir,
            class_names=class_names,
            score_thresh=args.score_thresh,
        )
        old_pred_name = os.path.join(pred_dir, "det_0000.jpg")
        new_pred_name = os.path.join(pred_dir, f"sample_{i+1}.jpg")
        if os.path.exists(old_pred_name):
            if os.path.exists(new_pred_name):
                os.remove(new_pred_name)
            os.rename(old_pred_name, new_pred_name)

        # Save ground truth visualization
        save_detection_results(
            images,
            gt_list,
            gt_dir,
            class_names=class_names,
            score_thresh=0.0,
        )
        old_gt_name = os.path.join(gt_dir, "det_0000.jpg")
        new_gt_name = os.path.join(gt_dir, f"sample_{i+1}.jpg")
        if os.path.exists(old_gt_name):
            if os.path.exists(new_gt_name):
                os.remove(new_gt_name)
            os.rename(old_gt_name, new_gt_name)

        # 3. Concatenate side-by-side [ GROUND TRUTH | PREDICTION | GRAD-CAM ]
        if os.path.exists(new_gt_name) and os.path.exists(new_pred_name):
            img_gt = Image.open(new_gt_name)
            img_pred = Image.open(new_pred_name)
            img_cam = cam_overlay_pil

            w, h = img_gt.size
            header_h = 40
            combined_w = w * 3
            combined_h = h + header_h

            combined = Image.new("RGB", (combined_w, combined_h), (30, 30, 30))

            # Paste 3 panels
            combined.paste(img_gt, (0, header_h))
            combined.paste(img_pred, (w, header_h))
            combined.paste(img_cam.resize((w, h)), (w * 2, header_h))

            # Draw header labels
            draw = ImageDraw.Draw(combined)
            try:
                font = ImageFont.truetype("arial.ttf", 18)
            except (IOError, OSError):
                font = ImageFont.load_default()

            headers = [
                "GROUND TRUTH",
                f"PREDICTION (Score >= {args.score_thresh})",
                "GRAD-CAM (ATTENTION MAP)",
            ]

            for panel_idx, title in enumerate(headers):
                panel_x = panel_idx * w
                text_bbox = draw.textbbox((0, 0), title, font=font)
                text_w = text_bbox[2] - text_bbox[0]
                text_h = text_bbox[3] - text_bbox[1]
                pos_x = panel_x + (w - text_w) // 2
                pos_y = (header_h - text_h) // 2
                draw.text((pos_x, pos_y), title, fill=(255, 255, 255), font=font)

            out_combined_path = os.path.join(args.output_dir, f"side_by_side_{i+1}.jpg")
            combined.save(out_combined_path)

    gradcam_generator.remove_hooks()

    print(f"\nCompleted evaluation & visualization!")
    print(f"  Combined 3-panel images saved to: {args.output_dir}/side_by_side_*.jpg")
    print(f"  Ground truth images saved to:     {gt_dir}")
    print(f"  Prediction images saved to:       {pred_dir}")
    print(f"  Grad-CAM images saved to:         {cam_dir}")


if __name__ == "__main__":
    main()
