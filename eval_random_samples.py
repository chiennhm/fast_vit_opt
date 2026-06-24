import argparse
import os
import random
import torch
from torch.utils.data import DataLoader, Subset

from detection.visualize import save_detection_results, VOC_CLASSES
from coco_dataset import build_coco_datasets, coco_collate, COCO_CLASSES
from voc_dataset import build_voc_datasets, detection_collate
from detection.maskrcnn_detector import FastViTMaskRCNN
from detection.fastvit_detector import FastViTDetector

def main():
    parser = argparse.ArgumentParser(description="Evaluate on random samples with a specific threshold")
    parser.add_argument("--dataset", type=str, default="coco", choices=["voc", "coco"])
    parser.add_argument("--data-dir", type=str, default="./data/coco")
    parser.add_argument("--arch", type=str, default="maskrcnn", choices=["fastvit", "maskrcnn"])
    parser.add_argument("--model", type=str, default="fastvit_sa12")
    parser.add_argument("--checkpoint", type=str, default="best.pth")
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--score-thresh", type=float, default=0.5, help="Threshold for visualization and prediction")
    parser.add_argument("--output-dir", type=str, default="./output/random_eval")
    args = parser.parse_args()

    # Load dataset
    print("Loading dataset...")
    if args.dataset == "coco":
        _, val_dataset = build_coco_datasets(data_dir=args.data_dir, img_size=800)
        collate_fn = coco_collate
        num_classes = 80
        class_names = COCO_CLASSES
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
    print("Building model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.arch == "maskrcnn":
        model = FastViTMaskRCNN(
            num_classes=num_classes,
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
            k: v for k, v in state_dict.items()
            if k in model_dict and v.shape == model_dict[k].shape
        }
        model.load_state_dict(filtered, strict=False)
        print(f"Loaded checkpoint from {args.checkpoint} ({len(filtered)}/{len(model_dict)} keys matched)")
    else:
        print(f"Warning: Checkpoint {args.checkpoint} not found. Using untrained weights.")

    model.to(device)
    model.eval()

    gt_dir = os.path.join(args.output_dir, "ground_truth")
    pred_dir = os.path.join(args.output_dir, "predicts")
    os.makedirs(gt_dir, exist_ok=True)
    os.makedirs(pred_dir, exist_ok=True)

    print(f"Evaluating {args.num_samples} random samples with threshold {args.score_thresh}...")
    
    with torch.inference_mode():
        for i, (images, targets) in enumerate(dataloader):
            images = images.to(device)
            
            # Predict
            predictions = model.predict(
                images, 
                score_thresh=args.score_thresh, 
                nms_thresh=0.5, 
                max_detections=100
            )

            # Convert predictions to CPU for visualization
            for pred in predictions:
                for k, v in pred.items():
                    if isinstance(v, torch.Tensor):
                        pred[k] = v.cpu()

            # Prepare ground truth for visualization
            gt_list = []
            for t in targets:
                gt_list.append({
                    "boxes": t["boxes"].cpu(),
                    "labels": t["labels"].cpu()
                })

            # Save prediction visualization
            save_detection_results(
                images,
                predictions,
                pred_dir,
                class_names=class_names,
                score_thresh=args.score_thresh,
            )
            
            # Rename the saved pred file
            old_pred_name = os.path.join(pred_dir, "det_0000.jpg")
            new_pred_name = os.path.join(pred_dir, f"random_sample_{i+1}.jpg")
            if os.path.exists(old_pred_name):
                if os.path.exists(new_pred_name): os.remove(new_pred_name)
                os.rename(old_pred_name, new_pred_name)

            # Save ground truth visualization
            save_detection_results(
                images,
                gt_list,
                gt_dir,
                class_names=class_names,
                score_thresh=0.0,
            )
            
            # Rename the saved gt file
            old_gt_name = os.path.join(gt_dir, "det_0000.jpg")
            new_gt_name = os.path.join(gt_dir, f"random_sample_{i+1}.jpg")
            if os.path.exists(old_gt_name):
                if os.path.exists(new_gt_name): os.remove(new_gt_name)
                os.rename(old_gt_name, new_gt_name)
                
    print(f"Done! Ground truth saved to {gt_dir}")
    print(f"Done! Predictions saved to {pred_dir}")

if __name__ == '__main__':
    main()
