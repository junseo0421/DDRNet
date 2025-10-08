import os
import argparse
import numpy as np
from PIL import Image
from glob import glob
from tqdm import tqdm
from sklearn.metrics import confusion_matrix

def load_image(path):
    return np.array(Image.open(path)).astype(np.uint8)

def compute_miou(confusion, num_classes):
    ious = []
    for cls in range(num_classes):
        TP = confusion[cls, cls]
        FP = confusion[:, cls].sum() - TP
        FN = confusion[cls, :].sum() - TP
        denom = TP + FP + FN
        if denom == 0:
            iou = float('nan')
        else:
            iou = TP / denom
        ious.append(iou)
    miou = np.nanmean(ious)
    return miou, ious

def evaluate(result_dir, label_dir, num_classes):
    pred_paths = sorted(glob(os.path.join(result_dir, "*_leftImg8bit.png")))
    print(f'Found {len(pred_paths)} segmentation result images')
    all_confusion = np.zeros((num_classes, num_classes), dtype=np.int64)

    for pred_path in tqdm(pred_paths, desc="Evaluating"):
        file_id = os.path.basename(pred_path).replace("_leftImg8bit.png", "")
        label_path = os.path.join(label_dir, f"{file_id}_gtFine_CategoryId.png")

        if not os.path.exists(label_path):
            print(f"Label not found for {file_id}, skipping.")
            continue

        pred = load_image(pred_path).flatten()
        label = load_image(label_path).flatten()

        # Mask out invalid pixels (e.g., 255 = ignore)
        mask = label != 255
        pred = pred[mask]
        label = label[mask]

        conf = confusion_matrix(label, pred, labels=list(range(num_classes)))
        all_confusion += conf

    miou, ious = compute_miou(all_confusion, num_classes)
    print(f"\n📊 mIoU: {miou:.4f}")
    for i, iou in enumerate(ious):
        print(f"Class {i}: IoU = {iou:.4f}" if not np.isnan(iou) else f"Class {i}: IoU = NaN (ignored)")

    return miou, ious

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_dir", type=str, default="/home/user/pythonCodes/25_Sechal/distribution/result/image/test/set1",  help="Directory with predicted *_img.png files")
    parser.add_argument("--label_dir", type=str, default="/home/user/ext_hdd/dataset/SeChal_2025/SemanticDatasetTestGT_final/labelmap/test/set1",  help="Directory with ground-truth *_label.png files")
    parser.add_argument("--num_classes", type=int, default=19, help="Number of segmentation classes")

    args = parser.parse_args()
    evaluate(args.result_dir, args.label_dir, args.num_classes)