import os
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset
from utils.preprocessing import resize_image_adaptive, set_target_image_size
from utils.feature_registry import FEATURE_REGISTRY
from PIL import Image
import cv2

def preprocess_pil_image(pil_img, target_size=512):
    # Convert to grayscale
    img = pil_img.convert("L")
    
    # Convert to numpy array and normalize
    img_np = np.array(img).astype(np.float32) / 255.0

    # Resize using adaptive strategy (e.g., cv2.INTER_AREA for downsampling, INTER_CUBIC for up)
    if img_np.shape[0] > target_size or img_np.shape[1] > target_size:
        interp = cv2.INTER_AREA
    else:
        interp = cv2.INTER_CUBIC

    img_resized = cv2.resize(img_np, (target_size, target_size), interpolation=interp)

    return img_resized


def compute_features_batch(batch, selected_features, target_size):
    feature_rows = []

    for i, pil_img in enumerate(batch["png"]):
        try:
            img = preprocess_pil_image(pil_img, target_size=target_size)

            shared_data = {}
            feats = {name: FEATURE_REGISTRY[name](img, shared=shared_data) for name in selected_features}

            # Add metadata
            feats["label"] = "Fake"
            feats["model.txt"] = batch["model.txt"][i]
            feats["prompt.cls"] = batch["prompt.cls"][i]

            feature_rows.append(feats)
        except Exception as e:
            print(f"[WARNING] Skipping image due to error: {e}")
    
    return feature_rows



def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"[INFO] Loading DRAGON dataset subset='{args.subset}', split='{args.split}' with streaming...")
    ds = load_dataset("lesc-unifi/dragon", args.subset, split=args.split, streaming=True)

    if args.limit:
        ds = ds.take(args.limit)

    ds = ds.shuffle(buffer_size=1000, seed=42)
    ds = ds.batch(args.batch_size)

    output_path = os.path.join(args.output_dir, "dragon_features_with_metadata.csv")

    with open(output_path, "w") as f_out:
        header_written = False
        for batch in tqdm(ds, desc="Streaming batches"):
            batch_features = compute_features_batch(batch, args.features, args.target_size)
            if not batch_features:
                continue

            df_batch = pd.DataFrame(batch_features)

            if not header_written:
                df_batch.to_csv(f_out, index=False, header=True)
                header_written = True
            else:
                df_batch.to_csv(f_out, index=False, header=False)

    print(f"[INFO] Saved features and metadata to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", type=str, default="large", help="Subset of DRAGON dataset to use")
    parser.add_argument("--split", type=str, default="train", help="Data split (train/validation/test)")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for streaming")
    parser.add_argument("--features", nargs="+", required=True, help="List of feature names to extract")
    parser.add_argument("--output_dir", type=str, default="dragon_features", help="Directory to save feature CSV")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit on number of images")
    parser.add_argument("--target_size", type=int, default=512, help="Target size for resized images")

    args = parser.parse_args()
    main(args)
