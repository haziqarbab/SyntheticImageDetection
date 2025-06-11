import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from src.utils.io import load_cached_features, save_cached_features
from src.utils.preprocessing import preprocess_image
from src.utils.feature_registry import FEATURE_REGISTRY

# --------------------------- CONFIG ---------------------------
DEFAULT_CACHE_DIR = "src/cache"

def extract_features_from_image(img_path, selected_features, cache_dir=DEFAULT_CACHE_DIR):
    # Attempt to load cached features
    cached = load_cached_features(img_path, cache_dir)
    if cached is not None:
        cached_feats = cached.to_dict(orient="records")[0]
        # Check if cache is missing any selected features
        if all(f in cached_feats for f in selected_features):
            return {f: cached_feats[f] for f in selected_features}

    # Load and preprocess image
    img = preprocess_image(img_path)
    
    # Extract and cache
    result = {}
    shared_data = {}  # Shared buffers for reuse (e.g. FFT, residuals, etc)
    for feat_name in selected_features:
        func = FEATURE_REGISTRY[feat_name]
        result[feat_name] = func(img, shared=shared_data)

    # Save complete set to cache
    all_feats = cached_feats if cached is not None else {}
    all_feats.update(result)
    save_cached_features(img_path, all_feats, cache_dir)
    return result

def extract_dataset_features(real_dir,
                             fake_dir,
                             selected_features,
                             max_images=None,
                             cache_dir=DEFAULT_CACHE_DIR,
                             selected_models=None,
                             metadata_path=None):
    
    rows = []

    # --- Filter Fake Images by Model if specified ---
    if selected_models and metadata_path:
        meta = pd.read_csv(metadata_path)

        # Sanity check: models in metadata
        available_models = sorted(meta['model'].unique())
        print(f"[INFO] Available generative models in metadata: {available_models}")

        filtered_meta = meta[meta['model'].isin(selected_models)]
        selected_in_metadata = sorted(filtered_meta['model'].unique())
        print(f"[INFO] Selected models used for filtering: {selected_in_metadata}")

        if filtered_meta.empty:
            raise ValueError(f"No fake images found for selected models: {selected_models}")

        if max_images:
            filtered_meta = (
                filtered_meta.groupby("model")
                .apply(lambda x: x.sample(n=min(len(x), max_images), random_state=42), include_groups=False)
                .reset_index(drop=True)
            )

        fake_files = filtered_meta["filename"].tolist()
    elif metadata_path:
        meta = pd.read_csv(metadata_path)
        available_models = sorted(meta['model'].unique())
        print(f"[INFO] Using data from generative models: {available_models}")
        fake_files = sorted(os.listdir(fake_dir))[:max_images]
    else:
        fake_files = sorted(os.listdir(fake_dir))[:max_images]

    # --- Select Real Images ---
    real_files = sorted(os.listdir(real_dir))[:len(fake_files)]

    print(f"[INFO] Using {len(real_files)} real and {len(fake_files)} fake images.")

    # --- Extract Features ---
    for label, files, folder in [("Real", real_files, real_dir), ("Fake", fake_files, fake_dir)]:
        for fname in tqdm(files, desc=f"Processing {label}"):
            path = os.path.join(folder, fname)
            try:
                feats = extract_features_from_image(path, selected_features, cache_dir)
                feats["label"] = label
                feats["filename"] = fname
                rows.append(feats)
            except Exception as e:
                print(f"[ERROR] {fname}: {e}")

    return pd.DataFrame(rows)
