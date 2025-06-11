import hashlib
import os
import pandas as pd

def compute_hash(path):
    return hashlib.md5(path.encode()).hexdigest()

def load_cached_features(image_path, cache_dir):
    h = compute_hash(image_path)
    cache_file = os.path.join(cache_dir, f"{h}.feather")
    if os.path.exists(cache_file):
        return pd.read_feather(cache_file)
    return None

def save_cached_features(image_path, features, cache_dir):
    h = compute_hash(image_path)
    cache_file = os.path.join(cache_dir, f"{h}.feather")
    pd.DataFrame([features]).to_feather(cache_file)
