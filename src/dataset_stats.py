import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from collections import Counter
from tqdm import tqdm

# ---------------- CONFIG ----------------
REAL_DIR = './data/real'
FAKE_METADATA_CSV = './data/fake/fake_metadata.csv'
SPLIT_FILE = './models/split_info.csv'


# ---------------- UTILITIES ----------------
def get_image_size_dict(image_dir):
    size_map = {}
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    for file in tqdm(image_files, desc=f"Scanning {os.path.basename(image_dir)}"):
        try:
            with Image.open(os.path.join(image_dir, file)) as img:
                w, h = img.size
                size_map[file] = (w, h)
        except Exception as e:
            print(f"Error reading {file}: {e}")
    return size_map

def analyze_fake_image_sizes():
    print("\n--- Analyzing FAKE image sizes ---")
    size_map = get_image_size_dict("./data/fake")

    # Convert to DataFrame
    size_df = pd.DataFrame.from_dict(size_map, orient='index', columns=['width', 'height'])
    size_df.reset_index(inplace=True)
    size_df.rename(columns={'index': 'filename'}, inplace=True)

    # Join with metadata
    metadata = pd.read_csv(FAKE_METADATA_CSV)
    merged = pd.merge(metadata, size_df, on='filename', how='inner')

    # Global stats for all fake images
    widths = merged['width'].values
    heights = merged['height'].values
    sizes = list(zip(widths, heights))
    describe_dimensions(widths, heights, sizes, label="Fake Images")
    plot_histograms(widths, heights, label="Fake Images")

    # Per-model stats
    print("\n--- Per-Model Size Stats ---")
    for model, group in merged.groupby("model"):
        w = group['width'].values
        h = group['height'].values
        s = list(zip(w, h))
        print(f"\n[Model: {model}]")
        describe_dimensions(w, h, s, label=f"Fake ({model})")

def get_image_sizes(image_dir):
    widths, heights, sizes = [], [], []
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    for file in tqdm(image_files, desc=f"Scanning {os.path.basename(image_dir)}"):
        try:
            with Image.open(os.path.join(image_dir, file)) as img:
                w, h = img.size
                widths.append(w)
                heights.append(h)
                sizes.append((w, h))
        except Exception as e:
            print(f"Error reading {file}: {e}")

    return np.array(widths), np.array(heights), sizes


def describe_dimensions(widths, heights, sizes, label="Dataset"):
    print(f"\n--- Stats for {label} ---")
    print(f"Total images analyzed: {len(widths)}")
    
    for t in [256, 512, 768, 1024]:
        count = np.sum((widths >= t) | (heights >= t))
        print(f" > {t}px: {count} images ({(count / len(widths)) * 100:.2f}%)")

    print(f"Smallest image: {np.min(widths)} x {np.min(heights)}")
    print(f"Largest image:  {np.max(widths)} x {np.max(heights)}")

    size_counts = Counter(sizes)
    most_common_size, count = size_counts.most_common(1)[0]
    print(f"Most common size: {most_common_size[0]} x {most_common_size[1]} ({count} images)")
    print(f"Most common width:  {Counter(widths.tolist()).most_common(1)[0][0]}")
    print(f"Most common height: {Counter(heights.tolist()).most_common(1)[0][0]}")


def plot_histograms(widths, heights, label="Dataset"):
    width_range = np.percentile(widths, [1, 99])
    height_range = np.percentile(heights, [1, 99])

    plt.figure(figsize=(12, 5))
    plt.suptitle(f"{label} Image Size Distributions", fontsize=14)

    plt.subplot(1, 2, 1)
    plt.hist(widths[(widths >= width_range[0]) & (widths <= width_range[1])], bins=50, color='skyblue')
    plt.title("Width Distribution (1–99%)")
    plt.xlabel("Width (pixels)")
    plt.ylabel("Count")

    plt.subplot(1, 2, 2)
    plt.hist(heights[(heights >= height_range[0]) & (heights <= height_range[1])], bins=50, color='salmon')
    plt.title("Height Distribution (1–99%)")
    plt.xlabel("Height (pixels)")
    plt.ylabel("Count")

    plt.tight_layout()
    plt.show()


# ---------------- MAIN ----------------

def analyze_real_images():
    widths, heights, sizes = get_image_sizes(REAL_DIR)
    describe_dimensions(widths, heights, sizes, label="Real Images")
    plot_histograms(widths, heights, label="Real Images")


def analyze_fake_metadata():
    df = pd.read_csv(FAKE_METADATA_CSV)
    model_counts = df['model'].value_counts().sort_values(ascending=False)

    print("\n--- Fake Image Count by Generation Model ---\n")
    print(model_counts)

    model_counts.plot(kind='bar', figsize=(12, 5), title="Image Count by Generation Model")
    plt.xlabel("Model")
    plt.ylabel("Number of Images")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()


def analyze_split_stats(split_path=SPLIT_FILE, real_dir=REAL_DIR):
    if not os.path.exists(split_path):
        print("\nSplit file not found. Skipping split-wise analysis.")
        return

    split_info = pd.read_csv(split_path)
    for split in ["train", "test"]:
        filenames = split_info[split_info["split"] == split]["filename"].values
        widths, heights, sizes = [], [], []
        for fname in tqdm(filenames, desc=f"Scanning {split} split"):
            try:
                with Image.open(os.path.join(real_dir, fname)) as img:
                    w, h = img.size
                    widths.append(w)
                    heights.append(h)
                    sizes.append((w, h))
            except Exception as e:
                print(f"[{split}] Error reading {fname}: {e}")

        widths, heights = np.array(widths), np.array(heights)
        describe_dimensions(widths, heights, sizes, label=f"Real ({split})")
        plot_histograms(widths, heights, label=f"Real ({split})")


if __name__ == "__main__":
    analyze_fake_metadata()
    analyze_real_images()
    analyze_fake_image_sizes()
    analyze_split_stats()
