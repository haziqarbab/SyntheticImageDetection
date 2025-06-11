import os
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

# Path to your ImageNet val images
imagenet_dir = 'data/real'  # <-- UPDATE THIS

widths = []
heights = []
sizes = []

# Scan all image files
image_files = [f for f in os.listdir(imagenet_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

# Extract dimensions
for file in tqdm(image_files, desc="Analyzing image sizes"):
    try:
        with Image.open(os.path.join(imagenet_dir, file)) as img:
            w, h = img.size
            widths.append(w)
            heights.append(h)
            sizes.append((w, h))
    except Exception as e:
        print(f"Error reading {file}: {e}")

# Convert to NumPy arrays for easier math
widths_np = np.array(widths)
heights_np = np.array(heights)

# Compute percentages
total = len(widths)
thresholds = [256, 512, 768, 1024]
print(f"Total images analyzed: {total}")
print("Percentage of images where width OR height exceeds threshold:")
for t in thresholds:
    count = np.sum((widths_np >= t) | (heights_np >= t))
    print(f" > {t}px: {count} images ({(count / total) * 100:.2f}%)")

# Print min and max sizes
min_w, max_w = np.min(widths_np), np.max(widths_np)
min_h, max_h = np.min(heights_np), np.max(heights_np)
print(f"\nSmallest image: {min_w} x {min_h}")
print(f"Largest image:  {max_w} x {max_h}")

# Most common sizes
size_counts = Counter(sizes)
most_common_size, count = size_counts.most_common(1)[0]
print(f"\nMost common size: {most_common_size[0]} x {most_common_size[1]} ({count} images)")

# Most common width and height
most_common_width, w_count = Counter(widths).most_common(1)[0]
most_common_height, h_count = Counter(heights).most_common(1)[0]
print(f"Most common width:  {most_common_width} ({w_count} images)")
print(f"Most common height: {most_common_height} ({h_count} images)")

# Histogram with outlier filtering (1st–99th percentile)
width_range = np.percentile(widths_np, [1, 99])
height_range = np.percentile(heights_np, [1, 99])

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.hist(widths_np[(widths_np >= width_range[0]) & (widths_np <= width_range[1])], bins=50, color='skyblue')
plt.title("Width Distribution (1–99%)")
plt.xlabel("Width (pixels)")
plt.ylabel("Count")

plt.subplot(1, 2, 2)
plt.hist(heights_np[(heights_np >= height_range[0]) & (heights_np <= height_range[1])], bins=50, color='salmon')
plt.title("Height Distribution (1–99%)")
plt.xlabel("Height (pixels)")
plt.ylabel("Count")

plt.tight_layout()
plt.show()
