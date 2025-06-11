import os
import csv
import shutil
from datasets import load_dataset
from torchvision.datasets.utils import download_and_extract_archive
from tqdm import tqdm
from PIL import Image

# ---------- CONFIG ----------
TARGET_DIR = "./data"
REAL_DIR = os.path.join(TARGET_DIR, "real")
FAKE_DIR = os.path.join(TARGET_DIR, "fake")
IMAGENET_URL = "https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar"
IMAGENET_FILENAME = "ILSVRC2012_img_val.tar"
EXTRACTED_IMAGENET_DIR = os.path.join(TARGET_DIR, "imagenet_val_raw")
DRAGON_DATASET = "lesc-unifi/dragon"
DRAGON_SUBSET = "Regular"
DRAGON_SPLIT = "train"
MAX_IMAGES = 25000  # Adjust for test/dev runs
models = ['Flash_PixArt', 'Flash_SD', 'Flash_SD3', 'Flash_SDXL', 'Flux_1', 'Hyper_SD', 'IF', 'JuggernautXL', 'Kandinsky', 
          'Kolors', 'LCM_SDXL', 'LCM_SSD_1B', 'Lumina', 'Mobius', 'PixArt_Alpha', 'PixArt_Sigma', 'Realistic_Stock_Photo', 
          'SD_1.5', 'SD_2.1', 'SD_3', 'SD_Cascade', 'SDXL', 'SDXL_Lightning', 'SDXL_Turbo', 'SSD_1B']

# ---------- CREATE DIRS ----------
os.makedirs(REAL_DIR, exist_ok=True)
os.makedirs(FAKE_DIR, exist_ok=True)
os.makedirs(EXTRACTED_IMAGENET_DIR, exist_ok=True)

# ---------- DOWNLOAD IMAGENET VAL ----------
def download_imagenet_val():
    if not os.path.exists(IMAGENET_FILENAME):
        print("Downloading ImageNet validation set...")
        download_and_extract_archive(IMAGENET_URL, download_root=".")
    else:
        print("ImageNet tar already exists.")

    print("Extracting ImageNet validation images...")
    os.system(f"tar -xf {IMAGENET_FILENAME} -C {EXTRACTED_IMAGENET_DIR}")

    all_images = sorted(os.listdir(EXTRACTED_IMAGENET_DIR))[:MAX_IMAGES]
    print(f"Copying {len(all_images)} images to {REAL_DIR}...")
    for img in tqdm(all_images, desc="Copying real images"):
        src = os.path.join(EXTRACTED_IMAGENET_DIR, img)
        dst = os.path.join(REAL_DIR, img)
        shutil.copyfile(src, dst)

    # # Optional cleanup
    # print("Cleaning up tar and extracted dir...")
    # os.remove(IMAGENET_FILENAME)
    # shutil.rmtree(EXTRACTED_IMAGENET_DIR)

# ---------- DOWNLOAD DRAGON IMAGES ----------
def download_dragon_images():
    print(f"Loading DRAGON dataset from HuggingFace (subset={DRAGON_SUBSET}, split={DRAGON_SPLIT})...")
    ds = load_dataset(DRAGON_DATASET, name=DRAGON_SUBSET, split=DRAGON_SPLIT)

    metadata_path = os.path.join(FAKE_DIR, "fake_metadata.csv")
    with open(metadata_path, mode="w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["filename", "model", "imagenet_class"])  # CSV header

        print(f"Saving {min(MAX_IMAGES, len(ds))} images to {FAKE_DIR}...")
        for i, sample in enumerate(tqdm(ds, total=MAX_IMAGES, desc="Saving fake images")):
            if i >= MAX_IMAGES:
                break

            img = sample["png"]
            model = sample.get("model.txt", "unknown")
            cls = sample.get("prompt.cls", "unknown")
            fname = f"dragon_{i:05}.png"

            # Save image
            img.save(os.path.join(FAKE_DIR, fname))

            # Log metadata
            writer.writerow([fname, model, cls])

    print(f"📝 Metadata written to {metadata_path}")

# ---------- MAIN ----------
if __name__ == "__main__":
    download_imagenet_val()
    download_dragon_images()
    print("\n✅ Dataset creation complete.")
