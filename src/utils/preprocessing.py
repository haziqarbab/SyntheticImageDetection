import cv2
import numpy as np

# ------------------------ Global Config ------------------------

TARGET_SIZE = (512, 512)

def set_target_image_size(size):
    """
    Sets the global target size for all image resizing.
    Accepts either an int or (height, width) tuple.
    """
    global TARGET_SIZE
    if isinstance(size, int):
        TARGET_SIZE = (size, size)
    elif isinstance(size, tuple) and len(size) == 2:
        TARGET_SIZE = size
    else:
        raise ValueError("Size must be an int or (height, width) tuple.")

# ------------------------ Adaptive Resize ------------------------

def resize_image_adaptive(image, target_size=None):
    """
    Resize the image using adaptive interpolation:
    - Use INTER_AREA for downsampling
    - Use INTER_LANCZOS4 for upsampling
    """
    if target_size is None:
        target_size = TARGET_SIZE

    h, w = image.shape[:2]
    th, tw = target_size

    interp = cv2.INTER_AREA if (th < h or tw < w) else cv2.INTER_LANCZOS4
    return cv2.resize(image, (tw, th), interpolation=interp)

# ------------------------ Preprocessing ------------------------

def preprocess_image(img_path, target_size=None):
    """
    Load an image, convert to grayscale, normalize to [0, 1], and resize.
    """
    if target_size is None:
        target_size = TARGET_SIZE

    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Image not found or unreadable: {img_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_norm = gray.astype(np.float32) / 255.0
    gray_resized = resize_image_adaptive(gray_norm, target_size)

    return gray_resized
