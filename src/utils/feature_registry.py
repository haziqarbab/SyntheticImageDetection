import numpy as np
import cv2
from scipy.stats import linregress
from skimage.restoration import denoise_nl_means, estimate_sigma
from skimage.filters import gabor

# ----------------- Utility Functions -----------------

def fft_mag(gray, shared):
    if "fft_mag" not in shared:
        fft = np.fft.fftshift(np.fft.fft2(gray))
        shared["fft"] = fft
        shared["fft_mag"] = np.abs(fft)
    return shared["fft_mag"]

def residual_nlm(gray, shared):
    if "residual" not in shared:
        sigma = np.mean(estimate_sigma(gray, channel_axis=None))
        denoised = denoise_nl_means(gray, h=1.15 * sigma, fast_mode=True)
        shared["residual"] = gray - denoised
    return shared["residual"]

# ----------------- Feature Functions -----------------

def fft_alpha(gray, shared):
    mag = fft_mag(gray, shared)
    y, x = np.indices(mag.shape)
    r = np.sqrt((x - x.mean())**2 + (y - y.mean())**2).astype(np.int32)
    radial_mean = np.bincount(r.ravel(), mag.ravel()) / (np.bincount(r.ravel()) + 1e-8)
    freqs = np.arange(1, len(radial_mean) + 1)
    slope, *_ = linregress(np.log(freqs), np.log(radial_mean + 1e-8))
    return slope

def fft_horz(gray, shared):
    return float(np.mean(fft_mag(gray, shared)[gray.shape[0] // 2, :]))

def fft_vert(gray, shared):
    return float(np.mean(fft_mag(gray, shared)[:, gray.shape[1] // 2]))

def corner_energy(gray, shared, ratio=0.25):
    mag = fft_mag(gray, shared)
    h, w = mag.shape
    ch, cw = int(h * ratio), int(w * ratio)
    corners = [
        mag[:ch, :cw], mag[:ch, -cw:],
        mag[-ch:, :cw], mag[-ch:, -cw:]
    ]
    combined = np.concatenate([c.flatten() for c in corners])
    return float(np.mean(np.log(np.abs(np.log(np.abs(combined) + 1)) + 1)))

def circular_power_ratio(gray, shared, ratio=0.4):
    mag = fft_mag(gray, shared)
    h, w = mag.shape
    cy, cx = h // 2, w // 2
    max_r = np.sqrt((h/2)**2 + (w/2)**2)
    y, x = np.ogrid[:h, :w]
    r = np.sqrt((y - cy)**2 + (x - cx)**2)
    mask = r <= (ratio * max_r)
    return float(np.sum(mag[mask]) / (np.sum(mag) + 1e-8))

def res_fft_var(gray, shared):
    res = residual_nlm(gray, shared)
    return float(np.var(np.fft.fftshift(np.abs(np.fft.fft2(res)))))

def dct_entropy(gray, shared):
    h, w = gray.shape
    gray = gray[:h - h % 8, :w - w % 8]
    blocks = gray.reshape(h // 8, 8, w // 8, 8).swapaxes(1, 2).reshape(-1, 8, 8)
    dct_blocks = np.array([cv2.dct(b.astype(np.float32)) for b in blocks])
    coeffs = dct_blocks.flatten()
    hist, _ = np.histogram(coeffs, bins=256, range=(-50, 50), density=True)
    return float(-np.sum(hist * np.log(hist + 1e-8)))

def phase_coherence(gray, shared):
    thetas = np.linspace(0, np.pi, 4, endpoint=False)
    gray_8u = (gray * 255).astype(np.uint8)
    results = []
    for theta in thetas:
        kernel = cv2.getGaborKernel((15, 15), 4.0, theta, 5.0, 0.5, 0, ktype=cv2.CV_32F)
        filtered = cv2.filter2D(gray_8u, cv2.CV_32F, kernel)
        results.append(np.mean(np.abs(filtered)) / (np.std(filtered) + 1e-8))
    return float(np.mean(results))

def fractal_dimension(gray, shared):
    def boxcount(Z, k):
        S = np.add.reduceat(np.add.reduceat(Z, np.arange(0, Z.shape[0], k), axis=0),
                            np.arange(0, Z.shape[1], k), axis=1)
        return np.count_nonzero(S)
    Z = (gray > gray.mean()).astype(np.uint8)
    n = 2 ** np.floor(np.log2(min(Z.shape))).astype(int)
    sizes = 2 ** np.arange(np.log2(n), 1, -1).astype(int)
    counts = [boxcount(Z, size) for size in sizes]
    try:
        coeffs = np.polyfit(np.log(sizes), np.log(counts), 1)
        return float(-coeffs[0])
    except Exception:
        return 1.5  # Reasonable fallback in the expected range of values


def spectral_corner_energy(gray, shared):
    return corner_energy(gray, shared, ratio=0.25)

def principal_directional_energy(gray, shared):
    mag = fft_mag(gray, shared)
    h, w = mag.shape
    return float(np.mean(mag[h//2, :])), float(np.mean(mag[:, w//2]))

def noise_residual_features(gray, shared):
    res = residual_nlm(gray, shared)
    return float(np.mean(res)), float(np.std(res)), float(np.var(np.fft.fftshift(np.abs(np.fft.fft2(res)))))

# ----------------- Registry -----------------

FEATURE_REGISTRY = {
    "fft_alpha": fft_alpha,
    "fft_horz": fft_horz,
    "fft_vert": fft_vert,
    "corner_energy": corner_energy,
    "circular_power_ratio": circular_power_ratio,
    "res_fft_var": res_fft_var,
    "dct_entropy": dct_entropy,
    "phase_coherence": phase_coherence,
    "fractal_dimension": fractal_dimension,
    "spectral_corner_energy": spectral_corner_energy,
    "principal_directional_energy": principal_directional_energy,
    "noise_residual_features": noise_residual_features,
}
