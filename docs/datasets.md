# 📊 Datasets Considered for AI vs Real Image Detection

## ✅ Selected: DRAGON (CVPR 2025)
- **Name**: DRAGON (Detecting Real and AI-Generated cONtent)
- **Published**: CVPR 2025
- **Content**: 20+ image generators (including SD 2.0, DALL-E 2, Midjourney)
- **Use Case**: General-purpose forensic detection, image-level benchmarking
- **Why Chosen**: Balanced, diverse, includes high-res Stable Diffusion 2.0 images

[🔗 Official GitHub Link](https://github.com/NVlabs/DRAGON)

---

## 🕐 Datasets Reviewed (Not Used Due to Time Constraints)

### 🟡 DiffusionDB
- Pros: Real user prompts, raw SD samples, 14M images
- Cons: Large (TB-scale), inconsistent formatting

### 🟡 LAION/AI-generated-images (HuggingFace)
- Pros: Easy access via `datasets`, metadata includes generator
- Cons: URL reliability and label noise in large-scale version

### 🟡 GenImage
- Pros: Multiple generative models including SD, BigGAN, StyleGAN3
- Cons: Smaller size; better for controlled experiments

### 🟡 DiffusionFace
- Pros: Focused on faces; well-labeled
- Cons: Too narrow for general image detection

---

## 💾 Real Image Sources Paired with DRAGON
- **COCO val2017**: Real-world annotated scenes (~5k images)
- **RAISE-1k**: DSLR RAW-format photography (non-compressed)
