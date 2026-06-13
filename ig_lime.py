"""
LIME (7x7 grid) vs Integrated Gradients (pixel-level) overlay.

Produces one side-by-side figure for the "grid masking vs pixel masking"
placeholder: input | LIME-grid heatmap | IG heatmap.

  - ResNet50 pretrained ImageNet-1k
  - LIME: fixed 7x7 grid, off-superpixels replaced by a baseline (black by default),
    Ridge surrogate on top-1 probability
  - IG: pixel-level, straight-line path from black baseline, gradient w.r.t. top-1 logit

Edit IMAGE_PATH below. Run: python lime_ig_overlay.py
"""

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.models import resnet50, ResNet50_Weights
from sklearn.linear_model import Ridge
from sklearn.metrics import pairwise_distances
from PIL import Image
import matplotlib.pyplot as plt


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
IMAGE_PATH = "church.JPEG"

GRID = 7                  # 7x7 grid -> 49 superpixels
N_SUPERPIXELS = GRID * GRID
N_SAMPLES = 1000          # LIME perturbation samples
KERNEL_WIDTH = 0.25
RIDGE_ALPHA = 1.0
MASK_MODE = "black"       # baseline for off-superpixels: "black" | "white" | "mean"

IG_STEPS = 64             # Riemann steps for IG
IG_BATCH = 16             # path-point batch size (grad pass)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 0

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])


# --------------------------------------------------------------------------- #
# Model / preprocessing
# --------------------------------------------------------------------------- #
def load_model():
    weights = ResNet50_Weights.IMAGENET1K_V2
    model = resnet50(weights=weights).to(DEVICE).eval()
    return model, weights


def preprocess(pil_img):
    """RGB float [0,1], 224x224, HxWxC, not normalized."""
    tf = T.Compose([T.Resize(256), T.CenterCrop(224)])
    img = tf(pil_img.convert("RGB"))
    return np.asarray(img).astype(np.float32) / 255.0


def to_model_tensor(batch_hwc):
    """(N,H,W,C) float [0,1] -> normalized (N,C,H,W) tensor."""
    x = torch.from_numpy(batch_hwc).permute(0, 3, 1, 2).float()
    mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=torch.float32).view(1, 3, 1, 1)
    return (x - mean) / std


@torch.no_grad()
def predict_probs(model, batch_hwc, bs=64):
    out = []
    for i in range(0, len(batch_hwc), bs):
        chunk = to_model_tensor(batch_hwc[i:i + bs]).to(DEVICE)
        logits = model(chunk)
        out.append(F.softmax(logits, dim=1).cpu().numpy())
    return np.concatenate(out, axis=0)


# --------------------------------------------------------------------------- #
# Fixed 7x7 grid segmentation + baseline
# --------------------------------------------------------------------------- #
def make_grid_segments(h, w, grid=GRID):
    seg = np.zeros((h, w), dtype=np.int64)
    ys = np.linspace(0, h, grid + 1).astype(int)
    xs = np.linspace(0, w, grid + 1).astype(int)
    sid = 0
    for i in range(grid):
        for j in range(grid):
            seg[ys[i]:ys[i + 1], xs[j]:xs[j + 1]] = sid
            sid += 1
    return seg


def make_baseline(image, segments, mode):
    if mode == "black":
        return np.zeros_like(image)
    if mode == "white":
        return np.ones_like(image)
    if mode == "mean":
        base = np.zeros_like(image)
        for sid in np.unique(segments):
            m = segments == sid
            base[m] = image[m].reshape(-1, 3).mean(axis=0)
        return base
    raise ValueError(mode)


# --------------------------------------------------------------------------- #
# LIME (grid)
# --------------------------------------------------------------------------- #
def lime_explain(model, image, segments, baseline, label,
                 n_samples=N_SAMPLES, kernel_width=KERNEL_WIDTH,
                 alpha=RIDGE_ALPHA, rng=None):
    rng = rng or np.random.default_rng(SEED)
    n_seg = len(np.unique(segments))

    z = rng.integers(0, 2, size=(n_samples, n_seg))   # 1=keep, 0=baseline
    z[0] = 1                                            # original image

    perturbed = np.empty((n_samples, *image.shape), dtype=np.float32)
    for k in range(n_samples):
        img_k = image.copy()
        for sid in np.where(z[k] == 0)[0]:
            m = segments == sid
            img_k[m] = baseline[m]
        perturbed[k] = img_k

    probs = predict_probs(model, perturbed)[:, label]

    dist = pairwise_distances(z, z[0].reshape(1, -1), metric="cosine").ravel()
    weights = np.exp(-(dist ** 2) / (kernel_width ** 2))

    surrogate = Ridge(alpha=alpha, fit_intercept=True)
    surrogate.fit(z, probs, sample_weight=weights)
    return surrogate.coef_, surrogate.score(z, probs, sample_weight=weights)


def coef_to_heatmap(coef, segments):
    hm = np.zeros(segments.shape, dtype=np.float32)
    for sid, c in enumerate(coef):
        hm[segments == sid] = c
    return hm


# --------------------------------------------------------------------------- #
# Integrated Gradients (pixel-level)
# --------------------------------------------------------------------------- #
def integrated_gradients(model, image, label, steps=IG_STEPS, baseline=None):
    """image: (H,W,3) float [0,1]. Returns (H,W) saliency = |sum over channels|."""
    if baseline is None:
        baseline = np.zeros_like(image)
    alphas = np.linspace(0.0, 1.0, steps, dtype=np.float32)
    path = baseline[None] + alphas[:, None, None, None] * (image - baseline)[None]  # (S,H,W,3)

    total = torch.zeros(3, image.shape[0], image.shape[1], device=DEVICE)
    for i in range(0, len(path), IG_BATCH):
        x = to_model_tensor(path[i:i + IG_BATCH]).to(DEVICE).requires_grad_(True)
        logits = model(x)
        score = logits[:, label].sum()
        g, = torch.autograd.grad(score, x)
        total += g.sum(dim=0)                       # accumulate gradients along path
    avg_grad = (total / steps).cpu().numpy()         # (3,H,W)

    # IG = (input - baseline) * avg_grad, undo normalization scale on the input diff
    diff = (image - baseline).transpose(2, 0, 1)     # (3,H,W) in [0,1] units
    diff_norm = diff / IMAGENET_STD[:, None, None]   # match normalized-input gradients
    ig = diff_norm * avg_grad                        # (3,H,W)
    sal = np.abs(ig).sum(axis=0)                     # (H,W)
    return sal


# --------------------------------------------------------------------------- #
# Run
# --------------------------------------------------------------------------- #
def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    model, weights = load_model()
    categories = weights.meta["categories"]

    image = preprocess(Image.open(IMAGE_PATH))
    h, w = image.shape[:2]
    segments = make_grid_segments(h, w, GRID)

    base_prob = predict_probs(model, image[None])[0]
    label = int(base_prob.argmax())
    print(f"Predicted: {categories[label]} (p={base_prob[label]:.3f})")

    # LIME grid
    baseline = make_baseline(image, segments, MASK_MODE)
    coef, r2 = lime_explain(model, image, segments, baseline, label,
                            rng=np.random.default_rng(SEED))
    lime_hm = coef_to_heatmap(coef, segments)
    print(f"LIME (mask={MASK_MODE}) surrogate R^2={r2:.3f}  "
          f"coef[min,max]=[{coef.min():.4f},{coef.max():.4f}]")

    # IG pixel-level
    ig_sal = integrated_gradients(model, image, label)
    # percentile clip + normalize for display
    vmax_ig = np.percentile(ig_sal, 99)
    ig_disp = np.clip(ig_sal / (vmax_ig + 1e-12), 0, 1)

    # --- figure: input | LIME grid | IG --- #
    vmax_l = np.abs(lime_hm).max()
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(image)
    axes[0].set_title(f"input\n{categories[label]}")
    axes[0].axis("off")

    axes[1].imshow(image)
    im1 = axes[1].imshow(lime_hm, cmap="bwr", vmin=-vmax_l, vmax=vmax_l, alpha=0.6)
    axes[1].set_title(f"LIME 7x7 grid (R²={r2:.2f})")
    axes[1].axis("off")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.02)

    axes[2].imshow(image)
    im2 = axes[2].imshow(ig_disp, cmap="hot", alpha=0.6)
    axes[2].set_title("Integrated Gradients (pixel)")
    axes[2].axis("off")
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.02)

    fig.suptitle("Grid masking (LIME) vs pixel masking (IG)", y=1.03)
    plt.savefig("lime_ig_overlay.png", bbox_inches="tight", dpi=150)
    print("Saved: lime_ig_overlay.png")


if __name__ == "__main__":
    main()