"""
"Focus the budget" figure (IDGI style).

IDGI weights each interpolation step by how much the model output f actually
changes there. Steps where f barely moves (the saturated tail) get ~0 weight;
the budget concentrates on the transition region.

Along the path x(alpha) = baseline + alpha*(image-baseline), alpha in [0,1]:
  - f(alpha)          : target class score (the saturating curve)
  - df_k = f_{k+1}-f_k : per-step change in f  (IDGI's importance signal)
  - weight_k          : normalized importance budget (here |df_k| / sum|df|)
                        IDGI uses (df)^2 along the gradient direction; we expose
                        WEIGHT_POWER to switch between |df| (1) and (df)^2 (2).

Shows: top  = f(alpha) with the transition region shaded;
       bottom = per-step weight bars, colored by in/out of the transition band,
                with the cumulative budget overlaid.

Reuses the same model / preprocessing. Edit IMAGE_PATH below.
Run: python idgi_budget.py
"""

import numpy as np
import torch
import torchvision.transforms as T
from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image
import matplotlib.pyplot as plt


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
IMAGE_PATH = "church.JPEG"

STEPS = 128               # interpolation points
BATCH = 32
USE_LOGIT = True          # logit saturates more visibly than prob
WEIGHT_POWER = 2          # 1 -> |df| weighting; 2 -> (df)^2 (IDGI-like)
TRANSITION_FRAC = 0.80    # transition band = smallest alpha-range holding this
                          # fraction of total weight (for shading)
SEED = 0

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

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
    tf = T.Compose([T.Resize(256), T.CenterCrop(224)])
    img = tf(pil_img.convert("RGB"))
    return np.asarray(img).astype(np.float32) / 255.0


def to_model_tensor(batch_hwc):
    x = torch.from_numpy(batch_hwc).permute(0, 3, 1, 2).float()
    mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=torch.float32).view(1, 3, 1, 1)
    return (x - mean) / std


@torch.no_grad()
def path_values(model, image, label, steps=STEPS, baseline=None):
    """f(alpha) along baseline->image. Returns alphas (S,), f (S,)."""
    if baseline is None:
        baseline = np.zeros_like(image)
    alphas = np.linspace(0.0, 1.0, steps, dtype=np.float32)
    path = baseline[None] + alphas[:, None, None, None] * (image - baseline)[None]
    f = np.empty(steps, dtype=np.float32)
    for i in range(0, steps, BATCH):
        x = to_model_tensor(path[i:i + BATCH]).to(DEVICE)
        f[i:i + len(x)] = model(x)[:, label].cpu().numpy()
    return alphas, f


def transition_band(mids, weight, frac):
    """Smallest contiguous alpha-range (in step index) holding >= frac of weight."""
    total = weight.sum()
    if total <= 0:
        return mids[0], mids[-1]
    target = frac * total
    n = len(weight)
    best = (0, n - 1)
    best_w = mids[-1] - mids[0]
    cum = np.concatenate([[0], np.cumsum(weight)])
    for i in range(n):
        # smallest j with cum[j+1]-cum[i] >= target
        need = cum[i] + target
        j = np.searchsorted(cum, need) - 1
        if j >= n:
            break
        if mids[j] - mids[i] < best_w:
            best_w = mids[j] - mids[i]
            best = (i, j)
    return mids[best[0]], mids[best[1]]


# --------------------------------------------------------------------------- #
# Run
# --------------------------------------------------------------------------- #
def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    model, weights = load_model()
    categories = weights.meta["categories"]

    image = preprocess(Image.open(IMAGE_PATH))
    label = int(model(to_model_tensor(image[None]).to(DEVICE)).argmax().item())
    print(f"Predicted: {categories[label]}")

    alphas, f = path_values(model, image, label)

    # per-step change of f and the IDGI-style budget weight
    df = np.diff(f)                                   # (S-1,)
    mids = 0.5 * (alphas[:-1] + alphas[1:])           # step midpoints
    raw_w = np.abs(df) ** WEIGHT_POWER
    weight = raw_w / (raw_w.sum() + 1e-12)            # normalized budget, sums to 1
    cum = np.cumsum(weight)

    lo, hi = transition_band(mids, weight, TRANSITION_FRAC)
    uniform = 1.0 / len(weight)                       # what naive IG would spend per step

    # --- figure --- #
    fig, (axt, axb) = plt.subplots(
        2, 1, figsize=(9, 6.5), sharex=True,
        gridspec_kw=dict(height_ratios=[1.0, 1.2], hspace=0.08))

    # top: f(alpha) with transition region shaded
    axt.plot(alphas, f, color="#1f4e8c", lw=2.5)
    axt.axvspan(lo, hi, color="crimson", alpha=0.10)
    axt.set_ylabel("class score" + ("  (logit)" if USE_LOGIT else "  (prob)"))
    axt.set_title(f"Focus the budget (IDGI): weight steps by how much f changes  ·  {categories[label]}")
    axt.scatter([alphas[0], alphas[-1]], [f[0], f[-1]], s=45, color="black", zorder=5)
    axt.annotate("baseline", (alphas[0], f[0]), xytext=(0.03, f[0] + 0.06 * (f.max() - f.min())),
                 fontsize=9, ha="left")
    axt.annotate("input", (alphas[-1], f[-1]), xytext=(0.97, f[-1] - 0.10 * (f.max() - f.min())),
                 fontsize=9, ha="right")
    axt.grid(alpha=0.25)

    # bottom: per-step weight bars
    in_band = (mids >= lo) & (mids <= hi)
    colors = np.where(in_band, "crimson", "0.75")
    axb.bar(mids, weight, width=(alphas[1] - alphas[0]) * 0.9,
            color=colors, edgecolor="none", zorder=2,
            label="IDGI step weight  ∝ |Δf|" + ("²" if WEIGHT_POWER == 2 else ""))
    axb.axhline(uniform, color="0.3", ls="--", lw=1.2, zorder=3,
                label=f"uniform IG budget (1/{len(weight)})")

    # cumulative budget on a twin axis
    axc = axb.twinx()
    axc.plot(mids, cum, color="#0a7d24", lw=2, zorder=4)
    axc.set_ylabel("cumulative budget", color="#0a7d24")
    axc.tick_params(axis="y", labelcolor="#0a7d24")
    axc.set_ylim(0, 1.02)

    axb.set_xlabel(r"interpolation $\alpha$   (0 = baseline → 1 = input)")
    axb.set_ylabel("step weight (budget share)")
    axb.set_xlim(-0.01, 1.01)
    axb.grid(alpha=0.2, axis="y")

    # band label
    frac_pct = int(TRANSITION_FRAC * 100)
    axb.annotate(f"transition region\n(~{frac_pct}% of budget)",
                 xy=((lo + hi) / 2, weight.max() * 0.9),
                 xytext=((lo + hi) / 2 + 0.12, weight.max() * 0.7),
                 fontsize=9, color="crimson", ha="left",
                 arrowprops=dict(arrowstyle="->", color="crimson"))

    # merge legends from both axes
    h1, l1 = axb.get_legend_handles_labels()
    axb.legend(h1, l1, loc="upper right", frameon=False, fontsize=9)

    plt.savefig("idgi_budget.png", bbox_inches="tight", dpi=150)
    print(f"Saved: idgi_budget.png   transition band alpha=[{lo:.2f},{hi:.2f}]")


if __name__ == "__main__":
    main()