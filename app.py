# app.py — Pneumonia X-ray XAI (Gradio, single page). Research demo; not for clinical use.

import os, io, glob, re
import numpy as np
import pandas as pd
from PIL import Image

import torch, torch.nn.functional as F
from torch import nn
from torchvision import models, transforms

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import gradio as gr


# ============================== Utils ==============================

def get_device(prefer_mps=False):
    if prefer_mps and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def preprocess(img_obj, img_size=224):
    """Accepts PIL.Image or path-like. Returns (tensor[C,H,W], PIL)."""
    tfm = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])
    if isinstance(img_obj, Image.Image):
        pil = img_obj.convert("RGB")
    else:
        pil = Image.open(img_obj).convert("RGB")
    return tfm(pil), pil

def overlay_heatmap(pil, heat, alpha=0.5, size=224):
    import cv2
    pil = pil.resize((size, size))
    base = np.array(pil)
    heat = (heat - heat.min()) / (heat.max() - heat.min() + 1e-6)
    heat_u8 = np.uint8(255 * heat)
    heat_color = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)
    heat_color = cv2.cvtColor(heat_color, cv2.COLOR_BGR2RGB)
    return Image.fromarray(np.uint8(alpha*heat_color + (1-alpha)*base))

def label_from_path(p):
    name = str(p).lower()
    if "normal" in name: return 0
    if "pneumonia" in name: return 1
    return None

def metrics_from_probs(y, p, thr):
    y = np.asarray(y).astype(int); p = np.asarray(p).astype(float)
    y_pred = (p >= thr).astype(int)
    tn = int(((y==0) & (y_pred==0)).sum())
    fp = int(((y==0) & (y_pred==1)).sum())
    fn = int(((y==1) & (y_pred==0)).sum())
    tp = int(((y==1) & (y_pred==1)).sum())
    acc  = (tp+tn) / max(1, (tp+tn+fp+fn))
    prec = tp / max(1, (tp+fp))
    rec  = tp / max(1, (tp+fn))   # sensitivity
    spec = tn / max(1, (tn+fp))
    f1   = (2*prec*rec) / max(1e-9, (prec+rec))
    return (tn, fp, fn, tp), acc, prec, rec, spec, f1

def thr_at_spec(y, p, target_spec=0.90):
    y = np.asarray(y).astype(int); p = np.asarray(p).astype(float)
    if len(y) == 0: return 0.5
    order = np.argsort(-p)
    y_sorted = y[order]; p_sorted = p[order]
    N_neg = int((y==0).sum())
    if N_neg == 0: return 0.5
    fp_cum = np.cumsum((y_sorted==0).astype(int))
    spec = (N_neg - fp_cum) / N_neg
    idx = np.where(spec >= target_spec)[0]
    if len(idx)==0: return float(np.nextafter(p_sorted[0], 0))
    return float(p_sorted[idx[-1]])

def roc_curve_np(y_true, y_score):
    y = np.asarray(y_true).astype(int)
    s = np.asarray(y_score).astype(float)
    order = np.argsort(-s)
    y, s = y[order], s[order]
    P, N = y.sum(), (y==0).sum()
    tp=fp=0; last=np.inf; tpr=[]; fpr=[]
    for si, yi in zip(s, y):
        if si != last:
            tpr.append(tp/max(1,P)); fpr.append(fp/max(1,N)); last = si
        if yi==1: tp+=1
        else: fp+=1
    tpr.append(tp/max(1,P)); fpr.append(fp/max(1,N))
    return np.array(fpr), np.array(tpr)

def auc_trapz(x, y):
    idx = np.argsort(x)
    return float(np.trapz(y[idx], x[idx]))

def reliability_curve(y_true, y_prob, n_bins=10):
    y = np.asarray(y_true).astype(int)
    p = np.asarray(y_prob).astype(float)
    bins = np.linspace(0,1,n_bins+1)
    ece = 0.0; brier = float(np.mean((p - y)**2))
    bin_stats = []; N = len(y)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i+1]
        idx = (p >= lo) & (p < hi) if i < n_bins-1 else (p >= lo) & (p <= hi)
        n_k = int(idx.sum())
        if n_k == 0:
            bin_stats.append(((lo+hi)/2, 0.0, 0)); continue
        conf = float(p[idx].mean()); acc = float(y[idx].mean())
        ece += (n_k/N)*abs(acc - conf)
        bin_stats.append((conf, acc, n_k))
    return np.array(bin_stats, dtype=float), float(ece), float(brier)

def central_lung_mask(h, w, pad=0.12):
    y0, y1 = int(h*pad), int(h*(1-pad))
    x0, x1 = int(w*pad), int(w*(1-pad))
    m = np.zeros((h,w), dtype=np.float32); m[y0:y1, x0:x1] = 1.0
    return m


# ============================== XAI ==============================

def gradcam_map(model, x):
    from captum.attr import LayerGradCam
    layer = model.layer4[-1].conv3 if hasattr(model.layer4[-1],'conv3') else model.layer4[-1].conv2
    gc = LayerGradCam(model, layer)
    m = gc.attribute(x, target=0)
    m = F.interpolate(m, size=(x.shape[-2], x.shape[-1]), mode="bilinear", align_corners=False)
    return m.squeeze().detach().cpu().numpy()

def ig_map(model, x):
    from captum.attr import IntegratedGradients
    ig = IntegratedGradients(model)
    baseline = torch.zeros_like(x)
    m = ig.attribute(x, baselines=baseline, target=0).squeeze().detach().cpu().numpy()
    return np.abs(m).mean(axis=0)


# ============================== Model ==============================

_MODEL = None
_MODEL_DEVICE = None
_MODEL_SOURCE = None

def load_model(ckpt_path:str, device, allow_imagenet_fallback:bool):
    global _MODEL, _MODEL_DEVICE, _MODEL_SOURCE
    if (_MODEL is not None) and (_MODEL_DEVICE == str(device)) and (os.path.abspath(ckpt_path) == getattr(_MODEL, "_ckpt", "")):
        return _MODEL, _MODEL_SOURCE

    weights = None
    source = "random"
    if not os.path.exists(ckpt_path) and allow_imagenet_fallback:
        weights = models.ResNet50_Weights.DEFAULT
        source = "imagenet"

    model = models.resnet50(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, 1)

    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        state = ckpt.get("model", ckpt.get("state_dict", ckpt))
        state = {k.replace("module.",""): v for k,v in state.items()}
        model.load_state_dict(state, strict=False)
        source = "checkpoint"

    model.to(device).eval()
    model._ckpt = os.path.abspath(ckpt_path)
    _MODEL, _MODEL_DEVICE, _MODEL_SOURCE = model, str(device), source
    return model, source

def forward_prob(model, x, T=None):
    with torch.no_grad():
        logit = model(x).squeeze(1)
        if T and T > 0: logit = logit / T
        return torch.sigmoid(logit).item()


# ============================== Dataset helpers ==============================

def gather_dataset_files(root, split="test", limit_per_class=0):
    """Return list of file paths under root/<split>/(NORMAL|PNEUMONIA)/*"""
    exts = ("*.png","*.jpg","*.jpeg","*.bmp")
    if split not in ("train","val","test","all"): split = "test"
    splits = [split] if split != "all" else ["train","val","test"]
    paths = []
    for sp in splits:
        for cls in ("NORMAL","PNEUMONIA"):
            cls_files = []
            for ext in exts:
                cls_files += glob.glob(os.path.join(root, sp, cls, ext))
            cls_files = sorted(cls_files)
            if limit_per_class and limit_per_class > 0:
                cls_files = cls_files[:int(limit_per_class)]
            paths.extend(cls_files)
    return paths

def score_dataset(paths, ckpt_path, allow_imagenet_fallback, img_size, swap_labels, prefer_mps):
    device = get_device(prefer_mps)
    model, source = load_model(ckpt_path, device, allow_imagenet_fallback)
    y, p = [], []
    for fp in paths:
        x, _ = preprocess(fp, img_size)
        prob = forward_prob(model, x.unsqueeze(0).to(device))
        if swap_labels: prob = 1.0 - float(prob)
        p.append(prob)
        y.append(label_from_path(fp) if label_from_path(fp) in (0,1) else -1)
    return np.array(y), np.array(p), source, str(device)


# ============================== Compute blocks ==============================

def analyze_dataset(root, split, limit, ckpt_path, allow_imagenet_fallback,
                    img_size, swap_labels, prefer_mps, threshold):
    if not root or not os.path.isdir(root):
        return ("❌ Invalid dataset folder.", None, None, None,
                [], [], [], None, [],
                gr.update(choices=[], value=None))

    paths = gather_dataset_files(root, split, limit)
    if not paths:
        return ("❌ No images found. Expecting <root>/<split>/(NORMAL|PNEUMONIA)/*.jpg", None, None, None,
                [], [], [], None, [],
                gr.update(choices=[], value=None))

    y, p, source, device = score_dataset(paths, ckpt_path, allow_imagenet_fallback, img_size, swap_labels, prefer_mps)
    known = (y >= 0)
    yk, pk = y[known], p[known]

    # Metrics @ current thr
    (tn, fp_, fn, tp), acc, prec, rec, spec, f1 = metrics_from_probs(yk, pk, threshold)

    # Thr@90%Spec
    thr90 = thr_at_spec(yk, pk, 0.90)
    (tn90, fp90, fn90, tp90), acc90, prec90, rec90, spec90, f190 = metrics_from_probs(yk, pk, thr90)

    # ROC
    fpr, tpr = roc_curve_np(yk, pk); auc = auc_trapz(fpr, tpr)
    plt.figure(); plt.plot(fpr, tpr, label=f"AUC={auc:.3f}"); plt.plot([0,1],[0,1],'--')
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.legend(loc="lower right"); plt.tight_layout()
    buf_roc = io.BytesIO(); plt.savefig(buf_roc, format="png", dpi=150); plt.close(); buf_roc.seek(0)
    roc_img = Image.open(buf_roc)

    # Calibration
    bins, ece, brier = reliability_curve(yk, pk, n_bins=10)
    plt.figure(); plt.plot([0,1],[0,1],'--'); plt.scatter(bins[:,0], bins[:,1])
    plt.xlabel("Mean predicted p"); plt.ylabel("Empirical fraction positive")
    plt.title(f"ECE={ece:.3f} | Brier={brier:.3f}"); plt.tight_layout()
    buf_cal = io.BytesIO(); plt.savefig(buf_cal, format="png", dpi=150); plt.close(); buf_cal.seek(0)
    cal_img = Image.open(buf_cal)

    # Confusion matrix (current thr)
    cm_df = pd.DataFrame([[tn, fp_],[fn, tp]],
                         index=["True 0 (Normal)","True 1 (Pneumonia)"],
                         columns=["Pred 0","Pred 1"])

    # Dropdown options (ASCII-safe delimiter)
    options = [f"{i:04d} | {os.path.basename(fp)}" for i, fp in enumerate(paths)]

    summary = (
        f"**Device:** {device}  |  **Model:** {source}  |  **N (labeled):** {len(yk)} / {len(y)}  \n"
        f"**Current thr {threshold:.3f}** → Acc **{acc:.4f}**, Prec **{prec:.4f}**, Sens **{rec:.4f}**, "
        f"Spec **{spec:.4f}**, F1 **{f1:.4f}**  \n"
        f"**AUC:** {auc:.3f}  ·  **ECE:** {ece:.3f}  ·  **Brier:** {brier:.3f}  \n"
        f"**Thr@90%Spec:** {thr90:.3f} (Sens {rec90:.3f}, Spec {spec90:.3f})"
    )

    # IMPORTANT: return dropdown update atomically (choices + value)
    dd_update = gr.update(choices=options, value=options[0] if options else None)

    return (summary, cm_df, roc_img, cal_img,
            paths, y.tolist(), p.tolist(), float(thr90), options,
            dd_update)

def recompute_at_threshold(threshold, y, p):
    if not y or not p:
        return gr.update(value="Load a dataset first."), None
    y = np.array(y); p = np.array(p)
    known = (y >= 0)
    yk, pk = y[known], p[known]
    (tn, fp_, fn, tp), acc, prec, rec, spec, f1 = metrics_from_probs(yk, pk, threshold)
    cm_df = pd.DataFrame([[tn, fp_],[fn, tp]],
                         index=["True 0 (Normal)","True 1 (Pneumonia)"],
                         columns=["Pred 0","Pred 1"])
    summary = (
        f"**Current thr {threshold:.3f}** → Acc **{acc:.4f}**, Prec **{prec:.4f}**, "
        f"Sens **{rec:.4f}**, Spec **{spec:.4f}**, F1 **{f1:.4f}**"
    )
    return summary, cm_df

def explain_selected(sel, paths, y, p, ckpt_path, allow_imagenet_fallback,
                     img_size, swap_labels, prefer_mps, threshold):
    if not sel or not paths:
        return "Pick an image after loading dataset.", None, None, None
    try:
        idx = int(sel.split("|")[0].strip())
    except Exception:
        idx = 0
    idx = max(0, min(idx, len(paths)-1))
    fp = paths[idx]
    yy = label_from_path(fp)
    device = get_device(prefer_mps)
    model, source = load_model(ckpt_path, device, allow_imagenet_fallback)

    x, pil = preprocess(fp, img_size); x1 = x.unsqueeze(0).to(device)
    prob = forward_prob(model, x1)
    if swap_labels: prob = 1.0 - float(prob)

    pred_lbl = "PNEUMONIA" if prob >= threshold else "NORMAL"
    gt_lbl = "?" if yy is None else ("NORMAL" if yy==0 else "PNEUMONIA")

    try:
        gc = gradcam_map(model, x1)
        ig = ig_map(model, x1)
        gc_img = overlay_heatmap(pil, gc, size=img_size)
        ig_img = overlay_heatmap(pil, ig, size=img_size)
        mask = central_lung_mask(img_size, img_size, pad=0.12)
        gc_pos = gc - gc.min(); s = gc_pos.sum() + 1e-6
        in_lung = float((gc_pos*mask).sum()/s); out_lung = 1.0 - in_lung
        focus = f" | Saliency in lung: {in_lung:.0%} (outside {out_lung:.0%})"
    except Exception:
        gc_img = None; ig_img = None; focus = " | XAI unavailable"

    info = (
        f"**{os.path.basename(fp)}**  \n"
        f"GT: **{gt_lbl}**  |  Pred: **{pred_lbl}**  |  p={prob:.3f} @ thr={threshold:.2f}{focus}"
    )
    return info, pil, gc_img, ig_img


# ============================== UI (one page + real Sidebar) ==============================

with gr.Blocks(title="Pneumonia X-ray — XAI (single page)") as demo:
    gr.Markdown("# 🫁 Chest X-ray Pneumonia — All-in-one XAI Dashboard")

    # Shared state
    st_paths   = gr.State([])
    st_y       = gr.State([])
    st_p       = gr.State([])
    st_thr90   = gr.State(None)
    st_options = gr.State([])

    with gr.Row():
        with gr.Sidebar():
            gr.Markdown("### Settings")
            ds_root = gr.Textbox(label="Dataset root", value=os.path.join(os.path.dirname(__file__), "samples"))
            split = gr.Dropdown(choices=["train","val","test","all"], value="test", label="Split")
            limit = gr.Slider(0, 2000, value=0, step=10, label="Max images per class (0 = all)")

            ckpt_path = gr.Textbox(label="Model checkpoint (.pt/.pth)", value="outputs/best.pt")
            allow_imagenet = gr.Checkbox(label="If checkpoint missing, use ImageNet fallback", value=True)
            swap_labels = gr.Checkbox(label="Model outputs NORMAL prob (swap labels)", value=True)
            prefer_mps = gr.Checkbox(label="Use Apple MPS (local Mac only)", value=False)

            img_size = gr.Slider(128, 512, value=224, step=32, label="Image size")
            threshold = gr.Slider(0.0, 1.0, value=0.50, step=0.01, label="Decision threshold")

            load_btn = gr.Button("🔁 Load dataset & run metrics", variant="primary")
            use_thr90_btn = gr.Button("Set threshold to Thr@90%Spec")

            gr.Markdown("---")
            gr.Markdown("### Image selection")
            img_select = gr.Dropdown(choices=[], label="Pick an image from dataset")
            prev_btn = gr.Button("⬅️ Prev")
            next_btn = gr.Button("Next ➡️")

        with gr.Column(scale=3):
            gr.Markdown("### Dataset metrics (linked to sidebar threshold)")
            summary_md = gr.Markdown()
            cm_tbl = gr.Dataframe(label="Confusion matrix (current threshold)", wrap=True)
            with gr.Row():
                roc_img = gr.Image(label="ROC")
                cal_img = gr.Image(label="Calibration (reliability)")

            gr.Markdown("### Selected image — prediction & explanations")
            info_md = gr.Markdown()
            with gr.Row():
                orig_img = gr.Image(label="Original")
                gc_img = gr.Image(label="Grad-CAM")
                ig_img = gr.Image(label="Integrated Gradients")

    # Load & wire outputs — NOTE the dropdown update is returned atomically
    load_btn.click(
        analyze_dataset,
        inputs=[ds_root, split, limit, ckpt_path, allow_imagenet, img_size, swap_labels, prefer_mps, threshold],
        outputs=[summary_md, cm_tbl, roc_img, cal_img, st_paths, st_y, st_p, st_thr90, st_options, img_select]
    ).then(  # auto-explain the first image after choices are set
        explain_selected,
        inputs=[img_select, st_paths, st_y, st_p, ckpt_path, allow_imagenet, img_size, swap_labels, prefer_mps, threshold],
        outputs=[info_md, orig_img, gc_img, ig_img]
    )

    use_thr90_btn.click(
        lambda thr90: gr.update(value=float(thr90) if thr90 else gr.skip()),
        inputs=[st_thr90],
        outputs=[threshold]
    ).then(
        recompute_at_threshold,
        inputs=[threshold, st_y, st_p],
        outputs=[summary_md, cm_tbl]
    )

    threshold.change(  # live metrics update, does NOT touch dropdown
        recompute_at_threshold,
        inputs=[threshold, st_y, st_p],
        outputs=[summary_md, cm_tbl]
    )

    img_select.change(
        explain_selected,
        inputs=[img_select, st_paths, st_y, st_p, ckpt_path, allow_imagenet, img_size, swap_labels, prefer_mps, threshold],
        outputs=[info_md, orig_img, gc_img, ig_img]
    )

    # Prev/Next navigation uses stored options
    def nav(delta, sel, options):
        if not options: return sel
        try: i = options.index(sel)
        except ValueError: i = 0
        i = (i + delta) % len(options)
        return options[i]

    prev_btn.click(lambda s, opts: nav(-1, s, opts), inputs=[img_select, st_options], outputs=[img_select])\
            .then(explain_selected,
                  inputs=[img_select, st_paths, st_y, st_p, ckpt_path, allow_imagenet, img_size, swap_labels, prefer_mps, threshold],
                  outputs=[info_md, orig_img, gc_img, ig_img])

    next_btn.click(lambda s, opts: nav(+1, s, opts), inputs=[img_select, st_options], outputs=[img_select])\
            .then(explain_selected,
                  inputs=[img_select, st_paths, st_y, st_p, ckpt_path, allow_imagenet, img_size, swap_labels, prefer_mps, threshold],
                  outputs=[info_md, orig_img, gc_img, ig_img])

if __name__ == "__main__":
    demo.launch()  # set share=True to expose a public link locally
