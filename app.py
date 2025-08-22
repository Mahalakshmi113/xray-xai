# app.py — Chest X-ray Pneumonia XAI (single page). Research demo; not for clinical use.

# ── Streamlit must be configured before any other st.* call ────────────────────────
import streamlit as st
st.set_page_config(
    page_title="Pneumonia X-ray — XAI",
    layout="wide",
    page_icon="🫁",
    initial_sidebar_state="expanded",
)

# ── Standard imports ──────────────────────────────────────────────────────────────
import os, glob, re, math, random
import numpy as np
import pandas as pd

os.environ.setdefault("STREAMLIT_SERVER_PORT", "8501")

import torch
import torch.nn.functional as F
from torch import nn
from torchvision import models, transforms
from PIL import Image

# NOTE: cv2 is imported inside overlay_heatmap() so the app still runs without it.

# ============================== Utilities ==============================

def get_device(prefer_mps: bool = False):
    if prefer_mps and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def preprocess(img_obj, img_size=224):
    tfm = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])
    pil = Image.open(img_obj).convert("RGB")
    return tfm(pil), pil

def overlay_heatmap(pil, heat, alpha=0.5, size=224):
    import cv2  # lazy import
    pil = pil.resize((size, size))
    base = np.array(pil)
    heat = (heat - heat.min()) / (heat.max() - heat.min() + 1e-6)
    heat_u8 = np.uint8(255 * heat)
    heat_color = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)
    heat_color = cv2.cvtColor(heat_color, cv2.COLOR_BGR2RGB)
    return np.uint8(alpha*heat_color + (1-alpha)*base)

def list_images_in_split(root, split):
    pats = ["*.jpg","*.jpeg","*.png","*.bmp"]
    files = []
    for cls in ("NORMAL","PNEUMONIA"):
        for ext in pats:
            files += glob.glob(os.path.join(root, split, cls, ext))
    return sorted(files)

def list_test_images(data_dir):
    return list_images_in_split(data_dir, "test")

def label_from_path(p):
    cls = os.path.basename(os.path.dirname(p)).upper()
    return 0 if cls == "NORMAL" else 1

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
    """Find a threshold achieving >= target specificity (no sklearn)."""
    y = np.asarray(y).astype(int); p = np.asarray(p).astype(float)
    if len(y) == 0: return 0.5
    order = np.argsort(-p)  # desc by prob
    y_sorted = y[order]; p_sorted = p[order]
    N_neg = int((y==0).sum())
    if N_neg == 0: return 0.5
    fp_cum = np.cumsum((y_sorted==0).astype(int))
    spec = (N_neg - fp_cum) / N_neg
    idx = np.where(spec >= target_spec)[0]
    if len(idx)==0:
        return float(np.nextafter(p_sorted[0], 0))
    return float(p_sorted[idx[-1]])

def central_lung_mask(h, w, pad=0.12):
    y0, y1 = int(h*pad), int(h*(1-pad))
    x0, x1 = int(w*pad), int(w*(1-pad))
    m = np.zeros((h,w), dtype=np.float32); m[y0:y1, x0:x1] = 1.0
    return m

# ---------------- ROC & Calibration helpers ----------------
def roc_curve_np(y_true, y_score):
    y = np.asarray(y_true).astype(int)
    s = np.asarray(y_score).astype(float)
    order = np.argsort(-s)
    y = y[order]; s = s[order]
    P = y.sum(); N = (y==0).sum()
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
    bin_stats = []
    N = len(y)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i+1]
        idx = (p >= lo) & (p < hi) if i < n_bins-1 else (p >= lo) & (p <= hi)
        n_k = int(idx.sum())
        if n_k == 0:
            bin_stats.append(((lo+hi)/2, 0.0, 0))
            continue
        conf = float(p[idx].mean())
        acc  = float(y[idx].mean())
        ece += (n_k/N)*abs(acc - conf)
        bin_stats.append((conf, acc, n_k))
    return np.array(bin_stats, dtype=float), float(ece), float(brier)

# ============================== Explainability ==============================

def gradcam_map(model, x):
    from captum.attr import LayerGradCam
    layer = model.layer4[-1].conv3 if hasattr(model.layer4[-1],'conv3') else model.layer4[-1].conv2
    gc = LayerGradCam(model, layer)
    m = gc.attribute(x, target=0)
    m = F.interpolate(m, size=(x.shape[-2], x.shape[-1]), mode='bilinear', align_corners=False)
    return m.squeeze().detach().cpu().numpy()

def ig_map(model, x):
    from captum.attr import IntegratedGradients
    ig = IntegratedGradients(model)
    baseline = torch.zeros_like(x)
    m = ig.attribute(x, baselines=baseline, target=0).squeeze().detach().cpu().numpy()
    return np.abs(m).mean(axis=0)

# ============================== Model loader (no Streamlit inside) ==============================

def get_model(ckpt_path, device, allow_imagenet_fallback=False):
    """Return (model, loaded_flag, source_str)."""
    weights = None
    source = "random"
    if not os.path.exists(ckpt_path) and allow_imagenet_fallback:
        weights = models.ResNet50_Weights.DEFAULT
        source = "imagenet"
    model = models.resnet50(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, 1)

    loaded = False
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        state = ckpt.get("model", ckpt.get("state_dict", ckpt))
        state = {k.replace("module.",""): v for k,v in state.items()}
        model.load_state_dict(state, strict=False)
        loaded, source = True, "checkpoint"

    return model.to(device).eval(), loaded, source

def forward_prob(model, x, T=None):
    with torch.no_grad():
        logit = model(x).squeeze(1)
        if T and T > 0:
            logit = logit / T
        return torch.sigmoid(logit).item()

def predict_tta(model, x, n=12, T=None):
    probs = []
    with torch.no_grad():
        for _ in range(max(1,int(n))):
            x_aug = x.clone()
            if torch.rand(1).item() < 0.5:
                x_aug = torch.flip(x_aug, dims=[-1])
            logit = model(x_aug).squeeze(1)
            if T and T > 0:
                logit = logit / T
            probs.append(torch.sigmoid(logit).item())
    return float(np.mean(probs)), float(np.std(probs))

# ============================== Temperature scaling ==============================

def read_temperature():
    path = "reports/temperature.txt"
    if os.path.exists(path):
        try: return float(open(path).read().strip())
        except Exception: return None
    return None

def fit_temperature_on_val(model, data_dir, img_size, device, batch=32):
    from torch.utils.data import DataLoader
    from torchvision import datasets
    tfm = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])
    val_dir = os.path.join(data_dir, "val")
    if not os.path.isdir(val_dir):
        st.error("Validation folder not found (need <data>/val/... to fit temperature).")
        return None
    ds = datasets.ImageFolder(val_dir, transform=tfm)
    if len(ds) == 0:
        st.error("Validation set is empty.")
        return None
    loader = DataLoader(ds, batch_size=batch, shuffle=False, num_workers=2)
    model.eval()
    logits, targets = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            l = model(x).squeeze(1).cpu()
            logits.append(l)
            targets.append(y.float().cpu())
    logits = torch.cat(logits); targets = torch.cat(targets)
    T = torch.nn.Parameter(torch.ones(()))
    opt = torch.optim.LBFGS([T], lr=0.01, max_iter=50)
    def closure():
        opt.zero_grad()
        loss = nn.BCEWithLogitsLoss()(logits / T, targets)
        loss.backward()
        return loss
    opt.step(closure)
    T_val = float(T.detach())
    os.makedirs("reports", exist_ok=True)
    with open("reports/temperature.txt", "w") as f:
        f.write(f"{T_val:.6f}")
    return T_val

# ============================== UI ==============================

st.title("Chest X-ray Pneumonia — Explainable AI Dashboard")
st.caption("Research demo (non-clinical). Metrics, ROC/ECE, bootstrap CIs, misclass gallery, Grad-CAM/IG, lung-focus check.")

# Sidebar defaults: prefer bundled samples/ when deploying
default_samples = os.path.join(os.path.dirname(__file__), "samples")
default_local   = "/Users/mahalakshmibr/Downloads/Data"
default_data = default_samples if os.path.isdir(default_samples) else (default_local if os.path.isdir(default_local) else "data/raw/chest_xray")

data_dir  = st.sidebar.text_input("Dataset folder (contains train/val/test)", default_data)
ckpt_path = st.sidebar.text_input("Model checkpoint (.pt/.pth)", "outputs/best.pt")
img_size  = st.sidebar.number_input("Image size", min_value=128, max_value=512, value=224, step=32)
use_mps   = st.sidebar.checkbox("Use Apple MPS (Metal)", value=False)  # cloud is CPU; your Mac can tick this
allow_imagenet = st.sidebar.checkbox("If checkpoint missing, use ImageNet fallback (downloads once)", value=False)

# 🔁 label-swap toggle (if model predicts NORMAL prob)
swap_labels = st.sidebar.checkbox("Model outputs NORMAL prob (swap labels)", value=True)

# Threshold
if "threshold" not in st.session_state:
    st.session_state["threshold"] = 0.50
threshold = st.sidebar.slider("Decision threshold", 0.0, 1.0, float(st.session_state["threshold"]), 0.01)
st.session_state["threshold"] = threshold

# Prevalence & TTA
prev = st.sidebar.slider("Assumed prevalence (%)", 1, 50, 10, 1) / 100.0
use_tta = st.sidebar.checkbox("Use Test-Time Augmentation (TTA)", value=False)
tta_n   = st.sidebar.number_input("TTA samples", min_value=4, max_value=32, value=12, step=2, disabled=not use_tta)

# Temperature scaling
T_file = read_temperature()
if T_file:
    st.sidebar.success(f"Temperature (from file): T = {T_file:.3f}")
if st.sidebar.button("Fit temperature on validation set"):
    with st.spinner("Fitting temperature T using validation set..."):
        device_tmp = get_device(use_mps)
        model_tmp, _, _ = get_model(ckpt_path, device_tmp, allow_imagenet_fallback=allow_imagenet)
        T_fit = fit_temperature_on_val(model_tmp, data_dir, img_size, device_tmp)
        if T_fit:
            st.sidebar.success(f"Fitted T = {T_fit:.3f} and saved to reports/temperature.txt")
            T_file = T_fit

# Sidebar: read threshold from saved reports
def read_thr_from_reports():
    def grab(path, pattern):
        if os.path.exists(path):
            m = re.search(pattern, open(path).read())
            if m: return float(m.group(1))
        return None
    pat = r"Threshold@90%Spec.*:\s*([0-9.]+)"
    return grab("reports/metrics_calibrated.txt", pat) or grab("reports/metrics.txt", pat)

if st.sidebar.button("Use Threshold@90%Spec from reports"):
    thr90 = read_thr_from_reports()
    if thr90 is not None:
        st.session_state["threshold"] = thr90
        threshold = thr90
        st.sidebar.success(f"Threshold set to {thr90:.3f}")
    else:
        st.sidebar.warning("No threshold found in reports. Compute it below.")

# Quick leakage check (filename overlap)
with st.sidebar.expander("Data split overlap check"):
    def overlap_summary(root):
        splits = ["train","val","test"]
        names = {}
        for sp in splits:
            files = list_images_in_split(root, sp)
            names[sp] = set(os.path.basename(p) for p in files)
        res = {}
        for a in splits:
            for b in splits:
                if a<b:
                    res[f"{a}∩{b}"] = len(names[a] & names[b])
        return res
    try:
        ov = overlap_summary(data_dir)
        for k,v in ov.items(): st.write(f"{k}: {v}")
    except Exception:
        st.write("Could not compute (check dataset path).")

colA, colB = st.columns(2)

# ---------------- Left: Reports / Metrics ----------------
with colA:
    st.subheader("Overall metrics / reports")
    if os.path.exists("reports/metrics.txt"):
        st.code(open("reports/metrics.txt").read(), language="text")
    else:
        st.info("No reports/metrics.txt yet. Use the expander below to compute and save metrics.")

    # show existing plots if present
    if os.path.exists("reports/roc.png"):
        st.image("reports/roc.png", caption="ROC", use_container_width=True)
    if os.path.exists("reports/calibration.png"):
        st.image("reports/calibration.png", caption="Reliability diagram (calibration)", use_container_width=True)

    with st.expander("Compute test-set metrics + ROC/ECE + Threshold@90%Spec + CIs"):
        n_boot = st.number_input("Bootstrap samples for CIs", 50, 1000, 200, 50)
        st.caption("Runs inference, saves metrics to reports/metrics.txt; plots ROC & calibration; bootstraps AUC and Sens@Spec=0.90.")
        if st.button("Compute now"):
            files = list_test_images(data_dir)
            if not files:
                st.warning("No test images found. Expecting test/NORMAL and test/PNEUMONIA.")
            else:
                device_m = get_device(use_mps)
                model_m, loaded_m, source_m = get_model(ckpt_path, device_m, allow_imagenet_fallback=allow_imagenet)
                st.info(f"Model source: {source_m}; device: {device_m}")
                T_use = T_file

                y_list, p_list, path_list = [], [], []
                pb = st.progress(0.0); total = len(files)
                for i, fp in enumerate(files, 1):
                    y_list.append(label_from_path(fp)); path_list.append(fp)
                    x, _ = preprocess(fp, img_size)
                    x = x.unsqueeze(0).to(device_m)
                    prob = (predict_tta(model_m, x, n=tta_n, T=T_use)[0] if use_tta else forward_prob(model_m, x, T=T_use))
                    if swap_labels:
                        prob = 1.0 - float(prob)
                    p_list.append(prob)
                    pb.progress(i/total)

                y = np.array(y_list); p = np.array(p_list)

                # Operating points
                thr90 = thr_at_spec(y, p, 0.90)
                (tn, fp_, fn, tp), acc, prec, rec, spec, f1 = metrics_from_probs(y, p, threshold)
                (tn90, fp90, fn90, tp90), acc90, prec90, rec90, spec90, f190 = metrics_from_probs(y, p, thr90)

                # ROC
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt
                fpr, tpr = roc_curve_np(y, p); auc = auc_trapz(fpr, tpr)
                plt.figure()
                plt.plot(fpr, tpr, label=f"AUC={auc:.3f}")
                plt.plot([0,1],[0,1],'--')
                plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
                plt.legend(loc="lower right"); plt.tight_layout()
                os.makedirs("reports", exist_ok=True)
                plt.savefig("reports/roc.png", dpi=160); plt.close()

                # Calibration
                bins, ece, brier = reliability_curve(y, p, n_bins=10)
                plt.figure()
                plt.plot([0,1],[0,1],'--')
                plt.scatter(bins[:,0], bins[:,1])
                plt.xlabel("Mean predicted probability")
                plt.ylabel("Empirical fraction positive")
                plt.title(f"Reliability diagram — ECE={ece:.3f}, Brier={brier:.3f}")
                plt.tight_layout()
                plt.savefig("reports/calibration.png", dpi=160); plt.close()

                # Bootstrap CIs
                rng = np.random.default_rng(1337)
                auc_samples = []; sens90_samples = []
                pb2 = st.progress(0.0)
                N = len(y)
                for i in range(int(n_boot)):
                    idx = rng.integers(0, N, size=N)
                    y_b = y[idx]; p_b = p[idx]
                    thr_b = thr_at_spec(y_b, p_b, 0.90)
                    fpr_b, tpr_b = roc_curve_np(y_b, p_b)
                    auc_samples.append(auc_trapz(fpr_b, tpr_b))
                    sens_b = metrics_from_probs(y_b, p_b, thr_b)[3]  # recall at that thr
                    sens90_samples.append(sens_b)
                    pb2.progress((i+1)/n_boot)
                def ci(a, lo=2.5, hi=97.5):
                    q = np.quantile(a, [lo/100, hi/100])
                    return float(q[0]), float(q[1])
                auc_lo, auc_hi = ci(np.array(auc_samples))
                sens_lo, sens_hi = ci(np.array(sens90_samples))

                # Display
                st.markdown(
                    f"**Current threshold {threshold:.3f}**  \n"
                    f"Acc **{acc:.4f}** | Prec **{prec:.4f}** | Sens **{rec:.4f}** | Spec **{spec:.4f}** | F1 **{f1:.4f}**"
                )
                st.markdown(
                    f"**Threshold@90%Spec {thr90:.3f}**  \n"
                    f"Acc **{acc90:.4f}** | Prec **{prec90:.4f}** | Sens **{rec90:.4f}** | Spec **{spec90:.4f}** | F1 **{f190:.4f}**"
                )
                st.markdown(
                    f"**AUC:** {auc:.3f}  (95% CI {auc_lo:.3f} – {auc_hi:.3f})  \n"
                    f"**Sens@Spec=0.90:** {rec90:.3f}  (95% CI {sens_lo:.3f} – {sens_hi:.3f})  \n"
                    f"**ECE:** {ece:.3f}  |  **Brier:** {brier:.3f}"
                )

                # PPV/NPV at chosen prevalence using current threshold sens/spec
                ppv = (rec*prev) / max(1e-9, (rec*prev + (1-spec)*(1-prev)))
                npv = (spec*(1-prev)) / max(1e-9, ((1-rec)*prev + spec*(1-prev)))
                st.write(f"**At prevalence {prev*100:.0f}% → PPV:** {ppv:.3f} | **NPV:** {npv:.3f}")

                # Confusion matrix
                cm = pd.DataFrame([[tn, fp_],[fn, tp]],
                                  index=["True 0 (Normal)","True 1 (Pneumonia)"],
                                  columns=["Pred 0","Pred 1"])
                st.subheader("Confusion Matrix @ current threshold")
                st.table(cm)

                # Save textual report
                with open("reports/metrics.txt","w") as f:
                    f.write(
                        f"Threshold@90%Spec: {thr90:.3f}\n\n"
                        f"Current threshold {threshold:.3f}\n"
                        f"TN={tn} FP={fp_} FN={fn} TP={tp}\n"
                        f"Acc={acc:.4f} Prec={prec:.4f} Recall(Sens)={rec:.4f} Spec={spec:.4f} F1={f1:.4f}\n\n"
                        f"Metrics @ Thr90 {thr90:.3f}\n"
                        f"TN={tn90} FP={fp90} FN={fn90} TP={tp90}\n"
                        f"Acc={acc90:.4f} Prec={prec90:.4f} Recall(Sens)={rec90:.4f} Spec={spec90:.4f} F1={f190:.4f}\n\n"
                        f"AUC={auc:.3f} (95% CI {auc_lo:.3f}-{auc_hi:.3f})\n"
                        f"Sens@Spec=0.90={rec90:.3f} (95% CI {sens_lo:.3f}-{sens_hi:.3f})\n"
                        f"ECE={ece:.3f}  Brier={brier:.3f}\n"
                    )

                st.success("Saved: reports/metrics.txt, reports/roc.png, reports/calibration.png")

                # ---------- Misclassification gallery ----------
                with st.expander("Misclassification gallery (top FPs & FNs with XAI)"):
                    # collect mistakes
                    pred = (p >= threshold).astype(int)
                    idx_fp = np.where((y==0) & (pred==1))[0]
                    idx_fn = np.where((y==1) & (pred==0))[0]
                    # sort by confidence (most wrong & confident first)
                    idx_fp = idx_fp[np.argsort(-p[idx_fp])]  # high p on negatives
                    idx_fn = idx_fn[np.argsort(p[idx_fn])]   # low p on positives
                    show_k = st.number_input("Show top N per class", 1, 12, 3, 1)
                    sel_fp = idx_fp[:show_k]; sel_fn = idx_fn[:show_k]
                    if len(sel_fp)==0 and len(sel_fn)==0:
                        st.info("No misclassifications at this threshold.")
                    else:
                        device_g = device_m
                        model_g = model_m
                        def render_case(idx, title):
                            fp = path_list[idx]
                            x, pil = preprocess(fp, img_size)
                            x = x.unsqueeze(0).to(device_g)
                            gc = gradcam_map(model_g, x); ig = ig_map(model_g, x)
                            gc_img = overlay_heatmap(pil, gc, size=img_size)
                            ig_img = overlay_heatmap(pil, ig, size=img_size)
                            mask = central_lung_mask(img_size, img_size, pad=0.12)
                            gc_pos = gc - gc.min(); s = gc_pos.sum() + 1e-6
                            in_lung = float((gc_pos*mask).sum()/s); out_lung = 1.0 - in_lung
                            cols = st.columns(3)
                            with cols[0]: st.image(pil.resize((img_size,img_size)), caption=f"{title}\n{os.path.basename(fp)}", use_container_width=True)
                            with cols[1]: st.image(gc_img, caption=f"Grad-CAM (outside-lung {out_lung:.0%})", use_container_width=True)
                            with cols[2]: st.image(ig_img, caption="Integrated Gradients", use_container_width=True)

                        for i in sel_fp:
                            render_case(i, f"FP  (y=0, p={p[i]:.3f} ≥ thr {threshold:.2f})")
                        for i in sel_fn:
                            render_case(i, f"FN  (y=1, p={p[i]:.3f} < thr {threshold:.2f})")

                if st.button("Set threshold to 90% specificity"):
                    st.session_state["threshold"] = thr90
                    st.success(f"Threshold set to {thr90:.3f}")

# ---------------- Right: Model & Single-image ----------------
with colB:
    st.subheader("Model & single-image prediction")

    device = get_device(use_mps)
    model, loaded, source = get_model(ckpt_path, device, allow_imagenet_fallback=allow_imagenet)
    st.success(f"Model ready on {device} ✓")
    if loaded: st.success(f"Loaded checkpoint: {ckpt_path}")
    else:      st.warning(f"Checkpoint not found: {ckpt_path}. Using {source} init.")

    files = list_test_images(data_dir)

    up = st.file_uploader("Or upload one chest X-ray (PNG/JPG)", type=["png","jpg","jpeg"])
    choice = st.selectbox("Pick a test image (if available)", files, index=0 if files else None)

    target_img = None; src_label = None
    if up is not None:
        target_img = up
    elif choice:
        target_img = choice
        src_label = "NORMAL" if label_from_path(choice)==0 else "PNEUMONIA"

    if target_img is None:
        st.info("Provide an image (upload above) or ensure your dataset test folders have images.")
    else:
        x, pil = preprocess(target_img, img_size)
        x = x.unsqueeze(0).to(device)
        T_use = T_file
        if use_tta:
            prob, prob_std = predict_tta(model, x, n=tta_n, T=T_use)
        else:
            prob = forward_prob(model, x, T=T_use); prob_std = 0.0
        if swap_labels:
            prob = 1.0 - float(prob)

        pred = "PNEUMONIA" if prob >= threshold else "NORMAL"
        st.image(pil.resize((img_size, img_size)), caption=f"Original{' — GT: '+src_label if src_label else ''}", use_container_width=True)
        st.write(f"**Prediction:** {pred} | **p:** {prob:.3f} ± {prob_std:.3f} | **Threshold:** {threshold:.2f}")

        if st.button("Generate explanations (Grad-CAM + IG)"):
            gc = gradcam_map(model, x)
            ig = ig_map(model, x)
            c1, c2 = st.columns(2)
            with c1: st.image(overlay_heatmap(pil, gc, size=img_size), caption="Grad-CAM", use_container_width=True)
            with c2: st.image(overlay_heatmap(pil, ig, size=img_size), caption="Integrated Gradients", use_container_width=True)

            # Lung-focus sanity check
            mask = central_lung_mask(img_size, img_size, pad=0.12)
            outside = 1.0 - mask
            gc_pos = gc - gc.min()
            s = gc_pos.sum() + 1e-6
            in_lung = float((gc_pos * mask).sum() / s)
            out_lung = float((gc_pos * outside).sum() / s)
            st.write(f"**Saliency in lung box:** {in_lung:.2%} | **Outside-lung:** {out_lung:.2%}")
            if out_lung > 0.40:
                st.warning("High outside-lung saliency → potential spurious cues (edges/devices/markers).")

# ============================== Footer ==============================
st.caption("Saved reports: metrics.txt, roc.png, calibration.png in ./reports. Use 'Set threshold to 90% specificity' for a clinically meaningful operating point.")