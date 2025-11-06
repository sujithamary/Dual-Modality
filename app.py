# # app.py (replace your existing file with this)
# import streamlit as st
# import tempfile, os, zipfile, time, hashlib
# import numpy as np
# import cv2
# import torch

# # import local modules (assumes src is a package or in PYTHONPATH)
# from src.io_utils import load_case
# from src.preprocess import make_4ch_slice
# from src.model_unet import get_unet
# from src.visualize import get_gradcam_mask, overlay_mask_on_image
# from src.report import compose_report_full  # replace with your enhanced composer if present
# from src.measurements import binary_mask_from_prob, centroid_voxel, bounding_box, compute_volume_cm3

# st.set_page_config(layout="wide")
# st.title("AI Brain Tumor Preoperative Assistant (MRI-only)")

# # Config
# CHECKPOINT_PATH = "checkpoints/best.pt"
# TARGET_SIZE = 256
# DEFAULT_THRESHOLD = 0.5

# # Helper: load model safely
# @st.cache_resource
# def load_model(ckpt_path, device):
#     model = get_unet().to(device)
#     model.eval()
#     if os.path.exists(ckpt_path):
#         try:
#             model.load_state_dict(torch.load(ckpt_path, map_location=device))
#         except Exception as e:
#             st.warning(f"Could not load checkpoint: {e}")
#     else:
#         st.warning(f"Checkpoint not found at {ckpt_path}. The model will be untrained.")
#     return model

# uploaded = st.file_uploader("Upload a ZIP of a case folder (flair,t1,t1ce,t2 and optional seg)", type=['zip'])
# if not uploaded:
#     st.info("Upload a ZIP file containing the MRI case folder (FLAIR, T1, T1CE, T2).")
#     st.stop()

# # extract uploaded zip to temp dir
# tmpdir = tempfile.TemporaryDirectory()
# zpath = os.path.join(tmpdir.name, "case.zip")
# with open(zpath, "wb") as f:
#     f.write(uploaded.getbuffer())
# with zipfile.ZipFile(zpath, 'r') as z:
#     z.extractall(tmpdir.name)

# # find case folder (if zip contains a single folder, choose that)
# extracted_items = os.listdir(tmpdir.name)
# case_folder = tmpdir.name
# for it in extracted_items:
#     p = os.path.join(tmpdir.name, it)
#     if os.path.isdir(p):
#         case_folder = p
#         break

# st.write("Loaded folder:", case_folder)

# # load images (NIfTI arrays) and optional seg
# try:
#     imgs, seg = load_case(case_folder)
# except Exception as e:
#     st.error(f"Failed to load NIfTI files from folder: {e}")
#     st.stop()

# # device
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# st.write("Using device:", device)

# # choice: single-slice (fast) or full-volume (slower)
# mode = st.radio("Inference mode", ("Representative slice (fast)", "Full volume (slower, computes volume)"))

# model = load_model(CHECKPOINT_PATH, device)

# # pick representative slice: use brightest FLAIR slice (excluding zeros)
# flair_arr = None
# for k in imgs.keys():
#     if "flair" in k.lower():
#         flair_arr = imgs[k]
#         break
# if flair_arr is None:
#     flair_arr = list(imgs.values())[0]

# # find slice with largest sum (approx central tumor slice) but ignore blank slices
# slice_sums = np.array([flair_arr[:,:,z].sum() for z in range(flair_arr.shape[2])])
# if slice_sums.sum() == 0:
#     z_cand = flair_arr.shape[2] // 2
# else:
#     z_cand = int(np.argmax(slice_sums))

# st.write(f"Representative slice chosen: {z_cand}")

# # run inference
# start_time = time.time()
# results = {}
# if mode == "Representative slice (fast)":
#     x = make_4ch_slice(imgs, z_cand, target_size=TARGET_SIZE)
#     x_t = torch.from_numpy(x).unsqueeze(0).to(device)
#     with torch.no_grad():
#         pred = model(x_t)[0,0].cpu().numpy()
#     results[z_cand] = pred
# else:
#     Z = list(imgs.values())[0].shape[2]
#     for z in range(Z):
#         x = make_4ch_slice(imgs, z, target_size=TARGET_SIZE)
#         x_t = torch.from_numpy(x).unsqueeze(0).to(device)
#         with torch.no_grad():
#             pred = model(x_t)[0,0].cpu().numpy()
#         results[z] = pred

# inference_ms = (time.time() - start_time) * 1000 / max(1, len(results))
# st.write(f"Inference done — avg {inference_ms:.1f} ms per slice (approx)")

# # pick best slice (largest predicted tumor area)
# best_z = max(results.keys(), key=lambda z: results[z].sum())
# pred_best = results[best_z]
# st.write(f"Slice with largest predicted region: {best_z}")

# # compute mask & basic measurements
# thr = st.slider("Segmentation threshold", 0.1, 0.9, DEFAULT_THRESHOLD, 0.05)
# binmask = binary_mask_from_prob(pred_best, thr=thr)
# cent = centroid_voxel(binmask)
# bbox = bounding_box(binmask)
# max_area_px = int(binmask.sum())
# conf_mean = float(pred_best.mean())

# measurements = {
#     'tumor_present': bool(max_area_px > 10),
#     'slice_peak': int(best_z),
#     'max_area_px': int(max_area_px),
#     'max_area_mm2': None,
#     'volume_cm3': None,
#     'centroid_voxel': (float(cent[1]) if cent is not None else None, float(cent[0]) if cent is not None else None, int(best_z)),
#     'bbox': bbox,
#     'confidence_mean': conf_mean
# }

# # if full-volume and seg available, compute approximate volume
# if mode != "Representative slice (fast)":
#     # build 3D binary mask (Z,H,W)
#     Z = list(imgs.values())[0].shape[2]
#     seg3d = np.zeros((Z, TARGET_SIZE, TARGET_SIZE), dtype=np.uint8)
#     for z in range(Z):
#         seg3d[z] = (results[z] >= thr).astype(np.uint8)
#     # If your io.load_case returned nibabel objects you can access affine; if not, skip physical volume
#     try:
#         # try to find an affine if loader provided nib objects; earlier loader returned arrays, so this may fail
#         nifti_candidate = None
#         # search for a nifti in folder
#         for fname in os.listdir(case_folder):
#             if fname.endswith('.nii') or fname.endswith('.nii.gz'):
#                 import nibabel as nib
#                 nifti_candidate = nib.load(os.path.join(case_folder, fname))
#                 break
#         if nifti_candidate is not None:
#             affine = nifti_candidate.affine
#             # seg3d shape currently (Z,H,W) but compute_volume_cm3 expects 3D mask aligned with affine
#             # This is approximate because we resized to TARGET_SIZE; volume will be approximate.
#             # For more accurate volume, perform inference at native resolution.
#             measurements['volume_cm3'] = compute_volume_cm3(seg3d, affine)
#     except Exception as e:
#         st.info("Could not compute physical volume (affine missing or resize mismatch).")

# # prepare overlay images for UI
# flair_slice = make_4ch_slice(imgs, best_z, target_size=TARGET_SIZE)[0]  # flair channel
# overlay_rgb = overlay_mask_on_image(flair_slice, pred_best, alpha=0.5)

# # Grad-CAM (safe call)
# try:
#     x_for_cam = torch.from_numpy(make_4ch_slice(imgs, best_z, target_size=TARGET_SIZE)).unsqueeze(0)
#     cam_map = get_gradcam_mask(model, x_for_cam, target_mask=None, device=('cuda' if torch.cuda.is_available() else 'cpu'))
#     cam_norm = (cam_map - cam_map.min()) / (cam_map.max() - cam_map.min() + 1e-8)
#     cam_rgb = cv2.applyColorMap((cam_norm*255).astype('uint8'), cv2.COLORMAP_JET)
#     # blend with flair
#     flair_norm = flair_slice - flair_slice.min()
#     if flair_norm.max() > 0:
#         flair_norm = (flair_norm / flair_norm.max() * 255).astype('uint8')
#     else:
#         flair_norm = (flair_norm * 255).astype('uint8')
#     flair_rgb = cv2.cvtColor(flair_norm, cv2.COLOR_GRAY2BGR)
#     cam_blend = cv2.addWeighted(flair_rgb, 0.6, cam_rgb, 0.4, 0)
# except Exception as e:
#     st.warning(f"Grad-CAM failed: {e}")
#     cam_blend = flair_rgb if 'flair_rgb' in locals() else np.stack([flair_slice]*3, axis=-1)

# # Display results in Streamlit
# st.header("Results")
# col1, col2 = st.columns(2)
# with col1:
#     st.image(cv2.cvtColor(overlay_rgb, cv2.COLOR_BGR2RGB), caption=f"Segmentation overlay (slice {best_z})", use_column_width=True)
# with col2:
#     st.image(cv2.cvtColor(cam_blend, cv2.COLOR_BGR2RGB), caption="Grad-CAM attention map", use_column_width=True)

# st.subheader("Measurements")
# st.write(measurements)

# # Build report content
# metadata = {'case_id': os.path.basename(case_folder), 'patient_info': ''}
# findings = {
#     'summary': f"Tumor present: {'Yes' if measurements['tumor_present'] else 'No'}. Representative slice: {best_z}. Model confidence (mean prob): {measurements['confidence_mean']:.3f}",
#     'detailed': "Automated segmentation performed on FLAIR/T1/T1CE/T2. This is an AI-assisted result and requires radiologist confirmation."
# }

# # technical info
# weights_sha = "N/A"
# if os.path.exists(CHECKPOINT_PATH):
#     try:
#         weights_sha = hashlib.sha256(open(CHECKPOINT_PATH,'rb').read()).hexdigest()[:12]
#     except Exception:
#         weights_sha = "unable-to-hash"

# technical = {
#     'model_name': 'UNet-resnet34',
#     'model_version': '1.0',
#     'val_dice': 0.0,
#     'weights_sha256': weights_sha,
#     'inference_ms': round(inference_ms, 1),
#     'preproc': f"4-channel axial slices resized to {TARGET_SIZE}x{TARGET_SIZE}, per-slice normalization"
# }

# images_dict = {
#     f"Segmentation overlay (slice {best_z})": cv2.cvtColor(overlay_rgb, cv2.COLOR_BGR2RGB),
#     f"Grad-CAM attention (slice {best_z})": cv2.cvtColor(cam_blend, cv2.COLOR_BGR2RGB)
# }

# # Generate PDF
# if st.button("Generate & Download PDF report"):
#     out_pdf = os.path.join(tmpdir.name, f"report_{metadata['case_id']}.pdf")
#     compose_report_full(out_pdf, metadata, findings, measurements, images_dict, technical)
#     with open(out_pdf, "rb") as f:
#         st.download_button("Download PDF", f, file_name=f"report_{metadata['case_id']}.pdf")

# # Cleanup tempdir on session end is left to OS; it's temporary

# app.py
"""
Unified Streamlit app for MRI tumor detection:
- robust imports
- representative-slice or full-volume inference
- human-readable clinical summary + detailed findings
- Grad-CAM and overlay display
- PDF report generation via compose_report_full
"""

import os, sys, tempfile, zipfile, time, hashlib
import numpy as np
import cv2
import torch
import streamlit as st

# ensure project root and src on path (helps when running from different cwd)
proj_root = os.path.dirname(os.path.abspath(__file__))
if proj_root not in sys.path:
    sys.path.insert(0, proj_root)
src_path = os.path.join(proj_root, "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# imports from local src
from src.io_utils import load_case
from src.preprocess import make_4ch_slice
from src.model_unet import get_unet
from src.visualize import get_gradcam_mask, overlay_mask_on_image
from src.report import compose_report_full
from src.measurements import binary_mask_from_prob, centroid_voxel, bounding_box, compute_volume_cm3
# human-readable helper (inlined below if not present)
try:
    from src.reporting_helpers import human_readable_measurements
except Exception:
    # provide fallback inline function if module missing
    def hemisphere_from_centroid(cx, img_width):
        try:
            cx = float(cx)
        except:
            return "unknown"
        return "left" if cx < (img_width/2) else "right"

    def human_readable_measurements(measurements, imgs, best_z, pixel_dims=None):
        tumor_present = measurements.get('tumor_present', False)
        slice_peak = measurements.get('slice_peak', best_z)
        area_px = measurements.get('max_area_px', None)
        area_mm2 = measurements.get('max_area_mm2', None)
        volume_cm3 = measurements.get('volume_cm3', None)
        centroid = measurements.get('centroid_voxel', None)
        bbox = measurements.get('bbox', None)
        conf = measurements.get('confidence_mean', None)
        sample_mod = None
        for k in imgs.keys():
            sample_mod = imgs[k]; break
        H, W = sample_mod.shape[:2]
        hemi = "unknown"
        if centroid is not None and len(centroid) >= 2:
            cx = centroid[0]
            if cx is None and len(centroid) >= 2:
                cx = centroid[1]
            hemi = hemisphere_from_centroid(cx, W)
        if not tumor_present:
            summary = "No tumorous region detected by the automated model in this MRI volume. A radiologist review is recommended if clinical suspicion persists."
        else:
            s1 = f"Automated analysis detected a lesion likely representing tumour tissue. The largest cross-section appears at axial slice {slice_peak}."
            s2 = f"The lesion is located in the {hemi} cerebral hemisphere."
            s3 = f"Maximum cross-sectional area (measured on the representative slice): {area_mm2:.1f} mm²." if area_mm2 is not None else (f"Maximum cross-sectional area (measured on the representative slice): {area_px:,} pixels (approx.)." if area_px is not None else "")
            s4 = f"Estimated tumor volume (3D, approximate): {volume_cm3:.2f} cm³." if volume_cm3 is not None else "Estimated tumor volume: not available (computed only when full-volume inference is selected)."
            conf_text = ""
            if conf is not None:
                if conf >= 0.6:
                    conf_text = "Model confidence is high."
                elif conf >= 0.35:
                    conf_text = "Model confidence is moderate."
                else:
                    conf_text = "Model confidence is low; boundaries may be uncertain."
            summary = " ".join([part for part in [s1, s2, s3, s4, conf_text] if part])
        details = []
        if tumor_present:
            details.append(f"- Representative axial slice: {slice_peak}.")
            if centroid is not None:
                cx = centroid[0] if len(centroid)>0 else "NA"
                cy = centroid[1] if len(centroid)>1 else "NA"
                cz = int(centroid[2]) if len(centroid)>2 else slice_peak
                details.append(f"- Approximate centroid (voxel coords): x={cx:.1f}, y={cy:.1f}, z={cz}.")
            if bbox is not None:
                rmin, rmax, cmin, cmax = bbox
                details.append(f"- Approx. bounding box on representative slice (rows,cols): row {rmin}–{rmax}, col {cmin}–{cmax}.")
            if area_mm2 is not None:
                details.append(f"- Max cross-sectional area: {area_mm2:.1f} mm² (slice {slice_peak}).")
            elif area_px is not None:
                details.append(f"- Max cross-sectional area: {area_px:,} pixels (slice {slice_peak}).")
            if volume_cm3 is not None:
                details.append(f"- Estimated tumor volume: {volume_cm3:.2f} cm³ (computed from full-volume inference).")
        else:
            details.append("- No tumour-like region exceeded the detection threshold in this study.")
        interpretation = []
        if tumor_present:
            interpretation.append("IMPRESSION:")
            interpretation.append("1. Automated segmentation indicates a space-occupying lesion consistent with a tumor. Imaging pattern should be correlated with clinical history and contrast-enhanced sequences by a radiologist.")
            interpretation.append("2. Recommend formal radiology review, neuroradiology/neurosurgical consultation if clinically indicated, and histopathological confirmation where appropriate.")
            interpretation.append("3. Limitations: The automated measurement is approximate and depends on image acquisition and preprocessing. For surgical planning, perform dedicated volumetric measurements at native resolution.")
        else:
            interpretation.append("IMPRESSION:")
            interpretation.append("1. No tumor detected by the automated tool. If clinical concern remains, proceed with radiologist review and further imaging as needed.")
        summary_text = summary
        details_text = "\n".join(details)
        interpretation_text = "\n".join(interpretation)
        plain_paragraph = summary_text + "\n\n" + "Key findings:\n" + details_text + "\n\n" + interpretation_text
        return {
            "summary_text": summary_text,
            "details_text": details_text,
            "interpretation_text": interpretation_text,
            "plain_paragraph": plain_paragraph
        }

# Streamlit page config
st.set_page_config(layout="wide", page_title="AI Brain Tumor Preop Assistant")
st.title("AI Brain Tumor Preoperative Assistant (MRI-only)")

# Config constants
CHECKPOINT_PATH = "checkpoints/best.pt"
TARGET_SIZE = 256
DEFAULT_THRESHOLD = 0.5

@st.cache_resource
def load_model(ckpt_path, device):
    model = get_unet().to(device)
    model.eval()
    if os.path.exists(ckpt_path):
        try:
            model.load_state_dict(torch.load(ckpt_path, map_location=device))
        except Exception as e:
            st.warning(f"Could not load checkpoint: {e}")
    else:
        st.warning(f"Checkpoint not found at {ckpt_path}. Using untrained model for demo.")
    return model

uploaded = st.file_uploader("Upload a ZIP of a case folder (FLAIR, T1, T1CE, T2 and optional seg)", type=['zip'])
if not uploaded:
    st.info("Upload a ZIP file containing the MRI case folder (FLAIR, T1, T1CE, T2).")
    st.stop()

tmpdir = tempfile.TemporaryDirectory()
zpath = os.path.join(tmpdir.name, "case.zip")
with open(zpath, "wb") as f:
    f.write(uploaded.getbuffer())
with zipfile.ZipFile(zpath, 'r') as z:
    z.extractall(tmpdir.name)

# find the top-level extracted folder if present
case_folder = tmpdir.name
entries = os.listdir(tmpdir.name)
for e in entries:
    p = os.path.join(tmpdir.name, e)
    if os.path.isdir(p):
        case_folder = p
        break

st.write("Loaded folder:", case_folder)

# load NIfTI arrays
try:
    imgs, seg = load_case(case_folder)
except Exception as e:
    st.error(f"Failed to read NIfTI files in folder: {e}")
    st.stop()

# device selection
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
st.write("Using device:", device)

# inference mode choice
mode = st.radio("Inference mode", ("Representative slice (fast)", "Full volume (slower, computes volume)"))

# load model
model = load_model(CHECKPOINT_PATH, device)

# pick representative slice (prefer FLAIR)
flair_arr = None
for k in imgs.keys():
    if 'flair' in k.lower():
        flair_arr = imgs[k]; break
if flair_arr is None:
    flair_arr = list(imgs.values())[0]

slice_sums = np.array([flair_arr[:,:,z].sum() for z in range(flair_arr.shape[2])])
z_cand = int(np.argmax(slice_sums)) if slice_sums.sum() > 0 else (flair_arr.shape[2] // 2)


# run inference
start_time = time.time()
results = {}
if mode == "Representative slice (fast)":
    x = make_4ch_slice(imgs, z_cand, target_size=TARGET_SIZE)
    x_t = torch.from_numpy(x).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = model(x_t)[0,0].cpu().numpy()
    results[z_cand] = pred
else:
    Z = list(imgs.values())[0].shape[2]
    for z in range(Z):
        x = make_4ch_slice(imgs, z, target_size=TARGET_SIZE)
        x_t = torch.from_numpy(x).unsqueeze(0).to(device)
        with torch.no_grad():
            pred = model(x_t)[0,0].cpu().numpy()
        results[z] = pred

inference_ms = (time.time() - start_time) * 1000 / max(1, len(results))
st.write(f"Inference done — avg {inference_ms:.1f} ms per slice (approx)")

# pick best slice and compute measurements
best_z = max(results.keys(), key=lambda z: results[z].sum())
pred_best = results[best_z]

# --- visualize probability distribution (for analysis) ---
# import matplotlib.pyplot as plt

# st.subheader("Model output distribution (prediction confidence map)")
# fig, ax = plt.subplots(figsize=(5,3))
# ax.hist(pred_best.flatten(), bins=50, color='gray', edgecolor='black')
# ax.set_title("Histogram of predicted probabilities")
# ax.set_xlabel("Probability value (0 = non-tumor, 1 = tumor)")
# ax.set_ylabel("Pixel count")
# st.pyplot(fig) --> important graph


# threshold slider
thr = st.slider("Segmentation threshold", 0.1, 0.9, DEFAULT_THRESHOLD, 0.05)
binmask = binary_mask_from_prob(pred_best, thr=thr)
cent = centroid_voxel(binmask)
bbox = bounding_box(binmask)
max_area_px = int(binmask.sum())
conf_mean = float(pred_best.mean())

measurements = {
    'tumor_present': bool(max_area_px > 10),
    'slice_peak': int(best_z),
    'max_area_px': int(max_area_px),
    'max_area_mm2': None,
    'volume_cm3': None,
    'centroid_voxel': (float(cent[1]) if cent is not None else None, float(cent[0]) if cent is not None else None, int(best_z)),
    'bbox': bbox,
    'confidence_mean': conf_mean
}

# if full-volume, compute approx 3D volume (requires affine for real-world mm^3 -> cm^3)
if mode != "Representative slice (fast)":
    Z = list(imgs.values())[0].shape[2]
    seg3d = np.zeros((Z, TARGET_SIZE, TARGET_SIZE), dtype=np.uint8)
    for z in range(Z):
        seg3d[z] = (results[z] >= thr).astype(np.uint8)
    # attempt to compute volume using first nifti affine (best-effort; our loader returns arrays; we try to load a file for affine)
    try:
        import nibabel as nib
        nifti_candidate = None
        for fname in os.listdir(case_folder):
            if fname.endswith('.nii') or fname.endswith('.nii.gz'):
                nifti_candidate = nib.load(os.path.join(case_folder, fname))
                break
        if nifti_candidate is not None:
            affine = nifti_candidate.affine
            measurements['volume_cm3'] = compute_volume_cm3(seg3d, affine)
    except Exception:
        measurements['volume_cm3'] = None

# prepare overlay image
flair_slice = make_4ch_slice(imgs, best_z, target_size=TARGET_SIZE)[0]
overlay_rgb = overlay_mask_on_image(flair_slice, pred_best, alpha=0.5)

# Grad-CAM (safe)
try:
    x_for_cam = torch.from_numpy(make_4ch_slice(imgs, best_z, target_size=TARGET_SIZE)).unsqueeze(0)
    cam_map = get_gradcam_mask(model, x_for_cam, target_mask=None, device=('cuda' if torch.cuda.is_available() else 'cpu'))
    cam_norm = (cam_map - cam_map.min()) / (cam_map.max() - cam_map.min() + 1e-8)
    cam_rgb = cv2.applyColorMap((cam_norm*255).astype('uint8'), cv2.COLORMAP_JET)
    flair_norm = flair_slice - flair_slice.min()
    flair_norm = (flair_norm / flair_norm.max() * 255).astype('uint8') if flair_norm.max()>0 else (flair_norm*255).astype('uint8')
    flair_rgb = cv2.cvtColor(flair_norm, cv2.COLOR_GRAY2BGR)
    cam_blend = cv2.addWeighted(flair_rgb, 0.6, cam_rgb, 0.4, 0)
except Exception as e:
    st.warning(f"Grad-CAM failed: {e}")
    flair_rgb = cv2.cvtColor(((flair_slice - flair_slice.min())/(flair_slice.max()-flair_slice.min()+1e-8)*255).astype('uint8'), cv2.COLOR_GRAY2BGR)
    cam_blend = flair_rgb

# Display
st.header("Results")
col1, col2 = st.columns(2)
with col1:
    st.image(cv2.cvtColor(overlay_rgb, cv2.COLOR_BGR2RGB), caption=f"Segmentation overlay (slice {best_z})", use_container_width=True)
with col2:
    st.image(cv2.cvtColor(cam_blend, cv2.COLOR_BGR2RGB), caption="Grad-CAM attention map", use_container_width=True)

# Clinical summary for humans
readable = human_readable_measurements(measurements, imgs, best_z, pixel_dims=None)
st.markdown("## Clinical Summary (plain language)")
st.markdown(f"**{readable['summary_text']}**")

with st.expander("Detailed findings and interpretation (click to open)"):
    st.markdown("**Findings:**")
    st.markdown(readable['details_text'].replace("\n", "  \n"))
    st.markdown("**Interpretation & Recommendations:**")
    st.markdown(readable['interpretation_text'].replace("\n", "  \n"))

st.subheader("Measurements")
st.json(measurements)

# Prepare PDF content
metadata = {'case_id': os.path.basename(case_folder), 'patient_info': ''}
findings = {'summary': readable['summary_text'], 'detailed': readable['plain_paragraph']}
weights_sha = "N/A"
if os.path.exists(CHECKPOINT_PATH):
    try:
        weights_sha = hashlib.sha256(open(CHECKPOINT_PATH,'rb').read()).hexdigest()[:12]
    except Exception:
        weights_sha = "unable-to-hash"

technical = {
    'model_name': 'UNet-resnet34',
    'model_version': '1.0',
    'val_dice': 0.0,
    'weights_sha256': weights_sha,
    'inference_ms': round(inference_ms, 1),
    'preproc': f"4-channel axial slices resized to {TARGET_SIZE}x{TARGET_SIZE}, per-slice normalization"
}

images_dict = {
    f"Segmentation overlay (slice {best_z})": cv2.cvtColor(overlay_rgb, cv2.COLOR_BGR2RGB),
    f"Grad-CAM attention (slice {best_z})": cv2.cvtColor(cam_blend, cv2.COLOR_BGR2RGB)
}

# Download button
if st.button("Generate & Download PDF report"):
    out_pdf = os.path.join(tmpdir.name, f"report_{metadata['case_id']}.pdf")
    compose_report_full(out_pdf, metadata, findings, measurements, images_dict, technical)
    with open(out_pdf, "rb") as f:
        st.download_button("Download PDF", f, file_name=f"report_{metadata['case_id']}.pdf")
