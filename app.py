# app.py
import streamlit as st
import tempfile, os, zipfile
import torch
import numpy as np
import cv2

from src.io_utils import load_case
from src.preprocess import make_4ch_slice, normalize_img
from src.model_unet import get_unet
from src.visualize import get_gradcam_mask, overlay_mask_on_image
from src.report.report import compose_report
from src.report.report_builder import build_report_content

st.set_page_config(layout="wide")
st.title("AI Brain Tumor Preoperative Assistant (MRI-only)")

uploaded = st.file_uploader("Upload a ZIP of a case folder (flair,t1,t1ce,t2 and optional seg)", type=['zip'])
if uploaded:
    tmpdir = tempfile.TemporaryDirectory()
    zpath = os.path.join(tmpdir.name, "case.zip")
    with open(zpath, "wb") as f:
        f.write(uploaded.getbuffer())
    with zipfile.ZipFile(zpath, 'r') as z:
        z.extractall(tmpdir.name)
    # find subfolder with nifti files
    # assume extracted root contains files or a single folder
    candidates = [os.path.join(tmpdir.name, p) for p in os.listdir(tmpdir.name)]
    case_folder = None
    for c in candidates:
        if os.path.isdir(c):
            case_folder = c
            break
    if case_folder is None:
        case_folder = tmpdir.name
    st.write("Loaded folder:", case_folder)
    imgs, seg = load_case(case_folder)
    # load model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = get_unet().to(device)
    ckpt = "checkpoints/best.pt"
    if not os.path.exists(ckpt):
        st.error("Model checkpoint checkpoints/best.pt not found. Train and place checkpoint first.")
    else:
        model.load_state_dict(torch.load(ckpt, map_location=device))
        # pick slice with max flair intensity sum (approx center)
        flair = imgs['_flair.nii']
        z_cand = np.argmax(flair.sum(axis=(0,1)))
        x = make_4ch_slice(imgs, z_cand, target_size=256)
        x_t = torch.from_numpy(x).unsqueeze(0).to(device)
        with torch.no_grad():
            pred = model(x_t)[0,0].cpu().numpy()
        # overlay on flair channel
        flair_chan = x[0]
        ov = overlay_mask_on_image(flair_chan, pred)
        st.image(cv2.cvtColor(ov, cv2.COLOR_BGR2RGB), caption=f"Segmentation overlay (slice {z_cand})")
        # Grad-CAM
        cam = get_gradcam_mask(model, x_t, device=device)
        # normalize cam
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        cam_rgb = cv2.applyColorMap((cam*255).astype('uint8'), cv2.COLORMAP_JET)
        cam_blend = cv2.addWeighted(cv2.cvtColor(((flair_chan - flair_chan.min())/(flair_chan.max()-flair_chan.min()+1e-8)*255.0).astype('uint8'), cv2.COLOR_GRAY2BGR), 0.6, cam_rgb, 0.4, 0)
        st.image(cv2.cvtColor(cam_blend, cv2.COLOR_BGR2RGB), caption="Grad-CAM attention map")
        
        # prepare report
        # metadata = {'case_id': os.path.basename(case_folder)}
        
        # basic findings
        # tumor_present = pred.mean() > 0.001
        # summary = f"Tumor present: {'Yes' if tumor_present else 'No'}. Slice index: {z_cand}."
        # detailed = "Automated segmentation performed. This is an AI-assisted report; for any clinical decision consult a radiologist."
        # images = {'Segmentation overlay': cv2.cvtColor(ov, cv2.COLOR_BGR2RGB), 'Attention map': cv2.cvtColor(cam_blend, cv2.COLOR_BGR2RGB)}

        metadata, findings, images = build_report_content(case_folder, pred, ov, cam_blend, z_cand)
        outpdf = os.path.join(tmpdir.name, "report.pdf")

        compose_report(outpdf, metadata, findings, images)
        
        with open(outpdf, "rb") as f:
            st.download_button("Download PDF report", f, file_name="report.pdf")