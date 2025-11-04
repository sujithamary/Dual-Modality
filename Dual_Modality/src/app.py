# app.py
import streamlit as st
import tempfile, os
from src.io import load_case
from src.preprocess import make_4ch_slice
from src.model_unet import get_unet
import torch
import numpy as np
from src.report import compose_report
from src.visualize import overlay_mask_on_slice, get_attention_heatmap  # you create these

st.title("AI Brain Tumor Preoperative Assistant (MRI only)")

uploaded_dir = st.file_uploader("Upload folder (zip of FLAIR, T1, T1CE, T2 and optional seg)", type=['zip'])
if uploaded_dir:
    import zipfile
    tmp = tempfile.TemporaryDirectory()
    with zipfile.ZipFile(uploaded_dir) as z:
        z.extractall(tmp.name)
    # expect the folder contains the nifti files
    case_folder = tmp.name
    imgs, seg = load_case(case_folder)
    st.write("Loaded modalities.")
    # load model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = get_unet().to(device)
    model.load_state_dict(torch.load("checkpoints/best.pt", map_location=device))
    model.eval()
    # run inference on central slice / slices
    z_c = imgs['_flair.nii'].shape[2]//2
    x = make_4ch_slice(imgs, z_c)
    x_t = torch.from_numpy(x).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = model(x_t)[0,0].cpu().numpy()
    # visualize
    from src.visualize import overlay_mask_on_slice, plot_attention_gradcam
    ov = overlay_mask_on_slice(x[0], pred)  # use flair channel for background
    st.image(ov, caption="Segmentation overlay")
    # attention map
    att = plot_attention_gradcam(model, x_t)  # implement function to return np HxW heatmap
    st.image(att, caption="Attention / GradCAM")
    # prepare report
    metadata = {'case_id': os.path.basename(case_folder)}
    findings = {'summary': "Tumor present ..." , 'detailed': "Long text..."}
    images = {'seg_overlay': ov, 'attention': att}
    out_pdf = os.path.join(tmp.name, 'report.pdf')
    compose_report(out_pdf, metadata, findings, images)
    with open(out_pdf, "rb") as f:
        st.download_button("Download PDF report", f, file_name="report.pdf")
