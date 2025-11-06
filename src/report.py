# # src/report.py  (add or replace compose_report)
# from reportlab.lib.pagesizes import A4
# from reportlab.pdfgen import canvas
# from reportlab.lib.utils import ImageReader
# import matplotlib.pyplot as plt
# import io
# import datetime
# import hashlib
# import numpy as np

# def npimg_to_buf(img_np, cmap=None):
#     buf = io.BytesIO()
#     if img_np.ndim == 2:
#         plt.imsave(buf, img_np, cmap=cmap or 'gray', format='png')
#     else:
#         plt.imsave(buf, img_np, format='png')
#     buf.seek(0)
#     return buf

# def compose_report_full(output_pdf_path, metadata, findings, measurements, images_dict, technical):
#     # metadata: dict with case_id, patient fields
#     # findings: dict with 'summary' and 'detailed' (strings)
#     # measurements: dict with numeric metrics
#     # images_dict: {'seg_overlay': np.array, 'attention': np.array, ...}
#     # technical: dict with model_version, val_dice, weights_sha256, inference_ms
#     c = canvas.Canvas(output_pdf_path, pagesize=A4)
#     w, h = A4
#     margin_x = 40
#     # header
#     c.setFont("Helvetica-Bold", 16)
#     c.drawString(margin_x, h-50, "AI-Assisted MRI Brain Tumor Report")
#     c.setFont("Helvetica", 9)
#     c.drawRightString(w-40, h-50, f"Generated: {datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")
#     c.line(margin_x, h-60, w-margin_x, h-60)

#     # patient info + summary
#     c.setFont("Helvetica-Bold", 12)
#     c.drawString(margin_x, h-80, f"Case: {metadata.get('case_id','-')}")
#     c.setFont("Helvetica", 10)
#     c.drawString(margin_x, h-95, metadata.get('patient_info',''))
#     y = h-120
#     c.setFont("Helvetica-Bold", 11)
#     c.drawString(margin_x, y, "SUMMARY:")
#     c.setFont("Helvetica", 10)
#     text = c.beginText(margin_x, y-18)
#     text.textLines(findings.get('summary','-'))
#     c.drawText(text)

#     # Measurements box
#     box_y = y-110
#     c.setFont("Helvetica-Bold", 11)
#     c.drawString(margin_x, box_y+90, "MEASUREMENTS")
#     c.setFont("Helvetica", 10)
#     lines = [
#         f"Tumor present: {measurements.get('tumor_present')}",
#         f"Representative slice: {measurements.get('slice_peak')}",
#         f"Max cross-sectional area: {measurements.get('max_area_mm2', 'N/A'):.2f} mm²" if measurements.get('max_area_mm2') is not None else "Max area: N/A",
#         f"Estimated tumor volume: {measurements.get('volume_cm3', 'N/A'):.2f} cm³" if measurements.get('volume_cm3') is not None else "Volume: N/A",
#         f"Centroid (voxel): {measurements.get('centroid_voxel')}",
#         f"Bounding box (rmin,rmax,cmin,cmax): {measurements.get('bbox')}",
#         f"Model confidence (mean prob): {measurements.get('confidence_mean', 0):.3f}",
#     ]
#     ty = box_y+70
#     for ln in lines:
#         c.drawString(margin_x, ty, ln)
#         ty -= 14

#     # Page 2: Detailed findings
#     c.showPage()
#     c.setFont("Helvetica-Bold", 14)
#     c.drawString(margin_x, h-60, "Detailed Findings")
#     text = c.beginText(margin_x, h-90)
#     text.setFont("Helvetica", 11)
#     text.textLines(findings.get('detailed','-'))
#     c.drawText(text)

#     # Page 3+: Images
#     c.showPage()
#     c.setFont("Helvetica-Bold", 14)
#     c.drawString(margin_x, h-60, "Key Images and Attention Maps")
#     y = h-100
#     for title, img in images_dict.items():
#         buf = npimg_to_buf(img, cmap='jet' if img.ndim==2 else None)
#         width_img = 360
#         height_img = 360 * img.shape[0]/img.shape[1] if img.ndim==3 else 360
#         # cap height
#         if height_img > 300: height_img = 300
#         c.drawImage(ImageReader(buf), margin_x, y - height_img, width=width_img, height=height_img)
#         c.drawString(margin_x + width_img + 20, y - 20, title)
#         y -= (height_img + 30)
#         if y < 120:
#             c.showPage()
#             y = h-100

#     # Final page: technical appendix
#     c.showPage()
#     c.setFont("Helvetica-Bold", 12)
#     c.drawString(margin_x, h-60, "Technical Appendix")
#     t = c.beginText(margin_x, h-90)
#     t.setFont("Helvetica", 10)
#     t.textLines([
#         f"Model: {technical.get('model_name')}",
#         f"Model version: {technical.get('model_version')}",
#         f"Validation Dice: {technical.get('val_dice')}",
#         f"Weights SHA256: {technical.get('weights_sha256')}",
#         f"Inference time (avg): {technical.get('inference_ms')} ms / slice",
#         f"Preprocessing: {technical.get('preproc')}",
#         "",
#         "Disclaimer: This report is AI-assisted. Final clinical interpretation must be performed by a qualified radiologist or neurosurgeon."
#     ])
#     c.drawText(t)

#     # footer on all pages
#     # (ReportLab makes repeated footer easier using templates—omitted for brevity)
#     c.save()
# src/report.py
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from reportlab.lib import colors
import matplotlib.pyplot as plt
import io
import datetime
import numpy as np

def npimg_to_buf(img_np, cmap=None):
    buf = io.BytesIO()
    if img_np.ndim == 2:
        plt.imsave(buf, img_np, cmap=cmap or 'gray', format='png')
    else:
        plt.imsave(buf, img_np, format='png')
    buf.seek(0)
    return buf

def draw_wrapped_text(c, x, y, text, max_width, leading=12, font="Helvetica", font_size=10):
    """
    Draw text wrapped to max_width starting at (x,y) downward.
    Returns new y after drawing.
    """
    c.setFont(font, font_size)
    words = text.split()
    line = ""
    cur_y = y
    for w in words:
        test = (line + " " + w).strip()
        if c.stringWidth(test, font, font_size) <= max_width:
            line = test
        else:
            c.drawString(x, cur_y, line)
            cur_y -= leading
            line = w
    if line:
        c.drawString(x, cur_y, line)
        cur_y -= leading
    return cur_y

def compose_report_full(output_pdf_path, metadata, findings, measurements, images_dict, technical):
    """
    Creates a multi-page PDF with a prominent Clinical Summary box on page 1.

    Parameters
    - output_pdf_path: output file path
    - metadata: dict ('case_id', 'patient_info' optional)
    - findings: dict with 'summary' (short), 'detailed' (long)
    - measurements: dict with numeric metrics (tumor_present, slice_peak, volume_cm3, centroid_voxel, bbox, confidence_mean, etc.)
    - images_dict: {title: numpy_array_rgb_or_gray}
    - technical: dict (model_name, model_version, val_dice, weights_sha256, inference_ms, preproc)
    """
    c = canvas.Canvas(output_pdf_path, pagesize=A4)
    W, H = A4
    margin = 40

    # Header
    c.setFont("Helvetica-Bold", 18)
    c.drawString(margin, H - 48, "AI-Assisted MRI Brain Tumor Report")
    c.setFont("Helvetica", 9)
    c.drawRightString(W - margin, H - 48, f"Generated: {datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")
    c.line(margin, H - 56, W - margin, H - 56)

    # Sub-header metadata
    c.setFont("Helvetica-Bold", 11)
    c.drawString(margin, H - 76, f"Case: {metadata.get('case_id','-')}")
    c.setFont("Helvetica", 10)
    patient_info = metadata.get('patient_info', '')
    if patient_info:
        draw_wrapped_text(c, margin, H - 92, patient_info, max_width=W - 2*margin, leading=12)

    # ---- Clinical Summary box (prominent) ----
    box_x = margin
    box_y = H - 220
    box_w = W - 2*margin
    box_h = 120

    # Draw light background rectangle
    c.setFillColor(colors.HexColor("#F2F7FB"))   # pale blue
    c.roundRect(box_x, box_y, box_w, box_h, 6, fill=1, stroke=0)
    c.setFillColor(colors.black)

    # Box title
    c.setFont("Helvetica-Bold", 12)
    c.drawString(box_x + 12, box_y + box_h - 20, "Clinical Summary (AI-assisted)")

    # Box content — wrapped
    c.setFont("Helvetica", 10)
    text_x = box_x + 12
    text_y = box_y + box_h - 38
    max_w = box_w - 24
    # Draw summary (short) bold
    c.setFont("Helvetica-Bold", 10)
    summary = findings.get('summary', 'No summary available.')
    # Draw first line(s) of summary in bold
    # We will draw paragraph as wrapped lines; use draw_wrapped_text
    cur_y = text_y
    cur_y = draw_wrapped_text(c, text_x, cur_y, summary, max_w, leading=13, font="Helvetica-Bold", font_size=10)
    # leave small gap
    cur_y -= 4
    # additional short line for confidence if present in measurements
    conf_text = ""
    if measurements.get('confidence_mean') is not None:
        conf_text = f"Model confidence (mean prob): {measurements.get('confidence_mean'):.3f}"
        c.setFont("Helvetica", 9)
        cur_y = draw_wrapped_text(c, text_x, cur_y, conf_text, max_w, leading=12, font="Helvetica", font_size=9)
    # Draw small footer inside box: tumor present / volume
    footer_parts = []
    footer_parts.append(f"Tumor present: {'Yes' if measurements.get('tumor_present') else 'No'}")
    if measurements.get('volume_cm3') is not None:
        footer_parts.append(f"Estimated volume: {measurements.get('volume_cm3'):.2f} cm³")
    if measurements.get('slice_peak') is not None:
        footer_parts.append(f"Representative slice: {measurements.get('slice_peak')}")
    footer_text = "  |  ".join(footer_parts)
    c.setFont("Helvetica-Oblique", 9)
    c.drawString(text_x, box_y + 10, footer_text)

    # Move to measurements area to the right of box or below if small page
    meas_x = margin
    meas_y = box_y - 10

    # Measurements header
    c.setFont("Helvetica-Bold", 12)
    c.drawString(meas_x, meas_y, "Measurements")
    c.setFont("Helvetica", 10)
    meas_y -= 16

    # Draw measurement key-values (left column)
    kv_lines = []
    kv_lines.append(("Tumor detected", "Yes" if measurements.get('tumor_present') else "No"))
    if measurements.get('slice_peak') is not None:
        kv_lines.append(("Representative slice", str(measurements.get('slice_peak'))))
    if measurements.get('max_area_px') is not None:
        kv_lines.append(("Max area (pixels)", f"{measurements.get('max_area_px'):,} px"))
    if measurements.get('max_area_mm2') is not None:
        kv_lines.append(("Max area", f"{measurements.get('max_area_mm2'):.1f} mm²"))
    if measurements.get('volume_cm3') is not None:
        kv_lines.append(("Estimated volume", f"{measurements.get('volume_cm3'):.2f} cm³"))
    if measurements.get('centroid_voxel') is not None:
        cvt = measurements.get('centroid_voxel')
        kv_lines.append(("Centroid (voxel)", f"x={cvt[0]:.1f}, y={cvt[1]:.1f}, z={int(cvt[2])}"))
    if measurements.get('bbox') is not None:
        rmin,rmax,cmin,cmax = measurements.get('bbox')
        kv_lines.append(("Bounding box (r,c)", f"row {rmin}–{rmax}, col {cmin}–{cmax}"))
    if measurements.get('confidence_mean') is not None:
        kv_lines.append(("Mean probability", f"{measurements.get('confidence_mean'):.3f}"))

    # render key-value pairs in two columns if many
    col1_x = meas_x
    col2_x = meas_x + 260
    cur_y = meas_y
    half = (len(kv_lines) + 1) // 2
    for i, (k,v) in enumerate(kv_lines):
        x_pos = col1_x if i < half else col2_x
        if i == half:
            cur_y = meas_y
        c.setFont("Helvetica-Bold", 10)
        c.drawString(x_pos, cur_y, f"{k}:")
        c.setFont("Helvetica", 10)
        c.drawString(x_pos + 130, cur_y, f"{v}")
        cur_y -= 14

    # Next page: Detailed findings
    c.showPage()
    # Re-draw header on new page
    c.setFont("Helvetica-Bold", 16)
    c.drawString(margin, H - 48, "Detailed Findings")
    c.setFont("Helvetica", 11)
    y = H - 76
    long_text = findings.get('detailed', '')
    # Use wrapped text drawing
    y = draw_wrapped_text(c, margin, y, long_text, max_width=W - 2*margin, leading=13, font="Helvetica", font_size=11)

    # Page for images
    c.showPage()
    c.setFont("Helvetica-Bold", 14)
    c.drawString(margin, H - 48, "Key Images & Attention Maps")
    y = H - 88
    # iterate images and place them with captions
    img_w = 360
    img_h_max = 260
    for title, img in images_dict.items():
        buf = npimg_to_buf(img, cmap='jet' if img.ndim==2 else None)
        # compute aspect-ratio fit
        try:
            arr = img
            h_img, w_img = arr.shape[0], arr.shape[1]
        except Exception:
            h_img, w_img = img_h_max, img_w
        aspect = h_img / float(w_img)
        h_display = int(min(img_h_max, img_w * aspect))
        # draw image
        c.drawImage(ImageReader(buf), margin, y - h_display, width=img_w, height=h_display)
        # draw caption to right
        c.setFont("Helvetica", 11)
        c.drawString(margin + img_w + 12, y - 12, title)
        y -= (h_display + 30)
        if y < 120:
            c.showPage()
            c.setFont("Helvetica-Bold", 14)
            c.drawString(margin, H - 48, "Key Images & Attention Maps (cont.)")
            y = H - 88

    # Final page: Technical appendix
    c.showPage()
    c.setFont("Helvetica-Bold", 14)
    c.drawString(margin, H - 48, "Technical Appendix")
    t_y = H - 80
    c.setFont("Helvetica", 10)
    tech_lines = [
        f"Model: {technical.get('model_name', 'N/A')}",
        f"Model version: {technical.get('model_version', 'N/A')}",
        f"Validation Dice: {technical.get('val_dice', 'N/A')}",
        f"Weights SHA256: {technical.get('weights_sha256', 'N/A')}",
        f"Inference time (avg): {technical.get('inference_ms', 'N/A')} ms / slice",
        f"Preprocessing: {technical.get('preproc', 'N/A')}",
        "",
        "Disclaimer: This report is AI-assisted and intended for research/demo purposes only. Final clinical interpretation and management decisions must be made by a qualified radiologist or neurosurgeon."
    ]
    for ln in tech_lines:
        c.drawString(margin, t_y, ln)
        t_y -= 14

    c.save()
