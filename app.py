
import io
import math
import json
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from scipy import ndimage as ndi
from skimage import exposure, filters, measure, morphology, segmentation, feature, color
from skimage.util import img_as_ubyte

st.set_page_config(
    page_title="Advanced Metallurgical Microstructure Analyzer",
    page_icon="🔬",
    layout="wide",
)

# -----------------------------
# Utility functions
# -----------------------------
def rgb_from_pil(img):
    return np.array(img.convert("RGB"))

def gray_image(rgb):
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

def normalize_gray(gray):
    p2, p98 = np.percentile(gray, (2, 98))
    if p98 <= p2:
        return gray.copy()
    return exposure.rescale_intensity(gray, in_range=(p2, p98), out_range=(0, 255)).astype(np.uint8)

def clahe(gray):
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)

def overlay_mask(rgb, mask, alpha=0.45, color_rgb=(255, 0, 0)):
    out = rgb.copy().astype(np.float32)
    c = np.array(color_rgb, dtype=np.float32)
    out[mask] = (1 - alpha) * out[mask] + alpha * c
    return np.clip(out, 0, 255).astype(np.uint8)

def draw_labels(rgb, labels, min_area=5, color_rgb=(255, 0, 0)):
    out = rgb.copy()
    props = measure.regionprops(labels)
    for p in props:
        if p.area < min_area:
            continue
        y, x = p.centroid
        cv2.circle(out, (int(x), int(y)), 3, color_rgb, -1)
    return out

def scale_text(px, um_per_px):
    if um_per_px is None:
        return f"{px:.2f} px"
    return f"{px * um_per_px:.3f} µm"

def area_text(px2, um_per_px):
    if um_per_px is None:
        return f"{px2:.2f} px²"
    return f"{px2 * (um_per_px ** 2):.3f} µm²"

def get_scale():
    if st.session_state.get("use_scale", False):
        return float(st.session_state.get("um_per_px", 1.0))
    return None

def region_dataframe(labels, um_per_px=None, feature_name="Feature"):
    rows = []
    for i, p in enumerate(measure.regionprops(labels), start=1):
        if p.area <= 0:
            continue
        eq_px = p.equivalent_diameter_area
        feret_px = getattr(p, "feret_diameter_max", np.nan)
        rows.append({
            "ID": i,
            "Feature": feature_name,
            "Area_px2": float(p.area),
            "Area_um2": float(p.area * um_per_px**2) if um_per_px else np.nan,
            "Equivalent_Diameter_px": float(eq_px),
            "Equivalent_Diameter_um": float(eq_px * um_per_px) if um_per_px else np.nan,
            "Feret_Diameter_px": float(feret_px) if np.isfinite(feret_px) else np.nan,
            "Feret_Diameter_um": float(feret_px * um_per_px) if um_per_px and np.isfinite(feret_px) else np.nan,
            "Perimeter_px": float(p.perimeter),
            "Aspect_Ratio": float((p.major_axis_length / max(p.minor_axis_length, 1e-9))),
            "Circularity": float((4 * math.pi * p.area) / max(p.perimeter ** 2, 1e-9)),
            "Centroid_X_px": float(p.centroid[1]),
            "Centroid_Y_px": float(p.centroid[0]),
        })
    return pd.DataFrame(rows)

def clean_binary(mask, min_area=20, max_hole=30):
    mask = morphology.remove_small_objects(mask.astype(bool), min_size=min_area)
    mask = morphology.remove_small_holes(mask, area_threshold=max_hole)
    return mask

def connected_labels(mask):
    return measure.label(mask, connectivity=2)

def watershed_particles(mask, min_distance=8):
    distance = ndi.distance_transform_edt(mask)
    coords = feature.peak_local_max(
        distance,
        min_distance=min_distance,
        labels=mask,
        exclude_border=False
    )
    peak_mask = np.zeros_like(mask, dtype=bool)
    if len(coords):
        peak_mask[tuple(coords.T)] = True
    markers = ndi.label(peak_mask)[0]
    labels = segmentation.watershed(-distance, markers, mask=mask)
    return labels

def detect_pores(gray, threshold_method, threshold_value, min_area, max_area, dark=True):
    g = normalize_gray(gray)
    if threshold_method == "Otsu":
        t = filters.threshold_otsu(g)
    elif threshold_method == "Adaptive":
        t = filters.threshold_local(g, block_size=51, offset=0)
        mask = g < t if dark else g > t
        return filter_labels(mask, min_area, max_area)
    else:
        t = threshold_value
    mask = g < t if dark else g > t
    mask = morphology.remove_small_objects(mask, min_size=min_area)
    mask = morphology.remove_small_holes(mask, area_threshold=max(10, min_area))
    return filter_labels(mask, min_area, max_area)

def filter_labels(mask, min_area, max_area):
    lab = measure.label(mask, connectivity=2)
    keep = np.zeros_like(mask, dtype=bool)
    for p in measure.regionprops(lab):
        if min_area <= p.area <= max_area:
            keep[lab == p.label] = True
    return measure.label(keep, connectivity=2)

def porosity_analysis(gray, min_area, max_area, threshold_method, threshold_value):
    labels = detect_pores(gray, threshold_method, threshold_value, min_area, max_area, dark=True)
    mask = labels > 0
    df = region_dataframe(labels, get_scale(), "Pore")
    total_area = gray.shape[0] * gray.shape[1]
    frac = 100.0 * mask.sum() / max(total_area, 1)
    if len(df):
        stats = {
            "Pore count": int(len(df)),
            "Porosity area fraction (%)": frac,
            "Total pore area": area_text(mask.sum(), get_scale()),
            "Mean pore area": area_text(df["Area_px2"].mean(), get_scale()),
            "Mean equivalent pore diameter": scale_text(df["Equivalent_Diameter_px"].mean(), get_scale()),
            "Maximum pore diameter": scale_text(df["Equivalent_Diameter_px"].max(), get_scale()),
            "Mean circularity": float(df["Circularity"].mean()),
        }
    else:
        stats = {
            "Pore count": 0,
            "Porosity area fraction (%)": 0.0,
            "Total pore area": area_text(0, get_scale()),
            "Mean pore area": area_text(0, get_scale()),
            "Mean equivalent pore diameter": "0",
            "Maximum pore diameter": "0",
            "Mean circularity": 0.0,
        }
    return labels, df, stats

def crack_analysis(gray, min_area, min_aspect, threshold):
    # Dark elongated objects are a useful screening heuristic, not a certified crack classifier.
    g = normalize_gray(gray)
    mask = g < threshold
    mask = morphology.remove_small_objects(mask, min_size=min_area)
    labels0 = measure.label(mask, connectivity=2)
    keep = np.zeros_like(mask, dtype=bool)
    for p in measure.regionprops(labels0):
        aspect = p.major_axis_length / max(p.minor_axis_length, 1e-6)
        if p.area >= min_area and aspect >= min_aspect:
            keep[labels0 == p.label] = True
    labels = measure.label(keep, connectivity=2)
    df = region_dataframe(labels, get_scale(), "Crack")
    total = gray.shape[0] * gray.shape[1]
    length_density = float(df["Feret_Diameter_px"].sum() / math.sqrt(total)) if len(df) else 0.0
    stats = {
        "Crack candidates": int(len(df)),
        "Crack area fraction (%)": 100 * keep.sum() / max(total, 1),
        "Total crack area": area_text(keep.sum(), get_scale()),
        "Mean aspect ratio": float(df["Aspect_Ratio"].mean()) if len(df) else 0.0,
        "Normalized crack length": length_density,
    }
    return labels, df, stats

def inclusion_analysis(gray, min_area, max_area, threshold):
    # Bright-particle screening, suitable for inclusions/second phases depending on imaging contrast.
    g = normalize_gray(gray)
    mask = g > threshold
    mask = morphology.remove_small_objects(mask, min_size=min_area)
    labels = filter_labels(mask, min_area, max_area)
    df = region_dataframe(labels, get_scale(), "Inclusion/bright feature")
    total = gray.shape[0] * gray.shape[1]
    stats = {
        "Bright feature count": int(len(df)),
        "Bright feature area fraction (%)": 100 * (labels > 0).sum() / max(total, 1),
        "Mean feature area": area_text(df["Area_px2"].mean(), get_scale()) if len(df) else "0",
        "Maximum feature diameter": scale_text(df["Equivalent_Diameter_px"].max(), get_scale()) if len(df) else "0",
    }
    return labels, df, stats

def particle_analysis(gray, threshold, min_area, max_area, dark=True):
    labels = detect_pores(gray, "Manual", threshold, min_area, max_area, dark=dark)
    df = region_dataframe(labels, get_scale(), "Particle")
    return labels, df

def phase_segmentation(gray, n_classes=2):
    # Simple intensity-based phase screening.
    g = normalize_gray(gray)
    if n_classes == 2:
        t = filters.threshold_otsu(g)
        a = g < t
        b = ~a
        return a, b, t
    q1, q2 = np.percentile(g, [33, 66])
    a = g <= q1
    b = (g > q1) & (g <= q2)
    c = g > q2
    return a, b, c

def grain_segmentation(gray, min_distance=10, min_area=100):
    # Grain-boundary/contrast screening. Requires suitable etched micrograph.
    g = normalize_gray(clahe(gray))
    grad = filters.sobel(g)
    threshold = filters.threshold_otsu(grad)
    boundary = grad > threshold
    boundary = morphology.binary_dilation(boundary, morphology.disk(1))
    interior = ~boundary
    interior = morphology.remove_small_objects(interior, min_size=min_area)
    distance = ndi.distance_transform_edt(interior)
    coords = feature.peak_local_max(
        distance, min_distance=min_distance, labels=interior, exclude_border=False
    )
    peaks = np.zeros_like(interior, dtype=bool)
    if len(coords):
        peaks[tuple(coords.T)] = True
    markers = ndi.label(peaks)[0]
    labels = segmentation.watershed(-distance, markers, mask=interior)
    labels = morphology.remove_small_objects(labels, min_size=min_area)
    return labels, boundary

def grain_stats(labels):
    props = [p for p in measure.regionprops(labels) if p.area > 0]
    if not props:
        return {}, pd.DataFrame()
    df = region_dataframe(labels, get_scale(), "Grain")
    stats = {
        "Grain count": len(df),
        "Mean equivalent grain diameter": scale_text(df["Equivalent_Diameter_px"].mean(), get_scale()),
        "Median equivalent grain diameter": scale_text(df["Equivalent_Diameter_px"].median(), get_scale()),
        "Maximum grain diameter": scale_text(df["Equivalent_Diameter_px"].max(), get_scale()),
        "Mean aspect ratio": float(df["Aspect_Ratio"].mean()),
        "Mean circularity": float(df["Circularity"].mean()),
    }
    return stats, df

def hist_plot_data(values, bins=30):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return pd.DataFrame({"bin": [], "count": []})
    counts, edges = np.histogram(values, bins=min(bins, max(5, len(values)//2)))
    centers = (edges[:-1] + edges[1:]) / 2
    return pd.DataFrame({"bin": centers, "count": counts})

def df_to_csv_bytes(df):
    return df.to_csv(index=False).encode("utf-8")

def make_excel(summary_dict, dfs):
    from openpyxl import Workbook
    from openpyxl.utils.dataframe import dataframe_to_rows
    wb = Workbook()
    ws = wb.active
    ws.title = "Summary"
    ws.append(["Parameter", "Value"])
    for k, v in summary_dict.items():
        ws.append([k, v])
    for name, df in dfs.items():
        if df is None or df.empty:
            continue
        ws2 = wb.create_sheet(name[:31])
        for row in dataframe_to_rows(df, index=False, header=True):
            ws2.append(row)
    bio = io.BytesIO()
    wb.save(bio)
    return bio.getvalue()

# -----------------------------
# Session state
# -----------------------------
if "results" not in st.session_state:
    st.session_state.results = {}

# -----------------------------
# Header
# -----------------------------
st.title("🔬 Advanced Metallurgical Microstructure Analyzer")

# Developer / Student Information
st.markdown("""
### 👨‍🔬 Developed By
**Name:** Subham Sahu  
**Roll No.:** 426060  
**Registration No.:** 2301105744  
**Branch:** Metallurgical & Materials Engineering  
**College:** Indira Gandhi Institute of Technology (IGIT), Sarang
""")

st.caption(
    "Optical Microscopy (OM) + SEM + EDS • Quantitative Metallography • "
    "Porosity • Crack • Inclusion • Grain & Particle Analysis • Research/AI Ready"
)

# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.header("⚙️ Analysis Control")
    level = st.selectbox(
        "Analysis Level",
        ["Level 1 — Basic", "Level 2 — Intermediate", "Level 3 — Advanced", "Level 4 — Research / AI"]
    )
    microscopy = st.radio("Microscopy Technique", ["Optical Microscopy (OM)", "SEM"])

    st.subheader("📋 Sample Information")
    sample_id = st.text_input("Sample ID", "STEEL-SEM-001")
    material = st.text_input("Material", "Carbon Steel")
    condition = st.text_input("Heat treatment / condition", "Polished and etched")
    magnification = st.text_input("Magnification", "2000X")

    st.subheader("📏 Scale Calibration")
    use_scale = st.checkbox("Use calibrated scale", value=False, key="use_scale")
    if use_scale:
        um_per_px = st.number_input("Micrometres per pixel (µm/px)", min_value=0.000001, value=0.10, format="%.6f", key="um_per_px")
        st.info("Use the actual image scale bar. Do NOT infer µm/px from magnification alone.")
    else:
        st.warning("No scale calibration: size results will be reported in pixels.")

    st.subheader("📂 Image")
    uploaded = st.file_uploader(
        "Upload OM / SEM image",
        type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"]
    )

    st.subheader("Optional EDS")
    eds_file = st.file_uploader("Upload EDS CSV/TXT (optional)", type=["csv", "txt"])

    st.subheader("🔧 Detection Parameters")
    min_area = st.number_input("Minimum feature area (px²)", min_value=1, value=20, step=1)
    max_area = st.number_input("Maximum feature area (px²)", min_value=10, value=100000, step=100)
    dark_threshold = st.slider("Dark-feature threshold", 0, 255, 80)
    bright_threshold = st.slider("Bright-feature threshold", 0, 255, 190)
    crack_aspect = st.slider("Crack minimum aspect ratio", 2.0, 30.0, 5.0, 0.5)
    grain_distance = st.slider("Grain minimum separation (px)", 3, 100, 15)

# -----------------------------
# Main tabs
# -----------------------------
tabs = st.tabs(["🏠 Overview", "🖼️ Basic", "📊 Quantitative", "🧩 Segmentation", "📈 Statistics", "🧪 Defects", "⚗️ EDS", "📄 Report"])

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    rgb = rgb_from_pil(image)
    gray = gray_image(rgb)

    with tabs[0]:
        c1, c2 = st.columns(2)
        with c1:
            st.image(rgb, caption=f"{uploaded.name} • {rgb.shape[1]}×{rgb.shape[0]} px", use_container_width=True)
        with c2:
            st.markdown("### Sample metadata")
            st.write({
                "Sample ID": sample_id,
                "Material": material,
                "Condition": condition,
                "Microscopy": microscopy,
                "Magnification": magnification,
                "Scale": f"{get_scale():.6f} µm/px" if get_scale() else "Not calibrated",
            })
            st.info(
                "Quantitative metallography is only physically meaningful when the image has a "
                "known scale/calibration. Magnification alone is not a pixel-size calibration."
            )

    with tabs[1]:
        st.subheader("Basic Image Processing")
        mode = st.selectbox("Processing", ["Original", "Grayscale", "Contrast enhanced", "CLAHE", "Gaussian denoise", "Adaptive threshold"])
        if mode == "Original":
            processed = rgb
        elif mode == "Grayscale":
            processed = gray
        elif mode == "Contrast enhanced":
            processed = normalize_gray(gray)
        elif mode == "CLAHE":
            processed = clahe(gray)
        elif mode == "Gaussian denoise":
            processed = cv2.GaussianBlur(gray, (5, 5), 0)
        else:
            processed = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 51, 3)
        st.image(processed, caption=mode, use_container_width=True)

    with tabs[2]:
        st.subheader("Quantitative Microstructure")
        if st.button("▶ Run Quantitative Analysis", type="primary"):
            labels, boundary = grain_segmentation(gray, grain_distance, int(min_area))
            gst, gdf = grain_stats(labels)
            a, b = phase_segmentation(gray, 2)[:2]
            p1 = 100 * a.mean()
            p2 = 100 * b.mean()
            p_overlay = overlay_mask(rgb, a, 0.35, (255, 0, 0))
            p_overlay = overlay_mask(p_overlay, b, 0.20, (0, 255, 0))
            st.session_state.results["grain_labels"] = labels
            st.session_state.results["grain_df"] = gdf
            st.session_state.results["grain_stats"] = gst
            st.session_state.results["phase1_pct"] = p1
            st.session_state.results["phase2_pct"] = p2
            st.session_state.results["phase_overlay"] = p_overlay
            st.success("Quantitative analysis completed.")

        if "grain_stats" in st.session_state.results:
            gst = st.session_state.results["grain_stats"]
            cols = st.columns(4)
            cols[0].metric("Grain count", gst.get("Grain count", 0))
            cols[1].metric("Mean grain diameter", gst.get("Mean equivalent grain diameter", "-"))
            cols[2].metric("Median diameter", gst.get("Median equivalent grain diameter", "-"))
            cols[3].metric("Max diameter", gst.get("Maximum grain diameter", "-"))
            st.write("**Intensity-based phase screening:**")
            c1, c2 = st.columns(2)
            c1.metric("Dark phase area", f"{st.session_state.results['phase1_pct']:.2f}%")
            c2.metric("Light phase area", f"{st.session_state.results['phase2_pct']:.2f}%")
            st.image(st.session_state.results["phase_overlay"], caption="Phase screening overlay (red/green)", use_container_width=True)
            if not st.session_state.results["grain_df"].empty:
                st.dataframe(st.session_state.results["grain_df"].head(1000), use_container_width=True)
        else:
            st.info("Click Run Quantitative Analysis.")

    with tabs[3]:
        st.subheader("Segmentation")
        seg_mode = st.selectbox("Segmentation type", ["Grain/Wateshed", "Dark features", "Bright features", "Two intensity phases"])
        if st.button("Run Segmentation"):
            if seg_mode == "Grain/Wateshed":
                labels, boundary = grain_segmentation(gray, grain_distance, int(min_area))
                overlay = color.label2rgb(labels, image=rgb, bg_label=0, alpha=0.35)
                overlay = img_as_ubyte(overlay)
                st.session_state.results["segmentation_df"] = region_dataframe(labels, get_scale(), "Grain")
            elif seg_mode == "Dark features":
                labels = detect_pores(gray, "Manual", dark_threshold, int(min_area), int(max_area), True)
                overlay = draw_labels(rgb, labels, min_area)
                st.session_state.results["segmentation_df"] = region_dataframe(labels, get_scale(), "Dark feature")
            elif seg_mode == "Bright features":
                labels = detect_pores(gray, "Manual", bright_threshold, int(min_area), int(max_area), False)
                overlay = draw_labels(rgb, labels, min_area, (0, 255, 0))
                st.session_state.results["segmentation_df"] = region_dataframe(labels, get_scale(), "Bright feature")
            else:
                a, b = phase_segmentation(gray, 2)[:2]
                overlay = overlay_mask(rgb, a, 0.4, (255, 0, 0))
                overlay = overlay_mask(overlay, b, 0.25, (0, 255, 0))
                st.session_state.results["segmentation_df"] = pd.DataFrame({
                    "Phase": ["Dark", "Light"],
                    "Area_fraction_%": [100*a.mean(), 100*b.mean()]
                })
            st.image(overlay, caption=seg_mode, use_container_width=True)
            if not st.session_state.results["segmentation_df"].empty:
                st.dataframe(st.session_state.results["segmentation_df"].head(1000), use_container_width=True)

    with tabs[4]:
        st.subheader("Statistics")
        df = st.session_state.results.get("grain_df")
        if df is not None and not df.empty:
            st.write("### Grain/feature distribution")
            metric_col = "Equivalent_Diameter_um" if get_scale() else "Equivalent_Diameter_px"
            hist = hist_plot_data(df[metric_col])
            if not hist.empty:
                st.bar_chart(hist.set_index("bin"))
            st.write(df.describe(include="all").T)
        else:
            st.info("Run Quantitative Analysis first.")

    with tabs[5]:
        st.subheader("🧪 Defect & Porosity Analysis")
        st.warning(
            "These are image-based screening algorithms. They are not a substitute for expert metallographic "
            "interpretation, ASTM certification, or a trained defect-detection model."
        )
        defect = st.selectbox("Analysis", ["Porosity", "Crack screening", "Inclusion / bright-feature screening"])
        if defect == "Porosity":
            th_method = st.selectbox("Porosity threshold", ["Otsu", "Manual", "Adaptive"])
            if st.button("Detect Porosity", type="primary"):
                labels, pdf, stats = porosity_analysis(
                    gray, int(min_area), int(max_area), th_method, int(dark_threshold)
                )
                overlay = draw_labels(rgb, labels, min_area, (255, 0, 0))
                st.session_state.results["pore_df"] = pdf
                st.session_state.results["pore_stats"] = stats
                st.session_state.results["pore_overlay"] = overlay
            if "pore_stats" in st.session_state.results:
                s = st.session_state.results["pore_stats"]
                cols = st.columns(4)
                cols[0].metric("Pore count", s["Pore count"])
                cols[1].metric("Porosity", f"{s['Porosity area fraction (%)']:.3f}%")
                cols[2].metric("Mean pore diameter", s["Mean equivalent pore diameter"])
                cols[3].metric("Max pore diameter", s["Maximum pore diameter"])
                st.image(st.session_state.results["pore_overlay"], caption="Detected pores", use_container_width=True)
                st.dataframe(st.session_state.results["pore_df"].head(1000), use_container_width=True)

        elif defect == "Crack screening":
            if st.button("Detect Crack Candidates", type="primary"):
                labels, cdf, stats = crack_analysis(gray, int(min_area), float(crack_aspect), int(dark_threshold))
                overlay = draw_labels(rgb, labels, min_area, (255, 0, 255))
                st.session_state.results["crack_df"] = cdf
                st.session_state.results["crack_stats"] = stats
                st.session_state.results["crack_overlay"] = overlay
            if "crack_stats" in st.session_state.results:
                s = st.session_state.results["crack_stats"]
                cols = st.columns(4)
                cols[0].metric("Crack candidates", s["Crack candidates"])
                cols[1].metric("Area fraction", f"{s['Crack area fraction (%)']:.3f}%")
                cols[2].metric("Mean aspect ratio", f"{s['Mean aspect ratio']:.2f}")
                cols[3].metric("Normalized length", f"{s['Normalized crack length']:.3f}")
                st.image(st.session_state.results["crack_overlay"], caption="Crack candidate screening", use_container_width=True)
                st.dataframe(st.session_state.results["crack_df"].head(1000), use_container_width=True)

        else:
            if st.button("Detect Bright Inclusions / Features", type="primary"):
                labels, idf, stats = inclusion_analysis(gray, int(min_area), int(max_area), int(bright_threshold))
                overlay = draw_labels(rgb, labels, min_area, (0, 255, 0))
                st.session_state.results["inclusion_df"] = idf
                st.session_state.results["inclusion_stats"] = stats
                st.session_state.results["inclusion_overlay"] = overlay
            if "inclusion_stats" in st.session_state.results:
                s = st.session_state.results["inclusion_stats"]
                cols = st.columns(4)
                cols[0].metric("Bright features", s["Bright feature count"])
                cols[1].metric("Area fraction", f"{s['Bright feature area fraction (%)']:.3f}%")
                cols[2].metric("Mean area", s["Mean feature area"])
                cols[3].metric("Max diameter", s["Maximum feature diameter"])
                st.image(st.session_state.results["inclusion_overlay"], caption="Bright feature screening", use_container_width=True)
                st.dataframe(st.session_state.results["inclusion_df"].head(1000), use_container_width=True)

    with tabs[6]:
        st.subheader("⚗️ SEM-EDS Support")
        if eds_file is None:
            st.info("Upload a CSV/TXT EDS export from the sidebar.")
        else:
            try:
                if eds_file.name.lower().endswith(".csv"):
                    eds = pd.read_csv(eds_file)
                else:
                    eds = pd.read_csv(eds_file, sep=None, engine="python")
                st.dataframe(eds.head(1000), use_container_width=True)
                numeric = eds.select_dtypes(include=np.number)
                if not numeric.empty:
                    st.bar_chart(numeric)
                st.success("EDS data loaded. Column interpretation depends on your EDS software export format.")
            except Exception as e:
                st.error(f"Could not read EDS file: {e}")

    with tabs[7]:
        st.subheader("📄 Research Report & Export")
        summary = {
            "Generated": datetime.now().isoformat(timespec="seconds"),
            "Sample ID": sample_id,
            "Material": material,
            "Condition": condition,
            "Microscopy": microscopy,
            "Magnification": magnification,
            "Calibration": f"{get_scale()} µm/px" if get_scale() else "Not calibrated",
        }
        if "grain_stats" in st.session_state.results:
            summary.update(st.session_state.results["grain_stats"])
        if "pore_stats" in st.session_state.results:
            summary.update(st.session_state.results["pore_stats"])
        if "crack_stats" in st.session_state.results:
            summary.update(st.session_state.results["crack_stats"])
        if "inclusion_stats" in st.session_state.results:
            summary.update(st.session_state.results["inclusion_stats"])

        st.json(summary)

        dfs = {
            "Grains": st.session_state.results.get("grain_df"),
            "Pores": st.session_state.results.get("pore_df"),
            "Cracks": st.session_state.results.get("crack_df"),
            "Inclusions": st.session_state.results.get("inclusion_df"),
        }
        excel = make_excel(summary, dfs)
        st.download_button(
            "⬇️ Download Excel Research Dataset",
            data=excel,
            file_name=f"{sample_id}_microstructure_analysis.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
        for name, df in dfs.items():
            if df is not None and not df.empty:
                st.download_button(
                    f"⬇️ Download {name} CSV",
                    data=df_to_csv_bytes(df),
                    file_name=f"{sample_id}_{name.lower()}.csv",
                    mime="text/csv"
                )

else:
    st.info("Upload an OM or SEM micrograph from the sidebar to start.")
    st.markdown("""
### Included modules
- Image preprocessing: grayscale, contrast enhancement, CLAHE, denoising, adaptive threshold
- Quantitative analysis: grain/feature count, equivalent diameter, Feret diameter, aspect ratio, circularity
- Phase screening: intensity-based area fraction
- Grain segmentation: Sobel + watershed workflow
- **Porosity:** pore count, porosity area fraction, pore area and diameter, marked image
- **Crack screening:** elongated dark-feature detection
- **Inclusion / bright-feature screening**
- Statistics and distributions
- SEM-EDS CSV/TXT import
- Excel + CSV export
- Research report summary
- OM and SEM workflows
- Optional µm/px calibration

**Scientific note:** exact phase identification and certified defect classification require validated imaging conditions,
scale bars, metallographic preparation, expert interpretation, and—where appropriate—trained ML/AI models.
""")
