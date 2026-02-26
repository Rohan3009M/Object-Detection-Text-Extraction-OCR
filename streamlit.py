import io

import pandas as pd
import requests
import streamlit as st
from PIL import Image, ImageDraw


st.set_page_config(page_title="Number Plate Detector", layout="wide")


# -------------------------
# Sidebar
# -------------------------
st.sidebar.header("API Settings")
API_BASE = st.sidebar.text_input("Detector API Base URL", "http://127.0.0.1:8000")
API_URL_SINGLE = f"{API_BASE.rstrip('/')}/v1/analyze"
API_URL_BATCH = f"{API_BASE.rstrip('/')}/v1/analyze/batch"

conf_thr = st.sidebar.slider("Confidence threshold", 0.0, 1.0, 0.50, 0.01)
iou_thr = st.sidebar.slider("IoU threshold", 0.0, 1.0, 0.50, 0.01)
params = {"conf_thr": conf_thr, "iou_thr": iou_thr}

st.sidebar.caption("Streamlit calls only the detector API. OCR is handled by the backend service chain.")


# -------------------------
# Helpers
# -------------------------
def draw_boxes(img_pil: Image.Image, detections):
    out = img_pil.copy()
    draw = ImageDraw.Draw(out)

    for det in detections or []:
        bbox = det.get("bbox")
        conf = det.get("conf")
        cls = det.get("cls", "numberplate")
        if not bbox or len(bbox) != 4:
            continue

        x1, y1, x2, y2 = bbox
        for t in range(3):
            draw.rectangle([x1 - t, y1 - t, x2 + t, y2 + t], outline="red")

        label = f"{cls} {conf:.2f}" if isinstance(conf, (int, float)) else cls
        text_w = max(110, len(label) * 7)
        draw.rectangle([x1, max(0, y1 - 22), x1 + text_w, y1], fill="red")
        draw.text((x1 + 4, max(0, y1 - 18)), label, fill="white")

    return out


def call_api_single(file_name: str, file_bytes: bytes, mime: str):
    files = {"file": (file_name, file_bytes, mime)}
    return requests.post(API_URL_SINGLE, params=params, files=files, timeout=120)


def call_api_batch(uploaded_files):
    files = [("files", (up.name, up.getvalue(), up.type)) for up in uploaded_files]
    return requests.post(API_URL_BATCH, params=params, files=files, timeout=300)


def flatten_metrics(metrics: dict):
    flat = {}
    for k, v in (metrics or {}).items():
        if isinstance(v, dict):
            for kk, vv in v.items():
                flat[f"{k}.{kk}"] = vv
        else:
            flat[k] = v
    return flat


# -------------------------
# UI
# -------------------------
st.title("Number Plate Detector (YOLO + OCR)")
mode = st.radio("Mode", ["Single Image", "Batch (Multiple Images)"], horizontal=True)


if mode == "Single Image":
    uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if not uploaded:
        st.info("Upload an image to start.")
        st.stop()

    img_bytes = uploaded.read()
    img_pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("Input")
        st.image(img_pil, use_container_width=True)

    if st.button("Analyze"):
        with st.spinner("Running detection + OCR..."):
            resp = call_api_single(uploaded.name, img_bytes, uploaded.type)

        if resp.status_code != 200:
            st.error(f"API Error {resp.status_code}: {resp.text}")
            st.stop()

        data = resp.json()
        dets = data.get("detections", [])
        metrics = data.get("metrics", {})

        with col2:
            st.subheader("Result")
            st.metric("Present", "YES" if data.get("present") else "NO")
            st.metric("Max Conf", data.get("confidence"))
            st.metric("Plate Text", data.get("plate_text") or "-")
            st.metric(
                "OCR Conf",
                data.get("ocr_confidence") if data.get("ocr_confidence") is not None else "-",
            )
            st.write("**Detections:**", len(dets))
            st.write("**Latency (ms):**", data.get("latency_ms"))
            st.write("**Captured (IST):**", data.get("captured_at_readable") or "-")

        st.subheader("Visual Debug (BBox Overlay)")
        st.image(draw_boxes(img_pil, dets), use_container_width=True)

        c1, c2 = st.columns([1, 1])
        with c1:
            with st.expander("Metrics"):
                st.json(metrics)
        with c2:
            with st.expander("Full Response JSON"):
                st.json(data)


else:
    uploads = st.file_uploader(
        "Upload multiple images",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
    )

    colA, colB = st.columns([1, 1])
    with colA:
        run = st.button("Run Batch Analysis")
        show_gallery = st.checkbox("Show Gallery", value=True)
        thumbs_per_row = st.slider("Thumbnails per row", 2, 6, 4, 1)
        only_present = st.checkbox("Show only PRESENT images", value=False)

    if not uploads or not run:
        st.info("Upload images and click Run Batch Analysis.")
        st.stop()

    with st.spinner(f"Analyzing {len(uploads)} images (batch endpoint)..."):
        resp = call_api_batch(uploads)

    if resp.status_code != 200:
        st.error(f"Batch API Error {resp.status_code}: {resp.text}")
        st.stop()

    batch_data = resp.json()
    results = []
    overlays = {}

    progress = st.progress(0)
    for i, (up, item) in enumerate(zip(uploads, batch_data), start=1):
        try:
            img_pil = Image.open(io.BytesIO(up.getvalue())).convert("RGB")
            overlay = draw_boxes(img_pil, item.get("detections", []))
            buf = io.BytesIO()
            overlay.save(buf, format="PNG")
            overlays[up.name] = buf.getvalue()
        except Exception:
            overlays[up.name] = None

        dets = item.get("detections", [])
        metrics = item.get("metrics", {})
        flat = flatten_metrics(metrics)

        results.append(
            {
                "filename": up.name,
                "present": bool(item.get("present")),
                "confidence": item.get("confidence"),
                "plate_text": item.get("plate_text"),
                "ocr_confidence": item.get("ocr_confidence"),
                "num_detections": len(dets),
                "latency_ms": item.get("latency_ms"),
                "captured_at_readable": item.get("captured_at_readable"),
                **flat,
            }
        )

        progress.progress(i / len(uploads))

    df = pd.DataFrame(results)

    with colB:
        st.subheader("Batch Summary")
        total = len(df)
        present_ct = int(df["present"].sum()) if "present" in df else 0
        absent_ct = total - present_ct
        c1, c2, c3 = st.columns(3)
        c1.metric("Total", total)
        c2.metric("Present", present_ct)
        c3.metric("Absent", absent_ct)

    st.subheader("Results Table")
    st.dataframe(df, use_container_width=True)

    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download CSV Report",
        data=csv_bytes,
        file_name="numberplate_report.csv",
        mime="text/csv",
        use_container_width=True,
    )

    if show_gallery:
        st.subheader("Gallery (BBox Overlay)")
        df_gallery = df.copy()
        if only_present:
            df_gallery = df_gallery[df_gallery["present"] == True]

        names = df_gallery["filename"].tolist()
        if not names:
            st.info("No images to show (filters removed all).")
        else:
            rows = [names[i:i + thumbs_per_row] for i in range(0, len(names), thumbs_per_row)]
            for row in rows:
                cols = st.columns(len(row))
                for j, name in enumerate(row):
                    with cols[j]:
                        if overlays.get(name) is not None:
                            st.image(overlays[name], caption=name, use_container_width=True)
                        else:
                            st.warning(f"No preview for {name}")
                        row_data = df_gallery[df_gallery["filename"] == name].iloc[0]
                        st.write("PRESENT" if row_data["present"] else "ABSENT")
                        st.caption(f"conf: {row_data.get('confidence')}")
                        st.caption(f"plate: {row_data.get('plate_text') or '-'}")
                        st.caption(f"ocr_conf: {row_data.get('ocr_confidence')}")
                        st.caption(f"dets: {row_data.get('num_detections')}")
                        st.caption(f"latency_ms: {row_data.get('latency_ms')}")
