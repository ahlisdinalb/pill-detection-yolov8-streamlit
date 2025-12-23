import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import os
import tempfile
from pathlib import Path
import shutil
# from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration, WebRtcMode
# import av

st.set_page_config(page_title="Pill Detection", layout="wide")
st.title("Pill Detection & Counting (YOLOv8)")

MODEL_PATH = "best.pt"

if not os.path.exists(MODEL_PATH):
    st.warning("Model tidak ditemukan.")
    st.stop()

# load model once
model = YOLO(MODEL_PATH)

# Sidebar controls
st.sidebar.header("Settings")
conf = st.sidebar.slider("Confidence threshold", 0.1, 0.9, 0.50, step=0.05)
imgsz = st.sidebar.selectbox("Image size (inference)", [320, 416, 640, 960], index=2)
show_labels = st.sidebar.checkbox("Show class/conf labels on boxes", value=True)

# helper: draw boxes + count on numpy image (RGB)
def annotate_frame_rgb(frame_rgb: np.ndarray, results, show_labels=True):
    if frame_rgb.dtype != np.uint8:
        frame_rgb = np.clip(frame_rgb * 255.0, 0, 255).astype(np.uint8)
    if frame_rgb.ndim == 2:
        frame_rgb = cv2.cvtColor(frame_rgb, cv2.COLOR_GRAY2RGB)
    if frame_rgb.shape[2] == 4:
        frame_rgb = frame_rgb[:, :, :3]

    # convert to BGR for drawing with cv2
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

    count = 0
    try:
        boxes = results.boxes
    except Exception:
        boxes = []
    for b in boxes:
        x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
        score = float(b.conf[0]) if hasattr(b, "conf") else 0.0
        if score < conf:
            continue
        count += 1
        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 64, 255), 2) 
        if show_labels:
            label = f"pill {score:.2f}"
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame_bgr, (x1, y1 - 18), (x1 + w + 6, y1), (0,64,255), -1)
            cv2.putText(frame_bgr, label, (x1 + 3, y1 - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)

    # overlay total count text (top-left)
    cv2.putText(frame_bgr, f"Total Pills: {count}", (18, 36),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)

    return frame_bgr, count

# ---------- IMAGE MODE ----------
def handle_image_upload(uploaded_file):
    img = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(img) 
    res = model(img_np, imgsz=imgsz, conf=conf)[0]
    annotated_bgr, count = annotate_frame_rgb(img_np, res, show_labels=show_labels)
    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
    annotated_pil = Image.fromarray(annotated_rgb)
    return annotated_pil, count

# ---------- VIDEO MODE ----------
def process_uploaded_video(temp_video_path: str):
    """Process uploaded video, save annotated video, return output path and stats."""
    cap = cv2.VideoCapture(temp_video_path)
    if not cap.isOpened():
        raise RuntimeError("Gagal membuka video.")

    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_fd, out_path = tempfile.mkstemp(suffix=".mp4")
    os.close(out_fd)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h), isColor=True)

    frame_idx = 0
    counts = []
    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        try:
            res = model(frame_rgb, imgsz=imgsz, conf=conf)[0]
        except Exception as e:
            writer.write(frame_bgr)
            continue
        annotated_bgr, n = annotate_frame_rgb(frame_rgb, res, show_labels=show_labels)
        if annotated_bgr.dtype != np.uint8:
            annotated_bgr = np.clip(annotated_bgr, 0, 255).astype(np.uint8)
        writer.write(annotated_bgr)
        counts.append(n)
        frame_idx += 1

    cap.release()
    writer.release()

    stats = {
        "frames": frame_idx,
        "min_count": int(min(counts)) if counts else 0,
        "max_count": int(max(counts)) if counts else 0,
        "avg_count": float(sum(counts) / len(counts)) if counts else 0.0,
    }
    return out_path, stats

# ---------- WEBRTC / WEBCAM MODE ----------
use_webrtc = True
try:
    pass
except Exception as e:
    use_webrtc = False

# if use_webrtc:
    # class PillVideoTransformer(VideoTransformerBase):
    #     """
    #     This transformer runs in a separate thread. We capture the model and settings
    #     when the transformer is created (so set conf/imgsz before pressing start).
    #     """
    #     def __init__(self):
    #         self._model = model
    #         self._conf = float(conf)
    #         self._imgsz = int(imgsz)
    #         self._show_labels = bool(show_labels)

    #     def recv(self, frame):
    #         img_bgr = frame.to_ndarray(format="bgr24")
    #         img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    #         try:
    #             res = self._model(img_rgb, imgsz=self._imgsz, conf=self._conf)[0]
    #         except Exception:
    #             return av.VideoFrame.from_ndarray(img_bgr, format="bgr24")

    #         annotated_bgr, n = annotate_frame_rgb(img_rgb, res, show_labels=self._show_labels)
    #         return av.VideoFrame.from_ndarray(annotated_bgr, format="bgr24")

# else:
#     PillVideoTransformer = None

# ---------- FRONTEND UI ----------
mode = st.radio("Pilih Mode", ["Upload Image", "Upload Video", "Webcam (Disable)"])

if mode == "Upload Image":
    uploaded_image = st.file_uploader("Upload gambar (jpg/png/jpeg)", type=["jpg", "jpeg", "png"])
    if uploaded_image:
        with st.spinner("Running inference..."):
            annotated_pil, total = handle_image_upload(uploaded_image)
        st.image(annotated_pil, caption=f"Jumlah pil: {total}", use_column_width=True)
        # allow download
        buf = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        annotated_pil.save(buf.name)
        with open(buf.name, "rb") as f:
            st.download_button("Download annotated image", data=f, file_name="annotated_image.png", mime="image/png")
        try:
            os.unlink(buf.name)
        except Exception:
            pass

elif mode == "Upload Video":
    uploaded_video = st.file_uploader("Upload video (mp4/mov)", type=["mp4", "mov", "avi"])
    if uploaded_video:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_video.name).suffix)
        tfile.write(uploaded_video.read())
        tfile.flush()
        tfile.close()
        st.info("Memproses video...")
        with st.spinner("Processing video (frame-by-frame)..."):
            try:
                out_path, stats = process_uploaded_video(tfile.name)
                st.success(f"Selesai. Frames: {stats['frames']} | min/max/avg: {stats['min_count']}/{stats['max_count']}/{stats['avg_count']:.2f}")
                st.video(out_path)
                with open(out_path, "rb") as f:
                    st.download_button("Download annotated video", f, file_name="annotated_video.mp4", mime="video/mp4")
            except Exception as e:
                st.error(f"Error processing video: {e}")
            finally:
                try:
                    os.unlink(tfile.name)
                except Exception:
                    pass

else:  # Webcam
    if not use_webrtc:
        st.error("Realtime webcam tidak tersedia karena package 'streamlit-webrtc' atau dependensinya belum terinstall.")
        st.info("Install dengan: pip install streamlit-webrtc av")
    else:
        st.info("Pastikan kamu set Confidence & Image size sebelum menekan tombol Start.")
        RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

        webrtc_ctx = webrtc_streamer(
            key="pill-webcam",
            mode=WebRtcMode.SENDRECV, 
            video_transformer_factory=PillVideoTransformer,
            rtc_configuration=RTC_CONFIGURATION,
            media_stream_constraints={
                "video": {"facingMode": "user", "width": {"ideal": imgsz}, "height": {"ideal": imgsz}},
                "audio": False
            },
            async_transform=True,
        )

        if webrtc_ctx.state.playing:
            st.success("Webcam aktif — menampilkan deteksi real-time.")
        else:
            st.warning("Klik tombol *Start* pada widget webcam (pojok kiri atas) untuk mengaktifkan kamera.")

st.markdown("---")
st.markdown("Catatan: setting confidence & image size dibaca saat proses mulai. Untuk mengubah parameter webcam real-time, hentikan stream lalu ubah slider lalu mulai lagi.")
