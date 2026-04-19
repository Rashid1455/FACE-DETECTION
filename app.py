"""
FaceScope — Face Detection in Streamlit
Run:  streamlit run app.py
"""

import cv2
import numpy as np
import streamlit as st
import tempfile
import os
import time
from PIL import Image

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="FaceScope — Face Detection",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;800&display=swap');

/* Global */
html, body, [class*="css"] { font-family: 'Syne', sans-serif; }

.stApp {
    background: #080b10;
    color: #e8edf5;
}

/* Hide default header */
header[data-testid="stHeader"] { background: transparent; }
.block-container { padding-top: 2rem; padding-bottom: 2rem; }

/* Title */
.hero-title {
    background: linear-gradient(135deg, #00ff9d 0%, #00c8ff 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-family: 'Syne', sans-serif;
    font-size: 2.8rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    line-height: 1;
    margin-bottom: 0.2rem;
}
.hero-sub {
    font-family: 'Space Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.18em;
    color: #4a5568;
    text-transform: uppercase;
    margin-bottom: 2rem;
}
.divider {
    height: 1px;
    background: linear-gradient(90deg, #00ff9d33, #00c8ff33, transparent);
    margin: 1.5rem 0;
}

/* Metric cards */
.metric-row { display: flex; gap: 1rem; flex-wrap: wrap; margin: 1rem 0; }
.metric-card {
    background: #111820;
    border: 1px solid #1a2535;
    border-radius: 10px;
    padding: 1rem 1.5rem;
    min-width: 110px;
    text-align: center;
    flex: 1;
}
.metric-num {
    font-size: 2.2rem;
    font-weight: 800;
    background: linear-gradient(135deg, #00ff9d, #00c8ff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1;
}
.metric-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.62rem;
    color: #4a5568;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-top: 0.3rem;
}

/* Info banner */
.info-banner {
    background: rgba(0,255,157,0.07);
    border: 1px solid rgba(0,255,157,0.25);
    border-radius: 10px;
    padding: 0.9rem 1.2rem;
    font-family: 'Space Mono', monospace;
    font-size: 0.78rem;
    color: #00ff9d;
    letter-spacing: 0.05em;
    margin: 1rem 0;
}

/* Warning banner */
.warn-banner {
    background: rgba(255,200,0,0.07);
    border: 1px solid rgba(255,200,0,0.25);
    border-radius: 10px;
    padding: 0.9rem 1.2rem;
    font-family: 'Space Mono', monospace;
    font-size: 0.78rem;
    color: #ffc800;
    margin: 1rem 0;
}

/* Section label */
.section-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #00ff9d;
    margin-bottom: 0.8rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

/* Tab styling override */
.stTabs [data-baseweb="tab-list"] {
    background: #0e1318;
    border-radius: 10px;
    padding: 4px;
    border: 1px solid #1a2535;
    gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 7px;
    color: #4a5568;
    font-family: 'Syne', sans-serif;
    font-weight: 600;
    font-size: 0.9rem;
    padding: 0.55rem 1.4rem;
    background: transparent;
    border: none;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, rgba(0,255,157,0.15), rgba(0,200,255,0.15)) !important;
    color: #00ff9d !important;
    border: 1px solid rgba(0,255,157,0.3) !important;
}
.stTabs [data-baseweb="tab-highlight"] { display: none; }
.stTabs [data-baseweb="tab-border"] { display: none; }

/* Slider */
.stSlider [data-baseweb="slider"] { }
.stSlider [data-baseweb="thumb"] { background: #00ff9d; }

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #00ff9d, #00c8ff);
    color: #000;
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    border: none;
    border-radius: 8px;
    padding: 0.6rem 1.5rem;
    font-size: 0.9rem;
    transition: all 0.2s;
    letter-spacing: 0.03em;
}
.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 6px 20px rgba(0,255,157,0.3);
}

/* Image captions */
.stImage > div > div {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    color: #4a5568;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

/* Spinner */
.stSpinner > div { border-top-color: #00ff9d !important; }

/* Sidebar */
.css-1d391kg { background: #0e1318; }

/* Success / error */
.stSuccess { background: rgba(0,255,157,0.1); border-color: #00ff9d; }
.stError { background: rgba(255,77,109,0.1); border-color: #ff4d6d; }
</style>
""", unsafe_allow_html=True)


# ── Detection helpers ─────────────────────────────────────────
@st.cache_resource
def load_cascades():
    face_cc = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cc  = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_eye.xml')
    return face_cc, eye_cc

face_cascade, eye_cascade = load_cascades()


def detect_faces(img_bgr, scale=1.1, neighbors=5, min_size=30,
                 show_eyes=False, color_hex="#00ff9d"):
    """Run detection and draw annotations. Returns (annotated_bgr, face_count)."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=scale, minNeighbors=neighbors,
        minSize=(min_size, min_size))

    # Parse hex color → BGR
    h = color_hex.lstrip('#')
    r, g, b = int(h[0:2],16), int(h[2:4],16), int(h[4:6],16)
    bgr = (b, g, r)

    out = img_bgr.copy()
    count = len(faces) if not isinstance(faces, tuple) else 0

    if count > 0:
        for i, (x, y, w, h_) in enumerate(faces):
            # Main rectangle
            cv2.rectangle(out, (x, y), (x+w, y+h_), bgr, 2)

            # Corner accents
            cl = w // 6
            th = 3
            # TL
            cv2.line(out, (x, y), (x+cl, y), bgr, th)
            cv2.line(out, (x, y), (x, y+cl), bgr, th)
            # TR
            cv2.line(out, (x+w, y), (x+w-cl, y), bgr, th)
            cv2.line(out, (x+w, y), (x+w, y+cl), bgr, th)
            # BL
            cv2.line(out, (x, y+h_), (x+cl, y+h_), bgr, th)
            cv2.line(out, (x, y+h_), (x, y+h_-cl), bgr, th)
            # BR
            cv2.line(out, (x+w, y+h_), (x+w-cl, y+h_), bgr, th)
            cv2.line(out, (x+w, y+h_), (x+w, y+h_-cl), bgr, th)

            # Label badge
            label = f"FACE {i+1}"
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            cv2.rectangle(out, (x, y-24), (x+lw+8, y), bgr, -1)
            cv2.putText(out, label, (x+4, y-7),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)

            # Eyes (optional)
            if show_eyes:
                roi_gray = gray[y:y+h_, x:x+w]
                roi_color = out[y:y+h_, x:x+w]
                eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 10)
                for (ex, ey, ew, eh) in eyes:
                    cv2.circle(roi_color,
                               (ex + ew//2, ey + eh//2),
                               ew//2, (0, 200, 255), 2)

    return out, count


def bgr_to_rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def pil_to_bgr(pil_img):
    return cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2BGR)


# ── Header ────────────────────────────────────────────────────
col_logo, col_status = st.columns([4, 1])
with col_logo:
    st.markdown('<div class="hero-title">FaceScope</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">🔍 &nbsp; Neural Face Detection System</div>', unsafe_allow_html=True)
with col_status:
    st.markdown("<br/>", unsafe_allow_html=True)
    cascade_ok = not face_cascade.empty()
    if cascade_ok:
        st.success("✅ Cascade Loaded", icon=None)
    else:
        st.error("❌ Cascade Missing")

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# ── Sidebar — Detection Settings ─────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Detection Settings")
    st.markdown("---")

    scale = st.slider("Scale Factor", 1.05, 1.5, 1.1, 0.05,
                      help="Smaller = more detections, slower")
    neighbors = st.slider("Min Neighbors", 1, 15, 5,
                          help="Higher = fewer false positives")
    min_size = st.slider("Min Face Size (px)", 10, 100, 30)
    show_eyes = st.toggle("Detect Eyes Too", value=False)
    color = st.color_picker("Box Color", "#00ff9d")

    st.markdown("---")
    st.markdown("#### About")
    st.markdown("""
<div style="font-family:'Space Mono',monospace; font-size:0.7rem; color:#4a5568; line-height:1.8">
Engine: OpenCV Haar Cascade<br/>
Model: frontalface_default<br/>
Eye model: haarcascade_eye<br/>
</div>
""", unsafe_allow_html=True)


# ── Main Tabs ─────────────────────────────────────────────────
tab_img, tab_vid, tab_live = st.tabs(["🖼️  Image", "🎬  Video", "📷  Live Camera"])


# ═══════════════════════════════════════════════════════════════
# TAB 1 — IMAGE
# ═══════════════════════════════════════════════════════════════
with tab_img:
    st.markdown('<p class="section-label">▸ Upload an image to detect faces</p>',
                unsafe_allow_html=True)

    uploaded_img = st.file_uploader(
        "Drop image here", type=["jpg", "jpeg", "png", "bmp", "webp"],
        label_visibility="collapsed", key="img_upload")

    if uploaded_img:
        pil_img = Image.open(uploaded_img)
        img_bgr = pil_to_bgr(pil_img)

        with st.spinner("🔍 Detecting faces..."):
            result_bgr, face_count = detect_faces(
                img_bgr, scale, neighbors, min_size, show_eyes, color)

        # Metrics
        h_px, w_px = img_bgr.shape[:2]
        st.markdown(f"""
        <div class="metric-row">
          <div class="metric-card">
            <div class="metric-num">{face_count}</div>
            <div class="metric-label">Faces Found</div>
          </div>
          <div class="metric-card">
            <div class="metric-num">{w_px}×{h_px}</div>
            <div class="metric-label">Resolution</div>
          </div>
          <div class="metric-card">
            <div class="metric-num">{uploaded_img.size // 1024}<span style="font-size:1rem">KB</span></div>
            <div class="metric-label">File Size</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        if face_count == 0:
            st.markdown('<div class="warn-banner">⚠ No faces detected — try lowering Min Neighbors or Min Face Size in the sidebar.</div>',
                        unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="info-banner">✔ Detected {face_count} face{"s" if face_count != 1 else ""} successfully.</div>',
                        unsafe_allow_html=True)

        # Side-by-side display
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<p class="section-label">Original</p>', unsafe_allow_html=True)
            st.image(bgr_to_rgb(img_bgr), use_container_width=True)
        with col2:
            st.markdown('<p class="section-label">Detected</p>', unsafe_allow_html=True)
            st.image(bgr_to_rgb(result_bgr), use_container_width=True)

        # Download button
        from io import BytesIO
        buf = BytesIO()
        Image.fromarray(bgr_to_rgb(result_bgr)).save(buf, format="JPEG", quality=92)
        st.download_button(
            "⬇ Download Result",
            data=buf.getvalue(),
            file_name="facescope_result.jpg",
            mime="image/jpeg")
    else:
        st.markdown("""
        <div style="background:#111820; border:2px dashed #1a2535; border-radius:16px;
                    padding:4rem 2rem; text-align:center; color:#4a5568;">
            <div style="font-size:3rem; margin-bottom:1rem">🔍</div>
            <div style="font-family:'Syne',sans-serif; font-size:1.1rem; font-weight:700; margin-bottom:0.4rem; color:#e8edf5">
                Upload an image to get started
            </div>
            <div style="font-family:'Space Mono',monospace; font-size:0.8rem">
                JPG · PNG · WEBP · BMP
            </div>
        </div>
        """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# TAB 2 — VIDEO
# ═══════════════════════════════════════════════════════════════
with tab_vid:
    st.markdown('<p class="section-label">▸ Upload a video — every frame will be processed</p>',
                unsafe_allow_html=True)

    uploaded_vid = st.file_uploader(
        "Drop video here", type=["mp4", "avi", "mov", "mkv", "webm"],
        label_visibility="collapsed", key="vid_upload")

    if uploaded_vid:
        # Save to temp file
        tmp_in = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tmp_in.write(uploaded_vid.read())
        tmp_in.close()

        tmp_out_path = tmp_in.name.replace(".mp4", "_out.mp4")

        cap = cv2.VideoCapture(tmp_in.name)
        fps    = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        st.markdown(f"""
        <div class="metric-row">
          <div class="metric-card">
            <div class="metric-num">{total}</div>
            <div class="metric-label">Total Frames</div>
          </div>
          <div class="metric-card">
            <div class="metric-num">{fps:.0f}<span style="font-size:1rem">fps</span></div>
            <div class="metric-label">Frame Rate</div>
          </div>
          <div class="metric-card">
            <div class="metric-num">{width}×{height}</div>
            <div class="metric-label">Resolution</div>
          </div>
          <div class="metric-card">
            <div class="metric-num">{uploaded_vid.size//1024//1024}<span style="font-size:1rem">MB</span></div>
            <div class="metric-label">File Size</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        if st.button("▶ Process Video"):
            progress_bar = st.progress(0, text="Processing frames…")
            status_txt   = st.empty()
            preview_slot = st.empty()

            cap = cv2.VideoCapture(tmp_in.name)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out_writer = cv2.VideoWriter(tmp_out_path, fourcc, fps, (width, height))

            frame_idx  = 0
            max_faces  = 0
            total_faces = 0
            preview_every = max(1, total // 8)

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                annotated, count = detect_faces(
                    frame, scale, neighbors, min_size, show_eyes, color)
                out_writer.write(annotated)

                if count > max_faces:
                    max_faces = count
                total_faces += count
                frame_idx += 1

                pct = int(frame_idx / max(total, 1) * 100)
                progress_bar.progress(pct, text=f"Frame {frame_idx}/{total} — {count} face(s)")

                # Show preview every N frames
                if frame_idx % preview_every == 0:
                    preview_slot.image(
                        bgr_to_rgb(annotated),
                        caption=f"Preview — Frame {frame_idx}",
                        use_container_width=True)

            cap.release()
            out_writer.release()
            progress_bar.progress(100, text="✔ Done!")

            st.markdown(f"""
            <div class="metric-row">
              <div class="metric-card">
                <div class="metric-num">{max_faces}</div>
                <div class="metric-label">Max Faces</div>
              </div>
              <div class="metric-card">
                <div class="metric-num">{frame_idx}</div>
                <div class="metric-label">Frames Done</div>
              </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown('<div class="info-banner">✔ Processing complete. Download your video below.</div>',
                        unsafe_allow_html=True)

            # Offer download
            with open(tmp_out_path, "rb") as f:
                st.download_button(
                    "⬇ Download Processed Video",
                    data=f.read(),
                    file_name="facescope_video.mp4",
                    mime="video/mp4")

            # Cleanup
            try:
                os.unlink(tmp_in.name)
                os.unlink(tmp_out_path)
            except Exception:
                pass
    else:
        st.markdown("""
        <div style="background:#111820; border:2px dashed #1a2535; border-radius:16px;
                    padding:4rem 2rem; text-align:center; color:#4a5568;">
            <div style="font-size:3rem; margin-bottom:1rem">🎬</div>
            <div style="font-family:'Syne',sans-serif; font-size:1.1rem; font-weight:700; margin-bottom:0.4rem; color:#e8edf5">
                Upload a video to process
            </div>
            <div style="font-family:'Space Mono',monospace; font-size:0.8rem">
                MP4 · AVI · MOV · MKV · WEBM
            </div>
        </div>
        """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# TAB 3 — LIVE CAMERA
# ═══════════════════════════════════════════════════════════════
with tab_live:
    st.markdown('<p class="section-label">▸ Real-time face detection via webcam</p>',
                unsafe_allow_html=True)

    st.markdown("""
    <div class="warn-banner">
    ℹ  Streamlit accesses the webcam through OpenCV on the <b>server machine</b>.<br/>
    If you're running this locally, your webcam will work directly.<br/>
    If deployed on a remote server, use the <b>Image tab</b> instead.
    </div>
    """, unsafe_allow_html=True)

    col_ctrl1, col_ctrl2, col_ctrl3 = st.columns([1, 1, 3])
    with col_ctrl1:
        cam_index = st.number_input("Camera Index", 0, 5, 0, step=1)
    with col_ctrl2:
        st.markdown("<br/>", unsafe_allow_html=True)
        run_live = st.toggle("▶ Activate Camera", value=False, key="live_toggle")

    if run_live:
        st.markdown('<div class="info-banner">📷 Camera active — detection running live. Toggle off to stop.</div>',
                    unsafe_allow_html=True)

        FRAME_WINDOW = st.empty()
        col_m1, col_m2, col_m3 = st.columns(3)
        faces_metric  = col_m1.empty()
        fps_metric    = col_m2.empty()
        frames_metric = col_m3.empty()

        cap = cv2.VideoCapture(int(cam_index))

        if not cap.isOpened():
            st.error(f"❌ Could not open camera index {cam_index}. "
                     "Try a different index or check your webcam connection.")
        else:
            frame_count = 0
            t_start = time.time()
            fps_display = 0.0

            while run_live:
                ret, frame = cap.read()
                if not ret:
                    st.warning("⚠ Camera frame read failed.")
                    break

                annotated, count = detect_faces(
                    frame, scale, neighbors, min_size, show_eyes, color)

                FRAME_WINDOW.image(
                    bgr_to_rgb(annotated),
                    channels="RGB",
                    use_container_width=True,
                    caption=f"Live Feed — {count} face(s) detected")

                frame_count += 1
                elapsed = time.time() - t_start
                if elapsed >= 1.0:
                    fps_display = frame_count / elapsed
                    frame_count = 0
                    t_start = time.time()

                faces_metric.metric("Faces", count)
                fps_metric.metric("FPS", f"{fps_display:.1f}")
                frames_metric.metric("Frame", frame_count)

                # Re-check toggle state
                run_live = st.session_state.get("live_toggle", False)

            cap.release()
            FRAME_WINDOW.empty()
            st.markdown('<div class="info-banner">■ Camera stopped.</div>',
                        unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="background:#111820; border:2px dashed #1a2535; border-radius:16px;
                    padding:4rem 2rem; text-align:center; color:#4a5568;">
            <div style="font-size:3rem; margin-bottom:1rem">📷</div>
            <div style="font-family:'Syne',sans-serif; font-size:1.1rem; font-weight:700; margin-bottom:0.4rem; color:#e8edf5">
                Toggle "Activate Camera" to start
            </div>
            <div style="font-family:'Space Mono',monospace; font-size:0.8rem">
                Requires webcam access on the host machine
            </div>
        </div>
        """, unsafe_allow_html=True)
