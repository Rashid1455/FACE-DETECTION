# FaceScope — Streamlit Face Detection

A face detection app built with **Streamlit + OpenCV**.

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run
streamlit run app.py
```

App opens at **http://localhost:8501**

## Features

| Tab | What it does |
|-----|-------------|
| 🖼 **Image** | Upload photo → faces highlighted with bounding boxes + corner accents. Download result. |
| 🎬 **Video** | Upload video → every frame processed with live preview + progress bar. Download output. |
| 📷 **Live Camera** | Toggle-based webcam detection with live FPS counter (runs on host machine). |

## Sidebar Settings (tune detection)

| Setting | Effect |
|---------|--------|
| Scale Factor | Lower → catches smaller/more faces; slower |
| Min Neighbors | Higher → fewer false positives |
| Min Face Size | Ignore faces smaller than this (px) |
| Detect Eyes | Also highlight eyes within each face |
| Box Color | Pick any annotation color |

## Notes

- **Live Camera** uses OpenCV's `VideoCapture` — requires a webcam on the machine running Streamlit.  
  If running on a remote server, use the Image tab instead.
- Video processing writes a temp file and streams a download on completion.
- All detection uses OpenCV's Haar Cascade (`haarcascade_frontalface_default.xml`) — no GPU needed.
