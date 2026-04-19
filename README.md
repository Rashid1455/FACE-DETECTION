<div align="center">

```
███████╗ █████╗  ██████╗███████╗███████╗ ██████╗ ██████╗ ██████╗ ███████╗
██╔════╝██╔══██╗██╔════╝██╔════╝██╔════╝██╔════╝██╔════╝██╔═══██╗██╔══██╗██╔════╝
█████╗  ███████║██║     █████╗  ███████╗██║     ██║     ██║   ██║██████╔╝█████╗
██╔══╝  ██╔══██║██║     ██╔══╝  ╚════██║██║     ██║     ██║   ██║██╔═══╝ ██╔══╝
██║     ██║  ██║╚██████╗███████╗███████║╚██████╗╚██████╗╚██████╔╝██║     ███████╗
╚═╝     ╚═╝  ╚═╝ ╚═════╝╚══════╝╚══════╝ ╚═════╝ ╚═════╝ ╚═════╝ ╚═╝     ╚══════╝
```

# 🔍 FaceScope — Real-Time Face Detection

**A sleek, production-ready face detection system built with Streamlit & OpenCV.**  
Detect faces in images, videos, and live webcam feeds — all from a single Python file.

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=flat-square&logo=python)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32%2B-FF4B4B?style=flat-square&logo=streamlit)](https://streamlit.io)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B-green?style=flat-square&logo=opencv)](https://opencv.org)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)](LICENSE)

</div>

---

## ✨ What Is FaceScope?

FaceScope is a **fully-featured face detection application** that runs entirely in your browser through a Streamlit interface. Whether you want to analyze a photo, process a full video, or run live detection from your webcam — FaceScope handles it all with zero hassle.

No complex setup. No deep learning frameworks. No GPU required.  
Just Python, OpenCV, and a single command to launch.

---

## 🚀 Features at a Glance

| Feature | Description |
|--------|-------------|
| 🖼️ **Image Detection** | Upload any photo and see faces instantly highlighted with precision corner-accent bounding boxes |
| 🎬 **Video Processing** | Process entire video files frame-by-frame with a real-time progress bar and live preview |
| 📷 **Live Camera Feed** | Real-time webcam face detection with live FPS counter and face count metrics |
| ⚙️ **Tunable Detection** | Adjust scale factor, sensitivity, face size threshold, and box color — all from the sidebar |
| 👁️ **Eye Detection** | Optional eye detection layer rendered inside each detected face region |
| ⬇️ **Download Results** | Export annotated images and processed videos directly from the app |

---

## 📸 Detection Modes

### 🖼️ Image Mode
Drop in a photo (JPG, PNG, WEBP, BMP) and get an instant side-by-side comparison — original vs. annotated. Each detected face gets a labeled bounding box with stylized corner accents. Download your result with one click.

### 🎬 Video Mode
Upload any video file (MP4, AVI, MOV, MKV, WEBM) and watch it process in real time. A live preview updates every few frames so you can see detection happening as it runs. When done, download the fully annotated output video.

### 📷 Live Camera Mode
Toggle the camera switch and your webcam activates instantly. Faces are detected and drawn on a live canvas stream, with live metrics displayed:
- 👤 Current face count
- ⚡ Frames per second
- 🎞️ Total frames processed

> **Note:** Live camera uses OpenCV's `VideoCapture` on the server machine. Works perfectly for local setups. For cloud deployments, use the Image tab instead.

---

## ⚙️ Detection Settings (Sidebar)

Fine-tune detection behavior without restarting the app:

| Setting | Range | What it controls |
|---------|-------|-----------------|
| **Scale Factor** | 1.05 – 1.5 | How aggressively to search at different sizes. Lower = more detections, slower. |
| **Min Neighbors** | 1 – 15 | Confidence threshold. Higher = fewer false positives. |
| **Min Face Size** | 10 – 100px | Ignore faces smaller than this. |
| **Detect Eyes** | Toggle | Adds eye detection circles inside each face region. |
| **Box Color** | Color Picker | Customize the annotation color to anything you like. |

---

## 🛠️ Installation & Setup

### Prerequisites

- Python 3.8 or higher
- A webcam (for Live Camera mode only)

### Step 1 — Clone or download the project

```bash
git clone https://github.com/yourname/facescope.git
cd facescope
```

### Step 2 — Install dependencies

```bash
pip install -r requirements.txt
```

### Step 3 — Launch the app

```bash
streamlit run app.py
```

The app will open automatically at **http://localhost:8501** 🎉

---

## 📦 Requirements

```
streamlit>=1.32.0
opencv-python>=4.8.0
numpy>=1.24.0
Pillow>=10.0.0
```

Install everything at once:
```bash
pip install streamlit opencv-python numpy Pillow
```

---

## 🗂️ Project Structure

```
facescope/
│
├── app.py              ← 🧠 Main application (Streamlit + OpenCV logic)
├── requirements.txt    ← 📦 Python dependencies
└── README.md           ← 📖 You're here
```

That's it. One file to rule them all.

---

## 🔧 How It Works

FaceScope uses **OpenCV's Haar Cascade Classifier** — a fast, lightweight detection algorithm that requires no GPU and no pre-downloaded model files (the cascade ships bundled with OpenCV itself).

```
User Input
    │
    ├── Image  ──► PIL decode ──► OpenCV BGR ──► detectMultiScale() ──► Annotated Output
    │
    ├── Video  ──► Frame-by-frame capture ──► detectMultiScale() ──► VideoWriter ──► Download
    │
    └── Camera ──► VideoCapture(0) ──► Real-time frame loop ──► Streamlit image stream
```

**Detection pipeline per frame:**
1. Convert frame to grayscale
2. Run `detectMultiScale()` with your chosen parameters
3. Draw bounding rectangles + corner accent lines + labeled badges
4. Optionally detect and circle eyes within each face ROI
5. Return annotated RGB frame to Streamlit

---

## 🎨 Annotation Style

Each detected face is marked with:
- A **full bounding rectangle** in your chosen color
- **Corner accent lines** at all four corners for a clean, modern look
- A **dark label badge** (e.g., `FACE 1`, `FACE 2`) pinned above each box
- Optional **eye circles** rendered in contrasting cyan

---

## ❓ FAQ

**Q: Does this work on Mac / Linux / Windows?**  
A: Yes — fully cross-platform. Anywhere Python and OpenCV run, FaceScope runs.

**Q: Do I need a GPU?**  
A: No. Haar Cascades are CPU-based and run fast even on modest hardware.

**Q: Why isn't my face being detected?**  
A: Try lowering **Min Neighbors** to 2–3 and **Min Face Size** to 20px in the sidebar. Poor lighting and extreme angles reduce accuracy.

**Q: The live camera shows a black screen.**  
A: Make sure no other app is using your webcam. Try changing **Camera Index** from `0` to `1` if you have multiple cameras.

**Q: Can I deploy this on a remote server?**  
A: Image and Video tabs work on any server. Live Camera requires a physical webcam on the machine running Streamlit.

**Q: How large a video can I process?**  
A: Practical limit depends on available RAM. Videos up to ~200MB process comfortably on most machines.

---

## 🗺️ Roadmap

Some ideas for future versions:

- [ ] Deep learning-based detection (DNN / MediaPipe)
- [ ] Face blurring / anonymization mode
- [ ] Emotion detection overlay
- [ ] Face count statistics export (CSV)
- [ ] Multi-camera support
- [ ] Batch image processing

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!  
Feel free to open a pull request or file an issue on GitHub.

---

## 📄 License

This project is licensed under the **MIT License** — free to use, modify, and distribute.

---

<div align="center">

Built with ❤️ using [Streamlit](https://streamlit.io) + [OpenCV](https://opencv.org)

**⭐ Star this repo if you found it useful!**

</div>
