# 🚗 Car Number Plate Reader (ANPR) — Full-Stack Web App

An end-to-end, real-time **Automatic Number Plate Recognition** system with a proper backend + frontend:

```
Camera → capture frame by frame (OpenCV)
       → YOLO detects the plate, draws a box around it
       → crop out just the plate region (ignore the rest of the car)
       → enhance the crop: resize, denoise, fix brightness/contrast
       → OCR reads the characters (Tesseract)
       → served live to a browser dashboard (MJPEG stream)
       → and logged to a local SQLite database, viewable in the same dashboard
```

**Backend:** FastAPI (Python) — REST API + live MJPEG video stream + SQLite storage.
**Frontend:** a single-page HTML/CSS/JS dashboard — live camera feed, drag-and-drop image testing, and a live-updating table of logged plates. No build step, no Node.js required.

---

## Project structure

```
car-number-plate-reader/
├── README.md
├── requirements.txt
├── config.py                  # all tunable settings live here
├── backend/
│   ├── __init__.py
│   └── api.py                  # FastAPI app: REST endpoints + MJPEG video stream
├── frontend/
│   ├── index.html               # the dashboard page
│   ├── style.css                 # dashboard styling
│   └── app.js                     # live feed control, upload, polling the plates table
├── main.py                    # CLI alternative: LIVE pipeline in an OpenCV window (no browser)
├── image_main.py              # CLI alternative: single image, no camera/browser needed
├── view_logs.py                # CLI: print recently logged plates
├── src/
│   ├── __init__.py
│   ├── yolo_detector.py       # YOLO-based plate localization (pretrained, auto-downloaded)
│   ├── ocr.py                 # crop enhancement (resize/denoise/brightness) + Tesseract OCR
│   ├── pipeline.py             # wires detector + OCR together, draws results
│   ├── database.py             # SQLite storage of every recognized plate
│   └── utils.py                 # image I/O + synthetic test-image generator
├── output/
│   └── snapshots/              # auto-saved crops of each newly-logged plate
├── database/
│   └── plates.db                # created automatically on first run
└── tests/
    └── test_pipeline.py         # OCR / database / pipeline / API smoke tests
```

---

## 1. Setup

### System dependency: Tesseract OCR engine
**Windows:** download & install from https://github.com/UB-Mannheim/tesseract/wiki.
**macOS:** `brew install tesseract`
**Linux:** `sudo apt-get install -y tesseract-ocr`

The code **auto-detects** Tesseract if it's on your PATH, or at the common Windows install location (`C:\Program Files\Tesseract-OCR\tesseract.exe`). If you installed it somewhere else, pass the path explicitly — see `src/ocr.py`.

### Python dependencies
```bash
pip install -r requirements.txt
```
This installs FastAPI/uvicorn for the web app, plus `yolov5` (pulls in PyTorch — a few hundred MB, may take a few minutes). **The first time the pipeline runs**, it also auto-downloads the pretrained plate-detection weights (~14 MB) from Hugging Face.

---

## 2. Usage — the web app (recommended)

Start the backend:
```bash
uvicorn backend.api:app --reload --port 8000
```
Then open **http://localhost:8000** in your browser. You'll see:
- **Live feed** — enter a camera index (0, 1, 2...) and click **Start feed** to see the live annotated video stream, powered by the same detect → crop → enhance → OCR pipeline, streamed as MJPEG.
- **Test a single image** — drag and drop (or browse for) a photo; see the detected plate(s) and OCR text immediately, no camera needed.
- **Recent reads** — a live table of everything logged to the database, auto-refreshing every few seconds.

### Key API endpoints (for scripting/integration)
| Endpoint | Method | What it does |
|---|---|---|
| `/api/health` | GET | Liveness check |
| `/api/detect` | POST | Upload an image (`multipart/form-data`, field `file`); returns detected plates + annotated image (base64) |
| `/api/video_feed?camera=0` | GET | MJPEG live stream with detection overlay |
| `/api/plates?limit=50` | GET | Recent logged plates as JSON |
| `/api/snapshot/{filename}` | GET | Serve a saved plate snapshot image |

---

## 3. Usage — CLI alternatives (no browser)

### Live camera in a native OpenCV window
```bash
python main.py
python main.py --camera 1
python main.py --video path/to/footage.mp4
```
Press **q** to quit, **s** to save a manual snapshot.

### Single image
```bash
python image_main.py --image sample_images/car.jpg
```

### View what's been logged
```bash
python view_logs.py
```

### Run the tests
```bash
python -m pytest tests/ -v
```

---

## 4. How it works, step by step

1. **Capture** — `cv2.VideoCapture` pulls frames from the webcam (live-feed CLI/API) or a single uploaded image is decoded directly (upload path).
2. **Detection (`src/yolo_detector.py`)** — a **pretrained YOLOv5 model**, fine-tuned specifically on license plates (`keremberke/yolov5s-license-plate`, auto-downloaded from Hugging Face), returns bounding boxes + confidence scores.
3. **Cropping (`src/pipeline.py`)** — each box is padded slightly and cropped out — only the plate area is processed from here on.
4. **Enhancement (`src/ocr.py`)** — the crop is resized/upscaled 3x, denoised, brightness/contrast-normalized (CLAHE), and binarized (Otsu thresholding).
5. **OCR** — `pytesseract` reads the cleaned-up crop, restricted to A–Z/0–9 characters only.
6. **Delivery** — the API streams annotated frames as MJPEG (for the live feed) or returns a single annotated JPEG as base64 (for uploads); the frontend renders both plus a polling table of the database.
7. **Storage (`src/database.py`)** — plate text, detection confidence, a timestamp, and a snapshot path are inserted into SQLite, with de-duplication so the same plate isn't logged repeatedly while it's in frame.

---

## 5. Tuning (`config.py`)

| Setting | What it does |
|---|---|
| `CAMERA_INDEX` | Default webcam used by the CLI and the frontend's camera field |
| `MODEL_NAME` | Which pretrained YOLO plate model to load |
| `CONF_THRESHOLD` | Minimum YOLO confidence to keep a detected box |
| `MIN_PLATE_TEXT_LENGTH` | OCR reads shorter than this are flagged low-confidence instead of logged |
| `DEDUPE_SECONDS` | How long to wait before re-logging the same plate text again |
| `SAVE_SNAPSHOTS` | Whether to save a cropped image alongside each database entry |

---

## 6. Extending this project

- **Train your own YOLO model** on your region's plates, then point `config.MODEL_NAME` at your local `best.pt` file path instead of the Hugging Face model id.
- **Vehicle tracking** — add object tracking so the same physical plate across many frames is treated as one continuous read.
- **Auth** — the API currently has no authentication; add an API key or session auth before exposing it beyond localhost.
- **Alerts** — check each newly logged plate against a watchlist table and trigger a notification on a match.
- **Deploy** — for a permanently-running installation (e.g. a gate/parking system), run the backend as a service (e.g. via `systemd`, Docker, or Windows Task Scheduler) instead of `uvicorn --reload`.

---

## 7. Known limitations

- Tesseract can still confuse visually similar characters (`0`/`O`, `1`/`I`, `5`/`S`, `8`/`B`) on very low-resolution or heavily-angled crops.
- The pretrained YOLO model generalizes well but isn't tuned to any one country's specific plate design.
- The MJPEG live feed runs the full pipeline (YOLO + OCR) on every frame — CPU-heavy on lower-end machines. If it feels laggy, process every Nth frame instead (a quick change in `backend/api.py`'s `_mjpeg_generator`).
- The API has no authentication and is intended for local/trusted-network use as shipped.

