# Car Number Plate Reader

Automatic number plate recognition for Karnataka vehicles. A camera frame, uploaded photo, or uploaded video is scanned for a number plate, read with OCR, matched to a Karnataka RTO, classified by plate color, and stored in a local database that the dashboard reads back after a restart.

```
Camera / image / video
        → YOLO finds the plate
        → crop the plate (not the vehicle)
        → plate color (WHITE / YELLOW / GREEN / UNKNOWN)
        → OCR
        → normalize the registration number
        → Karnataka state, RTO code, and district
        → SQLite log
        → dashboard
```

## Project Overview

The backend is a FastAPI application. It serves the dashboard, a JSON API, and an MJPEG live stream from the computer's webcam. Plate text is normalized (spaces, hyphens, and a few ambiguous OCR characters) and then looked up in a Karnataka RTO table. `KA09` resolves to Mysuru. An unknown code is stored as `Unknown RTO/District` and does not stop the pipeline.

The dashboard is a single HTML/CSS/JS page. There is no separate frontend build.

## Features

- Number plate detection (pretrained YOLOv5 license-plate model)
- OCR (Tesseract)
- Karnataka RTO/district identification
- Number plate color detection (the plate crop only, not the vehicle)
- Vehicle logging that persists in SQLite
- Image processing
- Video processing
- Live webcam detection

## Tech Stack

- Python 3.11+
- FastAPI and Uvicorn
- OpenCV
- YOLOv5 (`keremberke/yolov5s-license-plate`, downloaded on first use)
- Tesseract OCR via `pytesseract`
- SQLite (file `database/plates.db`)
- HTML, CSS, and JavaScript dashboard (no Node.js)

## Installation

From the project directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Install the Tesseract binary as well. The Python package does not include it.

```bash
# macOS
brew install tesseract

# Debian / Ubuntu
sudo apt-get install -y tesseract-ocr

# Windows
# https://github.com/UB-Mannheim/tesseract/wiki
```

The first detection downloads the YOLO weights (about 14 MB) from Hugging Face. PyTorch is installed by `yolov5` and is large.

## Environment Variables

No API keys are required. Optional variables are listed in `.env.example`. Export them in the shell if you want to override the defaults in `config.py`:

| Variable | Default | Purpose |
|---|---|---|
| `CAMERA_INDEX` | `0` | Webcam index |
| `DB_PATH` | `database/plates.db` | SQLite file |
| `SNAPSHOT_DIR` | `output/snapshots` | Cropped plate images |
| `SAVE_SNAPSHOTS` | `1` | Set `0` to skip snapshot files |
| `TESSERACT_CMD` | (empty) | Full path to `tesseract` if it is not on `PATH` |
| `CONF_THRESHOLD` | `0.55` | Minimum YOLO confidence |
| `DEDUPE_SECONDS` | `5` | How long before the same live plate is logged again |
| `MODEL_NAME` | `keremberke/yolov5s-license-plate` | Hugging Face id or a local `best.pt` |

Relative paths are stored inside the project directory, so the log file does not depend on the shell's current directory.

## Database Setup

Nothing has to be created by hand. Starting the app creates `database/plates.db` if it is missing.

If you already have a `plates.db` from an older version of this project, open it with the app once. Missing columns, including `plate_color`, are added with `ALTER TABLE`. Existing rows are kept. Their district is filled in from the plate text when that column was empty. Old rows have plate color `UNKNOWN` because the color was not measured at the time.

To start over, stop the app and delete `database/plates.db`.

## Running Backend

From the project directory, with the virtual environment active:

```bash
python -m uvicorn backend.api:app --host 127.0.0.1 --port 8000
```

## Running Frontend

The dashboard is served by the backend. After the command above, open:

```text
http://127.0.0.1:8000
```

There is no `npm install` and no second server.

CLI alternatives, from the same directory:

```bash
python main.py
python main.py --camera 1
python main.py --video path/to/footage.mp4
python image_main.py --image sample_images/car.jpg
python view_logs.py
python -m pytest tests/ -v
```

## Webcam Setup

The **Start feed** button first asks the backend to open a webcam with OpenCV and stream it as MJPEG. That uses the computer's camera, so the process running Python needs camera permission:

- macOS: System Settings → Privacy & Security → Camera, and enable Terminal, iTerm, Cursor, or whichever app launched `uvicorn`.
- The browser does not need camera permission for this server stream. It only displays the pictures the server sends.

If the server cannot open a camera, the page asks the browser for a camera instead and posts frames to `/api/detect?source=webcam`. Allow the browser prompt when it appears. Stopping the feed, or closing the tab, releases the camera.

The camera index field defaults to `0`. If you have more than one camera, try `1` or `2`.

If neither camera can be opened, the page shows: `Unable to access camera. Please check camera permissions.` The server keeps running.

## Supported Plate Colors

The color is measured on the detected plate region, after the dark characters are masked out.

- `WHITE` — private vehicles
- `YELLOW` — commercial vehicles
- `GREEN` — green plates, including electric vehicles
- `UNKNOWN` — the crop was too small, too dark, or not one of the three colors

## API

| Endpoint | Method | Purpose |
|---|---|---|
| `/api/health` | GET | Liveness check |
| `/api/detect` | POST | Image upload (`file`). Optional `source=webcam` for browser frames |
| `/api/detect_video` | POST | Video upload (`file`) |
| `/api/video_feed?camera=0` | GET | Annotated MJPEG stream |
| `/api/camera/check?camera=0` | GET | Open the camera once and report whether it works |
| `/api/camera/stop` | POST | Ask the live stream to release the camera |
| `/api/plates?limit=50` | GET | Recent log rows |
| `/api/plates/{id}` | DELETE | Delete one log row |
| `/api/snapshot/{filename}` | GET | A saved plate crop |

## Troubleshooting

### Webcam

- `Unable to access camera. Please check camera permissions.` Grant camera access to the app that is running Python, then click Start feed again. On macOS the permission is per application, not per browser tab.
- A blank server feed usually means another program already has the camera. Quit that program, click Stop, then Start.
- Try another index in the `cam` field.
- The browser-camera fallback needs a secure page (`http://127.0.0.1` or `https`). It does not work from a `file://` URL.

### OCR

- `TesseractNotFoundError`: install the Tesseract binary and, if needed, set `TESSERACT_CMD` to its full path.
- Similar characters (`0`/`O`, `1`/`I`, `8`/`B`, `5`/`S`, `2`/`Z`) are corrected only when they sit in the wrong kind of slot and a plain reading does not already fit. The raw OCR string is stored as `ocr_raw` and shown when it differs from the normalized number.
- Very small or blurry plates stay in the log with status `OCR_UNCERTAIN` or `OCR_FAILED` instead of being discarded.

### Database

- Logs live in `database/plates.db`. Restarting the app does not clear them.
- If inserts fail, the API still returns the detection and the server log contains `Failed to save vehicle ...` plus the database error.
- Delete `database/plates.db` only when you want an empty log. The next start creates a new file.

### District lookup

- `KA 09`, `KA-09`, `KA09`, and `KA09AB1234` are the same RTO. `KA09` is Mysuru.
- A code that is not in the Karnataka table (for example `KA99`, or a plate from another state) shows `Unknown RTO/District`.
- The district is saved on the log row. The RTO office name (Mysuru West, Hunsur, and so on) is stored separately as `rto_name`.

### Dependencies

- `pip install -r requirements.txt` must finish before the first run. `yolov5` pulls in PyTorch.
- The YOLO weights download needs a network connection the first time.
- `setuptools` is pinned below version 82 because `yolov5` still imports `pkg_resources`.

### Frontend / backend connection

- Open `http://127.0.0.1:8000` while `uvicorn` is running. The page and the API share that origin.
- If the log table says `log refresh failed`, the backend is not reachable on that port.
- Upload errors are shown in the upload panel. The matching server line is prefixed with `[ERROR]`.

## Project layout

```
car-number-plate-reader/
├── backend/api.py          FastAPI routes, MJPEG stream, image and video uploads
├── frontend/               Dashboard
├── src/pipeline.py         Detect → color → OCR → district
├── src/plate_color.py      WHITE / YELLOW / GREEN / UNKNOWN
├── src/district.py         Normalization and Karnataka RTO table
├── src/ocr.py              Tesseract preprocessing
├── src/yolo_detector.py    Plate localization
├── src/database.py         SQLite log and safe column upgrades
├── src/camera.py           Webcam open / release
├── config.py               Settings and environment overrides
├── main.py                 OpenCV window
├── image_main.py           One image from the command line
└── view_logs.py            Print the log
```
