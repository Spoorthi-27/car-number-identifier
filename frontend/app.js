const feedImg = document.getElementById("feedImg");
const browserVideo = document.getElementById("browserVideo");
const feedPlaceholder = document.getElementById("feedPlaceholder");
const feedHint = document.getElementById("feedHint");
const cameraStatus = document.getElementById("cameraStatus");
const cameraStatusText = document.getElementById("cameraStatusText");
const cameraIndexInput = document.getElementById("cameraIndex");
const startBtn = document.getElementById("startBtn");
const stopBtn = document.getElementById("stopBtn");
const liveResult = document.getElementById("liveResult");

const dropzone = document.getElementById("dropzone");
const fileInput = document.getElementById("fileInput");
const browseBtn = document.getElementById("browseBtn");
const uploadResult = document.getElementById("uploadResult");

const logTableBody = document.getElementById("logTableBody");
const logCount = document.getElementById("logCount");

const idlePlaceholder = feedPlaceholder.innerHTML;
const CAMERA_ERROR = "Unable to access camera. Please check camera permissions.";

let feedRunning = false;
let browserStream = null;
let browserTimer = null;
let browserRequest = 0;
let liveSinceId = 0;

function escapeHtml(value) {
  return String(value ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function confidencePercent(value) {
  if (value === null || value === undefined || value === "") return "—";
  return `${Math.round(Number(value) * 100)}%`;
}

function colorBadge(color) {
  const name = color || "UNKNOWN";
  return `<span class="color-badge color-${escapeHtml(name)}">${escapeHtml(name)}</span>`;
}

function statusBadge(status) {
  const name = status || "—";
  const kind = name === "VALID" ? "valid" : name === "OCR_FAILED" ? "failed" : "uncertain";
  return `<span class="status-badge status-${kind}">${escapeHtml(name)}</span>`;
}

  // ---------- CSV download (single detection + history) ----------

function csvEscape(value) {
  const text = String(value ?? "");
  return /[",\r\n]/.test(text) ? `"${text.replace(/"/g, '""')}"` : text;
}

// Same column order as the backend's /api/plates/download and
// /api/plates/{id}/download, used here for detections not yet saved to
// the database (e.g. a deduped live-feed read has no id to fetch by).
function buildDetailCsv(plate) {
  const rows = [
    ["Vehicle Number", plate.plate_text || plate.text || ""],
    ["State", plate.state || ""],
    ["RTO Code", plate.rto_code || ""],
    ["RTO Office", plate.rto_name || ""],
    ["District", plate.district || "Unknown RTO/District"],
    ["Plate Color", plate.plate_color || "UNKNOWN"],
    ["Vehicle Type", plate.vehicle_type || "Unknown"],
    ["Plate Detection Confidence", confidencePercent(plate.confidence ?? plate.detection_conf)],
    ["OCR Confidence", confidencePercent(plate.ocr_confidence)],
    ["Raw OCR", plate.raw_text || ""],
    ["Source", plate.source || ""],
    ["Date", plate.date || ""],
    ["Time", plate.time || ""],
    ["Status", plate.status || ""],
  ];
  const lines = [rows.map(([label]) => csvEscape(label)).join(",")];
  lines.push(rows.map(([, value]) => csvEscape(value)).join(","));
  return "\ufeff" + lines.join("\r\n");
}

function triggerCsvDownload(filename, csvText) {
  const blob = new Blob([csvText], { type: "text/csv;charset=utf-8;" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
}

// Detection cards are rendered from plain data, so the record for each
// "Download Details" button is kept here rather than round-tripped through
// the DOM; the button only carries the small key needed to look it up.
let detectionRecordSeq = 0;

const detectionRecords = new Map();

function renderDetection(plate) {
  const vehicle = plate.plate_text || plate.text || "—";
  const rows = [
    ["Vehicle", `<strong>${escapeHtml(vehicle)}</strong>`],
    ["State", escapeHtml(plate.state || "—")],
    ["RTO", escapeHtml(plate.rto_code || "—")],
    ["District", escapeHtml(plate.district || "Unknown RTO/District")],
    ["Plate Color", colorBadge(plate.plate_color)],
    ["Vehicle Type", escapeHtml(plate.vehicle_type || "Unknown")],
    ["Confidence", escapeHtml(confidencePercent(plate.confidence ?? plate.detection_conf))],
    ["Source", escapeHtml(plate.source || "—")],
    ["Time", escapeHtml(plate.time || "—")],
  ];
  if (plate.date) rows.splice(7, 0, ["Date", escapeHtml(plate.date)]);
  if (plate.rto_name) rows.splice(4, 0, ["RTO Office", escapeHtml(plate.rto_name)]);
  if (plate.ocr_confidence !== null && plate.ocr_confidence !== undefined) {
    rows.push(["OCR Confidence", escapeHtml(confidencePercent(plate.ocr_confidence))]);
  }
  if (plate.raw_text && plate.raw_text !== vehicle) {
    rows.push(["Raw OCR", escapeHtml(plate.raw_text)]);
  }
  let html = `<div class="detection-card">${rows
    .map(([label, value]) => `<span class="label">${label}</span><span>${value}</span>`)
    .join("")}</div>`;
  if (plate.save_error) {
    html += `<p class="detection-note">Failed to save: ${escapeHtml(plate.save_error)}</p>`;
  }
  const recordKey = `d${++detectionRecordSeq}`;
  detectionRecords.set(recordKey, plate);
  html += `<div class="detection-actions">
    <button type="button" class="btn btn-download" data-detection-key="${recordKey}">Download Details</button>
  </div>`;
  return html;
}

document.addEventListener("click", (event) => {
  const button = event.target.closest(".btn-download[data-detection-key]");
  if (!button) return;
  const plate = detectionRecords.get(button.dataset.detectionKey);
  if (!plate) return;
  // A saved record has an id: ask the server for the same CSV the history
  // download uses, so the file matches what's stored. An unsaved one (a
  // deduped live-feed repeat) has no id, so it's built here instead.
  if (plate.id) {
    window.location.href = `/api/plates/${plate.id}/download`;
    return;
  }
  const vehicle = (plate.plate_text || plate.text || "vehicle").replace(/[^A-Za-z0-9]/g, "") || "vehicle";
  triggerCsvDownload(`${vehicle}.csv`, buildDetailCsv(plate));
});

function showDetections(target, plates, annotated) {
  let html = "";
  if (annotated) {
    html += `<img src="data:image/jpeg;base64,${annotated}" alt="Annotated result">`;
  }
  if (plates && plates.length) {
    html += plates.map(renderDetection).join("");
  } else if (!annotated) {
    html += `<p style="color:var(--text-dim)">No plates detected.</p>`;
  } else {
    html += `<p style="color:var(--text-dim); margin-top:10px;">No plates detected in this file.</p>`;
  }
  target.innerHTML = html;
}

// ---------- Live feed ----------

function setLiveChrome(active, hint) {
  feedRunning = active;
  startBtn.disabled = active;
  stopBtn.disabled = !active;
  feedHint.textContent = hint;
  cameraStatus.classList.toggle("live", active);
  cameraStatusText.textContent = active ? "live" : "camera idle";
}

function showCameraError(message) {
  stopBrowserCamera(false);
  feedImg.src = "";
  feedImg.style.display = "none";
  browserVideo.style.display = "none";
  feedPlaceholder.style.display = "block";
  feedPlaceholder.classList.add("error");
  feedPlaceholder.innerHTML = `<p>${escapeHtml(message)}</p>`;
  setLiveChrome(false, "camera error");
}

function resetPlaceholder() {
  feedPlaceholder.classList.remove("error");
  feedPlaceholder.innerHTML = idlePlaceholder;
  feedPlaceholder.style.display = "block";
}

async function rememberLogBaseline() {
  try {
    const response = await fetch("/api/plates?limit=1");
    const rows = await response.json();
    liveSinceId = Array.isArray(rows) && rows[0] ? rows[0].id : 0;
  } catch (err) {
    console.error("Could not read the current log:", err);
    liveSinceId = 0;
  }
  liveResult.innerHTML = "";
}

async function startFeed() {
  const camIndex = cameraIndexInput.value || 0;
  cameraStatusText.textContent = "opening camera...";
  feedHint.textContent = "checking camera";
  await rememberLogBaseline();
  try {
    const response = await fetch(`/api/camera/check?camera=${camIndex}`);
    const data = await response.json().catch(() => ({}));
    if (response.ok && data.ok) {
      startServerFeed(camIndex);
      return;
    }
  } catch (err) {
    console.error("Camera check failed:", err);
  }
  // The server process could not open a webcam. Fall back to the browser
  // camera, which is the device the page is actually allowed to use.
  const started = await startBrowserCamera();
  if (!started) {
    showCameraError(CAMERA_ERROR);
  }
}

function startServerFeed(camIndex) {
  stopBrowserCamera(false);
  feedImg.src = `/api/video_feed?camera=${camIndex}&t=${Date.now()}`;
  feedImg.style.display = "block";
  browserVideo.style.display = "none";
  feedPlaceholder.style.display = "none";
  setLiveChrome(true, `camera ${camIndex}`);
}

async function startBrowserCamera() {
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    return false;
  }
  try {
    browserStream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: "environment", width: { ideal: 1280 }, height: { ideal: 720 } },
      audio: false,
    });
  } catch (err) {
    console.error("Browser camera failed:", err);
    browserStream = null;
    return false;
  }
  feedImg.style.display = "none";
  feedImg.src = "";
  browserVideo.srcObject = browserStream;
  browserVideo.style.display = "block";
  feedPlaceholder.style.display = "none";
  setLiveChrome(true, "browser camera");
  browserTimer = setInterval(captureBrowserFrame, 1200);
  return true;
}

function captureBrowserFrame() {
  if (!feedRunning || !browserVideo.videoWidth) return;
  if (browserRequest) return;
  const canvas = document.createElement("canvas");
  canvas.width = browserVideo.videoWidth;
  canvas.height = browserVideo.videoHeight;
  canvas.getContext("2d").drawImage(browserVideo, 0, 0);
  browserRequest += 1;
  const requestId = browserRequest;
  canvas.toBlob((blob) => {
    if (!blob || !feedRunning) {
      browserRequest = 0;
      return;
    }
    const formData = new FormData();
    formData.append("file", blob, "frame.jpg");
    fetch("/api/detect?source=webcam", { method: "POST", body: formData })
      .then(async (response) => {
        const data = await response.json().catch(() => ({}));
        if (!response.ok) throw new Error(data.detail || `Request failed (${response.status})`);
        return data;
      })
      .then((data) => {
        if (requestId !== browserRequest || !feedRunning) return;
        if (data.plates && data.plates.length) {
          showDetections(liveResult, data.plates, null);
          refreshLog();
        }
      })
      .catch((err) => console.error("Browser frame detection failed:", err))
      .finally(() => {
        if (requestId === browserRequest) browserRequest = 0;
      });
  }, "image/jpeg", 0.85);
}

function stopBrowserCamera(updateUi) {
  if (browserTimer) {
    clearInterval(browserTimer);
    browserTimer = null;
  }
  browserRequest = 0;
  if (browserStream) {
    browserStream.getTracks().forEach((track) => track.stop());
    browserStream = null;
  }
  browserVideo.srcObject = null;
  if (updateUi) {
    browserVideo.style.display = "none";
  }
}

function stopFeed() {
  feedRunning = false;
  fetch("/api/camera/stop", { method: "POST" }).catch((err) => console.error("Camera stop failed:", err));
  stopBrowserCamera(false);
  feedImg.src = "";
  feedImg.style.display = "none";
  browserVideo.style.display = "none";
  resetPlaceholder();
  setLiveChrome(false, "not running");
}

startBtn.addEventListener("click", startFeed);
stopBtn.addEventListener("click", stopFeed);
window.addEventListener("beforeunload", () => {
  if (feedRunning) navigator.sendBeacon("/api/camera/stop");
  stopBrowserCamera(false);
});

feedImg.addEventListener("error", () => {
  if (!feedRunning || browserStream) return;
  showCameraError(CAMERA_ERROR);
});

// ---------- Upload / single-image or video test ----------

dropzone.addEventListener("click", (event) => {
  if (event.target === browseBtn) return;
  fileInput.click();
});

fileInput.addEventListener("change", () => {
  handleFile(fileInput.files[0]);
});

dropzone.addEventListener("dragover", (event) => {
  event.preventDefault();
  dropzone.classList.add("dragover");
});

dropzone.addEventListener("dragleave", () => {
  dropzone.classList.remove("dragover");
});

dropzone.addEventListener("drop", (event) => {
  event.preventDefault();
  dropzone.classList.remove("dragover");
  const file = event.dataTransfer.files && event.dataTransfer.files[0];
  handleFile(file);
});

function isVideoFile(file) {
  if (!file) return false;
  if (file.type.startsWith("video/")) return true;
  return /\.(mp4|avi|mov|mkv|webm|m4v|wmv)$/i.test(file.name || "");
}

function handleFile(file) {
  if (!file) return;
  uploadResult.innerHTML = `<p style="color:var(--text-dim)">Processing...</p>`;
  const formData = new FormData();
  formData.append("file", file);
  const endpoint = isVideoFile(file) ? "/api/detect_video" : "/api/detect";

  fetch(endpoint, { method: "POST", body: formData })
    .then(async (response) => {
      const data = await response.json().catch(() => ({}));
      if (!response.ok) {
        const detail = data.detail || `Request failed (${response.status})`;
        throw new Error(typeof detail === "string" ? detail : "Request failed");
      }
      return data;
    })
    .then((data) => {
      let html = "";
      if (data.annotated_image_base64) {
        html += `<img src="data:image/jpeg;base64,${data.annotated_image_base64}" alt="Annotated result">`;
      }
      if (data.plates && data.plates.length > 0) {
        html += data.plates.map(renderDetection).join("");
        
      } else {
        html += `<p style="color:var(--text-dim); margin-top:10px;">No plates detected in this file.</p>`;
      }
      html += `<button class="btn" id="resetUploadBtn" style="margin-top:12px; width:100%;">Try another file</button>`;
      uploadResult.innerHTML = html;
      document.getElementById("resetUploadBtn").addEventListener("click", () => {
        uploadResult.innerHTML = "";
        fileInput.value = "";
      });
      refreshLog();
    })
    .catch((err) => {
      console.error(err);
      uploadResult.innerHTML = `<p style="color:var(--red)">${escapeHtml(err.message || err)}</p>`;
    });
}

// ---------- Log table (polled) ----------

function confClass(conf) {
  if (conf === null || conf === undefined) return "";
  if (conf >= 0.75) return "conf-good";
  if (conf >= 0.5) return "conf-mid";
  return "conf-low";
}

function refreshLog() {
  fetch("/api/plates?limit=30")
    .then(async (response) => {
      const rows = await response.json().catch(() => null);
      if (!response.ok || !Array.isArray(rows)) {
        throw new Error("Log request failed");
      }
      return rows;
    })
    .then((rows) => {
      logCount.textContent = `${rows.length} logged`;
      if (rows.length === 0) {
        logTableBody.innerHTML = `<tr class="empty-row"><td colspan="10">No plates logged yet.</td></tr>`;
        return;
      }
      if (feedRunning && rows[0] && rows[0].id > liveSinceId) {
        showDetections(liveResult, [rows[0]], null);
      }
      logTableBody.innerHTML = rows
        .map((row) => {
          const confPct = confidencePercent(row.confidence);
          return `<tr data-id="${row.id}">
            <td class="time-cell">${escapeHtml(row.date || "—")}</td>
            <td class="time-cell">${escapeHtml(row.time || "—")}</td>
            <td class="plate-text">${escapeHtml(row.plate_text || "—")}</td>
            <td class="district-cell">${escapeHtml(row.district || "Unknown RTO/District")}</td>
            <td>${colorBadge(row.plate_color)}</td>
            <td class="type-cell">${escapeHtml(row.vehicle_type || "Unknown")}</td>
            <td class="${confClass(row.confidence)}">${escapeHtml(confPct)}</td>
            <td>${escapeHtml(row.source || "—")}</td>
            <td>${statusBadge(row.status)}</td>
            <td><button class="delete-btn" data-id="${row.id}" title="Delete this log entry">&times;</button></td>
          </tr>`;
        })
        .join("");
    })
    .catch((err) => {
      console.error("Failed to refresh logs:", err);
      logCount.textContent = "log refresh failed";
    });
}

logTableBody.addEventListener("click", (event) => {
  const button = event.target.closest(".delete-btn");
  if (!button) return;
  const plateId = button.dataset.id;
  button.disabled = true;
  fetch(`/api/plates/${plateId}`, { method: "DELETE" })
    .then((response) => {
      if (!response.ok) throw new Error(`Delete failed (${response.status})`);
      return response.json();
    })
    .then(() => refreshLog())
    .catch((err) => {
      button.disabled = false;
      console.error(err);
    });
});

refreshLog();
setInterval(refreshLog, 3000);
