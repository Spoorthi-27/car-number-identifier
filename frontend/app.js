const feedImg = document.getElementById("feedImg");
const feedPlaceholder = document.getElementById("feedPlaceholder");
const feedHint = document.getElementById("feedHint");
const cameraStatus = document.getElementById("cameraStatus");
const cameraStatusText = document.getElementById("cameraStatusText");
const cameraIndexInput = document.getElementById("cameraIndex");
const startBtn = document.getElementById("startBtn");
const stopBtn = document.getElementById("stopBtn");

const dropzone = document.getElementById("dropzone");
const fileInput = document.getElementById("fileInput");
const browseBtn = document.getElementById("browseBtn");
const uploadResult = document.getElementById("uploadResult");

const logTableBody = document.getElementById("logTableBody");
const logCount = document.getElementById("logCount");

let feedRunning = false;

// ---------- Live feed ----------

function startFeed() {
  const camIndex = cameraIndexInput.value || 0;
  feedImg.src = `/api/video_feed?camera=${camIndex}&t=${Date.now()}`;
  feedImg.style.display = "block";
  feedPlaceholder.style.display = "none";
  feedRunning = true;
  startBtn.disabled = true;
  stopBtn.disabled = false;
  feedHint.textContent = `camera ${camIndex}`;
  cameraStatus.classList.add("live");
  cameraStatusText.textContent = "live";
}

function stopFeed() {
  feedImg.src = "";
  feedImg.style.display = "none";
  feedPlaceholder.style.display = "block";
  feedRunning = false;
  startBtn.disabled = false;
  stopBtn.disabled = true;
  feedHint.textContent = "not running";
  cameraStatus.classList.remove("live");
  cameraStatusText.textContent = "camera idle";
}

startBtn.addEventListener("click", startFeed);
stopBtn.addEventListener("click", stopFeed);

feedImg.addEventListener("error", () => {
  if (feedRunning) {
    feedHint.textContent = "camera error - check index / permissions";
  }
});

// ---------- Upload / single-image test ----------
// browseBtn is now a <label for="fileInput">, so clicking it opens the file
// picker natively - no JS .click() call needed (and it works even when an
// extension blocks scripted file-dialog triggers).

dropzone.addEventListener("click", (e) => {
  if (e.target === browseBtn) return; // let the label's native behavior handle it
  fileInput.click();
});

fileInput.addEventListener("change", () => {
  handleFile(fileInput.files[0]);
});

dropzone.addEventListener("dragover", (e) => {
  e.preventDefault();
  dropzone.classList.add("dragover");
});

dropzone.addEventListener("dragleave", () => {
  dropzone.classList.remove("dragover");
});

dropzone.addEventListener("drop", (e) => {
  e.preventDefault();
  dropzone.classList.remove("dragover");
  const file = e.dataTransfer.files && e.dataTransfer.files[0];
  handleFile(file);
});

function handleFile(file) {
  if (!file) return;
  uploadResult.innerHTML = `<p style="color:var(--text-dim)">Processing...</p>`;

  const formData = new FormData();
  formData.append("file", file);

  fetch("/api/detect", { method: "POST", body: formData })
    .then((res) => res.json())
    .then((data) => {
      let html = "";
      if (data.annotated_image_base64) {
        html += `<img src="data:image/jpeg;base64,${data.annotated_image_base64}" alt="Annotated result">`;
      }
      if (data.plates && data.plates.length > 0) {
        html += `<div class="upload-plate-list">`;
        data.plates.forEach((p) => {
          const district = p.district ? ` · ${p.district}` : "";
          html += `<div class="upload-plate-row"><span>${p.text || "?"}${district}</span><span>${(p.detection_conf * 100).toFixed(0)}%</span></div>`;
        });
        html += `</div>`;
      } else {
        html += `<p style="color:var(--text-dim); margin-top:10px;">No plates detected in this image.</p>`;
      }
      html += `<button class="btn" id="resetUploadBtn" style="margin-top:12px; width:100%;">Try another image</button>`;
      uploadResult.innerHTML = html;

      document.getElementById("resetUploadBtn").addEventListener("click", () => {
        uploadResult.innerHTML = "";
        fileInput.value = "";
      });

      refreshLog();
    })
    .catch((err) => {
      uploadResult.innerHTML = `<p style="color:var(--red)">Error: ${err}</p>`;
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
    .then((res) => res.json())
    .then((rows) => {
      logCount.textContent = `${rows.length} logged`;
      if (rows.length === 0) {
        logTableBody.innerHTML = `<tr class="empty-row"><td colspan="5">No plates logged yet.</td></tr>`;
        return;
      }
      logTableBody.innerHTML = rows
        .map((r) => {
          const confPct = r.confidence !== null ? `${(r.confidence * 100).toFixed(0)}%` : "-";
          const time = r.timestamp ? r.timestamp.replace("T", " ") : "-";
          const district = r.district || "-";
          return `<tr data-id="${r.id}">
            <td class="plate-text">${r.plate_text}</td>
            <td class="district-cell">${district}</td>
            <td class="${confClass(r.confidence)}">${confPct}</td>
            <td class="time-cell">${time}</td>
            <td><button class="delete-btn" data-id="${r.id}" title="Delete this log entry">&times;</button></td>
          </tr>`;
        })
        .join("");
    })
    .catch(() => {
      /* silent - table just won't update this tick */
    });
}

// Event delegation: handles delete buttons even after refreshLog() re-renders the table.
logTableBody.addEventListener("click", (e) => {
  const btn = e.target.closest(".delete-btn");
  if (!btn) return;

  const plateId = btn.dataset.id;
  btn.disabled = true;

  fetch(`/api/plates/${plateId}`, { method: "DELETE" })
    .then((res) => {
      if (!res.ok) throw new Error(`Delete failed (${res.status})`);
      return res.json();
    })
    .then(() => refreshLog())
    .catch((err) => {
      btn.disabled = false;
      console.error(err);
    });
});

refreshLog();
setInterval(refreshLog, 3000);
