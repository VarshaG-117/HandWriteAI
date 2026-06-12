const canvas = document.getElementById("drawCanvas");
const ctx = canvas.getContext("2d");
const predictBtn = document.getElementById("predictBtn");
const clearBtn = document.getElementById("clearBtn");
const imageInput = document.getElementById("imageInput");
const pdfInput = document.getElementById("pdfInput");
const predictionChar = document.getElementById("predictionChar");
const confidenceText = document.getElementById("confidenceText");
const confidenceBar = document.getElementById("confidenceBar");
const top3 = document.getElementById("top3");
const message = document.getElementById("message");
const historyList = document.getElementById("historyList");
const historyClearBtn = document.getElementById("historyClearBtn");
const pdfText = document.getElementById("pdfText");
const imageFileName = document.getElementById("imageFileName");
const pdfFileName = document.getElementById("pdfFileName");

let drawing = false;
const history = [];

function setupCanvas() {
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.lineWidth = 20;
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  ctx.strokeStyle = "#101318";
}

function pointFromEvent(event) {
  const rect = canvas.getBoundingClientRect();
  const client = event.touches ? event.touches[0] : event;
  return {
    x: ((client.clientX - rect.left) / rect.width) * canvas.width,
    y: ((client.clientY - rect.top) / rect.height) * canvas.height,
  };
}

function startDrawing(event) {
  event.preventDefault();
  drawing = true;
  const p = pointFromEvent(event);
  ctx.beginPath();
  ctx.moveTo(p.x, p.y);
}

function draw(event) {
  if (!drawing) return;
  event.preventDefault();
  const p = pointFromEvent(event);
  ctx.lineTo(p.x, p.y);
  ctx.stroke();
}

function stopDrawing() {
  drawing = false;
}

function setMessage(text, isError = true) {
  message.textContent = text || "";
  message.style.color = isError ? "var(--danger)" : "var(--accent)";
}

function renderResult(result, source = "Canvas") {
  if (result.error) {
    setMessage(result.error);
    return;
  }
  predictionChar.textContent = result.prediction;
  confidenceText.textContent = `${result.confidence.toFixed(2)}% confidence`;
  confidenceBar.style.width = `${Math.min(result.confidence, 100)}%`;
  top3.innerHTML = result.top3
    .map(
      (item) => `
        <div class="top-row">
          <strong>${item.character}</strong>
          <div class="mini-bar"><span style="width: ${item.confidence}%"></span></div>
          <span>${item.confidence.toFixed(2)}%</span>
        </div>
      `,
    )
    .join("");
  setMessage(`${source} prediction complete.`, false);
  addHistory(source, result);
}

function addHistory(source, result) {
  history.unshift({ source, ...result, time: new Date().toLocaleTimeString() });
  history.splice(8);
  historyList.innerHTML = history
    .map((item) => `<li><strong>${item.prediction}</strong> from ${item.source} at ${item.time} (${item.confidence.toFixed(2)}%)</li>`)
    .join("");
}

async function postJson(url, body) {
  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  return response.json();
}

async function postFile(url, fieldName, file) {
  const body = new FormData();
  body.append(fieldName, file);
  const response = await fetch(url, { method: "POST", body });
  return response.json();
}

canvas.addEventListener("mousedown", startDrawing);
canvas.addEventListener("mousemove", draw);
window.addEventListener("mouseup", stopDrawing);
canvas.addEventListener("touchstart", startDrawing, { passive: false });
canvas.addEventListener("touchmove", draw, { passive: false });
window.addEventListener("touchend", stopDrawing);

clearBtn.addEventListener("click", () => {
  setupCanvas();
  predictionChar.textContent = "-";
  confidenceText.textContent = "Waiting for input";
  confidenceBar.style.width = "0%";
  top3.innerHTML = "";
  setMessage("");
});

predictBtn.addEventListener("click", async () => {
  setMessage("Analyzing canvas...", false);
  const result = await postJson("/predict", { image: canvas.toDataURL("image/png") });
  renderResult(result, "Canvas");
});

imageInput.addEventListener("change", async () => {
  const file = imageInput.files[0];
  if (!file) return;
  imageFileName.textContent = file.name;
  setMessage("Analyzing uploaded image...", false);
  const result = await postFile("/upload-image", "image", file);
  renderResult(result, "Image");
  imageInput.value = "";
  imageFileName.textContent = "Choose file";
});

pdfInput.addEventListener("change", async () => {
  const file = pdfInput.files[0];
  if (!file) return;
  pdfFileName.textContent = file.name;
  setMessage("Extracting PDF characters...", false);
  const result = await postFile("/upload-pdf", "pdf", file);
  if (result.error) {
    setMessage(result.error);
    pdfFileName.textContent = "Choose file";
    return;
  }
  pdfText.textContent = result.text || "-";
  setMessage(`PDF complete: ${result.characters.length} character predictions.`, false);
  result.characters.slice(0, 8).forEach((item) => addHistory(`PDF p${item.page}`, item));
  pdfInput.value = "";
  pdfFileName.textContent = "Choose file";
});

historyClearBtn.addEventListener("click", () => {
  history.splice(0);
  historyList.innerHTML = "";
});

setupCanvas();
