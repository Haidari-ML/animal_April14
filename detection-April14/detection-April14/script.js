const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const statusBox = document.getElementById("status");

let session;

// ---------------- CLASSES (your model) ----------------
const classes = [
  "Zebra", "Lion", "Cheetah", "Tiger", "Bear",
  "Elephant", "Giraffe", "Deer", "Hippopotamus", "Rhinoceros"
];

// ---------------- CAMERA ----------------
async function startCamera() {
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { facingMode: "environment" }
  });

  video.srcObject = stream;

  return new Promise(res => {
    video.onloadedmetadata = () => {
      video.play();
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      res();
    };
  });
}

// ---------------- MODEL ----------------
async function loadModel() {
  session = await ort.InferenceSession.create("./best_10.onnx");
  statusBox.innerText = "Model loaded ✔";
}

// ---------------- PREPROCESS ----------------
function preprocess() {
  const size = 640;

  const off = document.createElement("canvas");
  off.width = size;
  off.height = size;

  const octx = off.getContext("2d");
  octx.drawImage(video, 0, 0, size, size);

  const data = octx.getImageData(0, 0, size, size).data;

  const input = new Float32Array(3 * size * size);

  let r = 0, g = size * size, b = 2 * size * size;

  for (let i = 0; i < data.length; i += 4) {
    input[r++] = data[i] / 255;
    input[g++] = data[i + 1] / 255;
    input[b++] = data[i + 2] / 255;
  }

  return input;
}

// ---------------- MATH ----------------
function sigmoid(x) {
  return 1 / (1 + Math.exp(-x));
}

function iou(a, b) {
  const x1 = Math.max(a.x, b.x);
  const y1 = Math.max(a.y, b.y);
  const x2 = Math.min(a.x + a.w, b.x + b.w);
  const y2 = Math.min(a.y + a.h, b.y + b.h);

  const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  const union = a.w * a.h + b.w * b.h - inter;

  return inter / union;
}

function nms(boxes, thresh = 0.45) {
  boxes.sort((a, b) => b.score - a.score);

  const out = [];

  for (const b of boxes) {
    let keep = true;

    for (const o of out) {
      if (iou(b, o) > thresh) {
        keep = false;
        break;
      }
    }

    if (keep) out.push(b);

    if (out.length > 15) break; // safety cap
  }

  return out;
}

// ---------------- YOLO11 CORRECT DECODER ----------------
function parseOutput(data) {

  const boxes = [];
  const numClasses = 10; // YOUR MODEL
  const threshold = 0.5;

  // YOLOv8/YOLO11 ONNX: [cx,cy,w,h,obj,cls...]
  const stride = 5 + numClasses;

  const numPred = Math.floor(data.length / stride);

  for (let i = 0; i < numPred; i++) {

    const offset = i * stride;

    let cx = data[offset];
    let cy = data[offset + 1];
    let w  = data[offset + 2];
    let h  = data[offset + 3];

    const obj = sigmoid(data[offset + 4]);

    if (obj < 0.4) continue;

    let bestClass = 0;
    let bestScore = 0;

    for (let c = 0; c < numClasses; c++) {
      const clsScore = sigmoid(data[offset + 5 + c]);

      if (clsScore > bestScore) {
        bestScore = clsScore;
        bestClass = c;
      }
    }

    const score = obj * bestScore;

    if (score > threshold) {

      // FILTER INVALID BOXES (CRITICAL FIX)
      if (
        !isFinite(cx) || !isFinite(cy) ||
        !isFinite(w) || !isFinite(h)
      ) continue;

      if (w <= 0 || h <= 0 || w > 640 || h > 640) continue;

      boxes.push({
        x: cx,
        y: cy,
        w: w,
        h: h,
        score,
        cls: bestClass
      });
    }
  }

  return nms(boxes);
}

// ---------------- DETECT ----------------
async function detect() {
  if (!session) return [];

  const tensor = new ort.Tensor(
    "float32",
    preprocess(),
    [1, 3, 640, 640]
  );

  const inputName = session.inputNames[0];

  const results = await session.run({
    [inputName]: tensor
  });

  const output = results[session.outputNames[0]];

  return parseOutput(output.data);
}

// ---------------- DRAW ----------------
function draw(boxes) {

  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

  const sx = canvas.width / 640;
  const sy = canvas.height / 640;

  ctx.lineWidth = 2;
  ctx.font = "16px Arial";

  for (const b of boxes) {

    const x = (b.x - b.w / 2) * sx;
    const y = (b.y - b.h / 2) * sy;
    const w = b.w * sx;
    const h = b.h * sy;

    const name = classes[b.cls] || "Unknown";

    ctx.strokeStyle = "lime";
    ctx.strokeRect(x, y, w, h);

    ctx.fillStyle = "lime";
    ctx.fillText(
      `${name} ${(b.score * 100).toFixed(1)}%`,
      x,
      y - 5
    );
  }
}

// ---------------- LOOP ----------------
async function loop() {
  const boxes = await detect();

  if (boxes.length > 0) draw(boxes);
  else ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

  requestAnimationFrame(loop);
}

// ---------------- START ----------------
async function main() {
  await startCamera();
  await loadModel();

  statusBox.innerText = "Running ✔";

  loop();
}

main();