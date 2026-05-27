const canvas = document.getElementById("sourceCanvas");
const ctx = canvas.getContext("2d");
const resultImage = document.getElementById("resultImage");
const pointsCode = document.getElementById("pointsCode");

const points = JSON.parse(canvas.dataset.points); // points in image coordinates
const imageWidth = Number(canvas.dataset.imageWidth);
const imageHeight = Number(canvas.dataset.imageHeight);
const paddingRatio = Number(canvas.dataset.paddingRatio || 0.2);
const padX = imageWidth * paddingRatio;
const padY = imageHeight * paddingRatio;
const minX = -padX;
const maxX = imageWidth - 1 + padX;
const minY = -padY;
const maxY = imageHeight - 1 + padY;

canvas.width = Math.round(imageWidth + 2 * padX);
canvas.height = Math.round(imageHeight + 2 * padY);

const image = new Image();
image.src = "/image";

let activeCorner = -1;
const CORNER_RADIUS = 10;

function toCanvasCoords(event) {
  const rect = canvas.getBoundingClientRect();
  const scaleX = canvas.width / rect.width;
  const scaleY = canvas.height / rect.height;
  return {
    x: (event.clientX - rect.left) * scaleX,
    y: (event.clientY - rect.top) * scaleY,
  };
}

function findCorner(x, y) {
  for (let i = 0; i < points.length; i += 1) {
    const canvasX = points[i][0] + padX;
    const canvasY = points[i][1] + padY;
    const dx = x - canvasX;
    const dy = y - canvasY;
    if (dx * dx + dy * dy <= CORNER_RADIUS * CORNER_RADIUS * 2) {
      return i;
    }
  }
  return -1;
}

function drawScene() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = "#f3f4f6";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(image, padX, padY, imageWidth, imageHeight);

  ctx.strokeStyle = "#00d26a";
  ctx.lineWidth = 4;
  ctx.beginPath();
  ctx.moveTo(points[0][0] + padX, points[0][1] + padY);
  for (let i = 1; i < points.length; i += 1) {
    ctx.lineTo(points[i][0] + padX, points[i][1] + padY);
  }
  ctx.closePath();
  ctx.stroke();

  for (let i = 0; i < points.length; i += 1) {
    ctx.beginPath();
    ctx.fillStyle = i === activeCorner ? "#ff9800" : "#00d26a";
    ctx.arc(points[i][0] + padX, points[i][1] + padY, CORNER_RADIUS, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = "#ffffff";
    ctx.lineWidth = 2;
    ctx.stroke();
  }

  renderPythonPoints();
}

function renderPythonPoints() {
  const rows = points
    .map((p) => `    [${Math.round(p[0])}, ${Math.round(p[1])}],`)
    .join("\n");
  pointsCode.textContent = `skull_src = np.float32([\n${rows}\n])`;
}

async function updateWarp() {
  const response = await fetch("/warp", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ points }),
  });
  if (!response.ok) return;
  const data = await response.json();
  resultImage.src = data.image;
}

canvas.addEventListener("mousedown", (event) => {
  const { x, y } = toCanvasCoords(event);
  activeCorner = findCorner(x, y);
  drawScene();
});

canvas.addEventListener("mousemove", (event) => {
  if (activeCorner === -1) return;
  const { x: cx, y: cy } = toCanvasCoords(event);
  const x = cx - padX;
  const y = cy - padY;
  points[activeCorner][0] = Math.max(minX, Math.min(maxX, x));
  points[activeCorner][1] = Math.max(minY, Math.min(maxY, y));
  drawScene();
});

window.addEventListener("mouseup", async () => {
  if (activeCorner === -1) return;
  activeCorner = -1;
  drawScene();
  await updateWarp();
});

image.onload = async () => {
  drawScene();
  await updateWarp();
};
