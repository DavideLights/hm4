// HM4: Stochastic Penetration Process (Bernoulli & Rademacher, LLN & CLT)

// Seeded PRNG (Mulberry32)
let baseSeed = 2026;
function mulberry32(seed) {
  let a = seed >>> 0;
  return function () {
    a |= 0;
    a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// Standard Normal CDF Phi(z) via Abramowitz & Stegun approximation
function normalCDF(z) {
  const sign = z < 0 ? -1 : 1;
  const x = Math.abs(z) / Math.sqrt(2);
  const t = 1 / (1 + 0.3275911 * x);
  const a1 = 0.254829592, a2 = -0.284496736, a3 = 1.421413741;
  const a4 = -1.453152027, a5 = 1.061405429;
  const erf = 1 - (((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t) * Math.exp(-x * x);
  return 0.5 * (1 + sign * erf);
}

// Kolmogorov-Smirnov distance against N(mean, sd^2)
function ksDistanceNormal(arr, mean, sd) {
  const sorted = Float64Array.from(arr).sort();
  const m = sorted.length;
  let maxD = 0;
  for (let i = 0; i < m; i++) {
    const z = (sorted[i] - mean) / sd;
    const phi = normalCDF(z);
    const d1 = Math.abs((i + 1) / m - phi);
    const d2 = Math.abs(i / m - phi);
    if (d1 > maxD) maxD = d1;
    if (d2 > maxD) maxD = d2;
  }
  return maxD;
}

// DOM Elements
const modelSelect = document.getElementById("modelSelect");
const pInput = document.getElementById("pInput");
const mSelect = document.getElementById("mSelect");
const nMaxSelect = document.getElementById("nMaxSelect");
const btnSimulate = document.getElementById("btnSimulate");
const btnNewSeed = document.getElementById("btnNewSeed");
const qtyTabs = document.getElementById("qtyTabs");
const pathInfoText = document.getElementById("pathInfoText");

const pathCanvas = document.getElementById("pathCanvas");
const distTotalCanvas = document.getElementById("distTotalCanvas");
const distAvgCanvas = document.getElementById("distAvgCanvas");
const distNormCanvas = document.getElementById("distNormCanvas");
const distNormCaption = document.getElementById("distNormCaption");

const multiNBody = document.getElementById("multiNBody");
const llnCanvas = document.getElementById("llnCanvas");
const varScaleCanvas = document.getElementById("varScaleCanvas");
const varScaleCaption = document.getElementById("varScaleCaption");

let currentQty = "total"; // "total" | "avg" | "norm"
let cachedPaths = [];
let cachedParams = null;

qtyTabs.addEventListener("click", (e) => {
  const btn = e.target.closest(".view-tab");
  if (!btn) return;
  qtyTabs.querySelectorAll(".view-tab").forEach((b) => b.classList.remove("active"));
  btn.classList.add("active");
  currentQty = btn.dataset.qty;
  drawPaths(cachedPaths, cachedParams);
});

function getModelParams() {
  const isBern = modelSelect.value === "bernoulli";
  const p = Math.min(0.95, Math.max(0.05, parseFloat(pInput.value) || 0.3));
  const m = parseInt(mSelect.value, 10);
  const nMax = parseInt(nMaxSelect.value, 10);

  const mu = isBern ? p : 0;
  const sigma2 = isBern ? p * (1 - p) : 1;
  const sigma = Math.sqrt(sigma2);

  return { isBern, p, m, nMax, mu, sigma2, sigma };
}

function runSimulation() {
  const params = getModelParams();
  cachedParams = params;
  const { isBern, p, m, nMax, mu, sigma2, sigma } = params;

  pathInfoText.textContent = isBern
    ? `Model: Bernoulli(p = ${p.toFixed(2)}) | μ = ${mu.toFixed(2)}, σ² = ${sigma2.toFixed(4)}`
    : `Model: Rademacher (±1, P = 0.5) | μ = 0, σ² = 1.0000`;

  // Checkpoints of n for multi-n analysis
  const checkpoints = [1, 5, 10, 25, 50, 100, 250, 500, 1000, 2500].filter((c) => c <= nMax);
  if (!checkpoints.includes(nMax)) checkpoints.push(nMax);

  // Store S_n across all m attackers at each checkpoint
  const snapS = {};
  const snapRadS = {}; // Also keep Rademacher S_n for KS column
  for (const c of checkpoints) {
    snapS[c] = new Float64Array(m);
    snapRadS[c] = new Float64Array(m);
  }

  const rng = mulberry32(baseSeed);
  const rngRad = mulberry32(baseSeed + 999);
  const numPaths = 30;
  const paths = [];

  for (let j = 0; j < m; j++) {
    let s = 0;
    let sRad = 0;
    const savePath = j < numPaths;
    const traj = savePath ? new Float64Array(nMax) : null;
    let cpIdx = 0;

    for (let k = 1; k <= nMax; k++) {
      const u = rng();
      const x = isBern ? (u < p ? 1 : 0) : (u < 0.5 ? 1 : -1);
      s += x;

      const uR = rngRad();
      sRad += uR < 0.5 ? 1 : -1;

      if (savePath) traj[k - 1] = s;
      if (cpIdx < checkpoints.length && k === checkpoints[cpIdx]) {
        snapS[k][j] = s;
        snapRadS[k][j] = sRad;
        cpIdx++;
      }
    }
    if (savePath) paths.push(traj);
  }

  cachedPaths = paths;
  drawPaths(paths, params);

  // Distributions at horizon n = nMax
  const finalTotal = snapS[nMax];
  const finalAvg = new Float64Array(m);
  const finalNorm = new Float64Array(m);
  for (let j = 0; j < m; j++) {
    finalAvg[j] = finalTotal[j] / nMax;
    finalNorm[j] = finalTotal[j] / Math.sqrt(nMax);
  }

  drawDistPanel(
    distTotalCanvas,
    finalTotal,
    nMax * mu,
    Math.sqrt(nMax) * sigma,
    `S_n (n = ${nMax})`
  );
  drawDistPanel(
    distAvgCanvas,
    finalAvg,
    mu,
    sigma / Math.sqrt(nMax),
    `S_n / n (n = ${nMax})`
  );
  drawDistPanel(
    distNormCanvas,
    finalNorm,
    Math.sqrt(nMax) * mu,
    sigma,
    `S_n / √n (n = ${nMax})`
  );
  distNormCaption.textContent = isBern
    ? `Normalized S_n / √n (Mean p√n = ${(Math.sqrt(nMax) * mu).toFixed(2)}, SD = σ = ${sigma.toFixed(2)})`
    : `Normalized S_n / √n vs. Standard Normal N(0, 1)`;

  // Multi-n summary table, LLN tail probabilities, and variance scaling data
  const eps = 0.05;
  multiNBody.innerHTML = "";
  const llnPoints = [];
  const varPoints = [];

  for (const c of checkpoints) {
    const arr = snapS[c];
    const radArr = snapRadS[c];

    let sumS = 0;
    for (let j = 0; j < m; j++) sumS += arr[j];
    const meanS = sumS / m;

    let ssS = 0;
    let exceedCount = 0;
    for (let j = 0; j < m; j++) {
      const d = arr[j] - meanS;
      ssS += d * d;
      const avgVal = arr[j] / c;
      if (Math.abs(avgVal - mu) > eps) exceedCount++;
    }

    const varS = ssS / (m - 1);
    const meanAvg = meanS / c;
    const varAvg = varS / (c * c);
    const varNorm = varS / c;

    const tailProb = exceedCount / m;
    const chebBound = Math.min(1.0, sigma2 / (c * eps * eps));

    // Rademacher S_n / sqrt(n) KS distance against N(0, 1)
    const radNorm = new Float64Array(m);
    for (let j = 0; j < m; j++) radNorm[j] = radArr[j] / Math.sqrt(c);
    const ksD = ksDistanceNormal(radNorm, 0, 1);

    llnPoints.push({ n: c, empTail: tailProb, cheb: chebBound });
    varPoints.push({
      n: c,
      varS,
      varAvg,
      varNorm,
      theoS: c * sigma2,
      theoAvg: sigma2 / c,
      theoNorm: sigma2
    });

    const tr = document.createElement("tr");
    if (c === nMax) tr.style.backgroundColor = "#EEF4FF";
    tr.innerHTML = `
      <td><strong>n = ${c}</strong></td>
      <td>${meanS.toFixed(2)}</td>
      <td>${varS.toFixed(2)} <span style="color:#555555;">[${(c * sigma2).toFixed(1)}]</span></td>
      <td class="val-highlight">${meanAvg.toFixed(4)}</td>
      <td>${varAvg.toFixed(5)}</td>
      <td class="val-highlight">${tailProb.toFixed(4)}</td>
      <td>${chebBound.toFixed(4)}</td>
      <td>${varNorm.toFixed(4)}</td>
      <td>${ksD.toFixed(4)}</td>
    `;
    multiNBody.appendChild(tr);
  }

  drawLLNPlot(llnPoints, eps);
  drawVarScalingPlot(varPoints);
}

function drawPaths(paths, params) {
  const ctx = pathCanvas.getContext("2d");
  const w = pathCanvas.width;
  const h = pathCanvas.height;
  ctx.fillStyle = "#FFFFFF";
  ctx.fillRect(0, 0, w, h);

  if (!paths || paths.length === 0) return;
  const { nMax, mu, sigma } = params;

  const padL = 52, padR = 20, padT = 22, padB = 28;
  const plotW = w - padL - padR;
  const plotH = h - padT - padB;

  const transform = (s, k) => {
    if (currentQty === "total") return s;
    if (currentQty === "avg") return s / k;
    return s / Math.sqrt(k);
  };

  const theoMeanAt = (k) => {
    if (currentQty === "total") return k * mu;
    if (currentQty === "avg") return mu;
    return Math.sqrt(k) * mu;
  };

  const theoSDAt = (k) => {
    if (currentQty === "total") return Math.sqrt(k) * sigma;
    if (currentQty === "avg") return sigma / Math.sqrt(k);
    return sigma;
  };

  let minY = Infinity, maxY = -Infinity;
  const stride = Math.max(1, Math.floor(nMax / 250));
  for (let k = 1; k <= nMax; k += stride) {
    const mVal = theoMeanAt(k);
    const sdVal = theoSDAt(k);
    minY = Math.min(minY, mVal - 2.6 * sdVal);
    maxY = Math.max(maxY, mVal + 2.6 * sdVal);
  }
  if (maxY - minY < 1e-6) {
    minY -= 1;
    maxY += 1;
  }

  const toX = (k) => padL + ((k - 1) / Math.max(1, nMax - 1)) * plotW;
  const toY = (v) => {
    const clamped = Math.max(minY, Math.min(maxY, v));
    return padT + plotH - ((clamped - minY) / (maxY - minY)) * plotH;
  };

  // Grid lines
  ctx.strokeStyle = "#EEEEEE";
  ctx.fillStyle = "#555555";
  ctx.font = "10px monospace";
  for (let i = 0; i <= 4; i++) {
    const v = minY + (i / 4) * (maxY - minY);
    const y = toY(v);
    ctx.beginPath();
    ctx.moveTo(padL, y);
    ctx.lineTo(w - padR, y);
    ctx.stroke();
    ctx.fillText(v.toFixed(currentQty === "avg" ? 2 : 1), 6, y + 3);
  }

  ctx.strokeStyle = "#CCCCCC";
  ctx.strokeRect(padL, padT, plotW, plotH);

  // Sample trajectories
  ctx.strokeStyle = "rgba(17, 17, 17, 0.30)";
  ctx.lineWidth = 1.1;
  for (const traj of paths) {
    ctx.beginPath();
    for (let k = 1; k <= nMax; k += stride) {
      const x = toX(k);
      const y = toY(transform(traj[k - 1], k));
      if (k === 1) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();
  }

  // Theoretical Mean & +/- 2 SD envelope in #0055FF
  ctx.strokeStyle = "#0055FF";
  ctx.lineWidth = 2;
  ctx.beginPath();
  for (let k = 1; k <= nMax; k += stride) {
    const x = toX(k);
    const y = toY(theoMeanAt(k));
    if (k === 1) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.stroke();

  ctx.lineWidth = 1.2;
  ctx.setLineDash([4, 4]);
  for (const sign of [-2, 2]) {
    ctx.beginPath();
    for (let k = 1; k <= nMax; k += stride) {
      const x = toX(k);
      const y = toY(theoMeanAt(k) + sign * theoSDAt(k));
      if (k === 1) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();
  }
  ctx.setLineDash([]);

  // X labels
  ctx.fillStyle = "#555555";
  for (let i = 0; i <= 5; i++) {
    const kVal = Math.max(1, Math.round((i / 5) * nMax));
    const x = toX(kVal);
    ctx.fillText(`k=${kVal}`, x - 14, h - 8);
  }
}

function drawDistPanel(canvas, data, theoMean, theoSD, title) {
  const ctx = canvas.getContext("2d");
  const w = canvas.width;
  const h = canvas.height;
  ctx.fillStyle = "#FFFFFF";
  ctx.fillRect(0, 0, w, h);

  const padL = 34, padR = 12, padT = 24, padB = 26;
  const plotW = w - padL - padR;
  const plotH = h - padT - padB;

  let minV = theoMean - 3.4 * theoSD;
  let maxV = theoMean + 3.4 * theoSD;
  for (let i = 0; i < data.length; i++) {
    if (data[i] < minV) minV = data[i];
    if (data[i] > maxV) maxV = data[i];
  }
  if (maxV - minV < 1e-6) {
    minV -= 1;
    maxV += 1;
  }

  const bins = 20;
  const binW = (maxV - minV) / bins;
  const counts = new Int32Array(bins);
  for (let i = 0; i < data.length; i++) {
    let b = Math.floor((data[i] - minV) / binW);
    if (b < 0) b = 0;
    if (b >= bins) b = bins - 1;
    counts[b]++;
  }

  let maxDens = 0;
  const densities = new Float64Array(bins);
  for (let b = 0; b < bins; b++) {
    densities[b] = counts[b] / (data.length * binW);
    if (densities[b] > maxDens) maxDens = densities[b];
  }
  const normalPeak = theoSD > 0 ? 1 / (theoSD * Math.sqrt(2 * Math.PI)) : 0;
  maxDens = Math.max(maxDens, normalPeak) * 1.15;

  ctx.strokeStyle = "#CCCCCC";
  ctx.strokeRect(padL, padT, plotW, plotH);

  ctx.fillStyle = "#111111";
  ctx.font = "bold 11px system-ui, sans-serif";
  ctx.fillText(title, padL, 15);

  const pxBinW = plotW / bins;
  ctx.fillStyle = "#DDDDDD";
  ctx.strokeStyle = "#111111";
  ctx.lineWidth = 1;
  for (let b = 0; b < bins; b++) {
    const bh = (densities[b] / maxDens) * plotH;
    const bx = padL + b * pxBinW;
    const by = padT + plotH - bh;
    ctx.fillRect(bx, by, pxBinW, bh);
    ctx.strokeRect(bx, by, pxBinW, bh);
  }

  // Normal density overlay in #0055FF
  if (theoSD > 0) {
    ctx.strokeStyle = "#0055FF";
    ctx.lineWidth = 1.8;
    ctx.beginPath();
    for (let px = 0; px <= plotW; px += 2) {
      const xVal = minV + (px / plotW) * (maxV - minV);
      const z = (xVal - theoMean) / theoSD;
      const dens = Math.exp(-0.5 * z * z) / (theoSD * Math.sqrt(2 * Math.PI));
      const py = padT + plotH - (dens / maxDens) * plotH;
      if (px === 0) ctx.moveTo(padL + px, py);
      else ctx.lineTo(padL + px, py);
    }
    ctx.stroke();
  }

  // X ticks
  ctx.fillStyle = "#555555";
  ctx.font = "9px monospace";
  for (let i = 0; i <= 2; i++) {
    const v = minV + (i / 2) * (maxV - minV);
    const x = padL + (i / 2) * plotW;
    ctx.fillText(v.toFixed(2), x - 12, h - 8);
  }
}

function drawLLNPlot(pts, eps) {
  const ctx = llnCanvas.getContext("2d");
  const w = llnCanvas.width;
  const h = llnCanvas.height;
  ctx.fillStyle = "#FFFFFF";
  ctx.fillRect(0, 0, w, h);

  const padL = 44, padR = 16, padT = 24, padB = 28;
  const plotW = w - padL - padR;
  const plotH = h - padT - padB;

  const logX = pts.map((p) => Math.log10(p.n));
  const minX = Math.min(...logX);
  const maxX = Math.max(...logX);

  const toX = (lx) => padL + ((lx - minX) / Math.max(1e-6, maxX - minX)) * plotW;
  const toY = (prob) => padT + plotH - Math.max(0, Math.min(1, prob)) * plotH;

  ctx.strokeStyle = "#EEEEEE";
  ctx.fillStyle = "#555555";
  ctx.font = "10px monospace";
  for (let i = 0; i <= 4; i++) {
    const pVal = i / 4;
    const y = toY(pVal);
    ctx.beginPath();
    ctx.moveTo(padL, y);
    ctx.lineTo(w - padR, y);
    ctx.stroke();
    ctx.fillText(pVal.toFixed(2), 6, y + 3);
  }

  ctx.strokeStyle = "#CCCCCC";
  ctx.strokeRect(padL, padT, plotW, plotH);

  // Chebyshev bound dashed line
  ctx.strokeStyle = "#555555";
  ctx.lineWidth = 1.5;
  ctx.setLineDash([4, 4]);
  ctx.beginPath();
  for (let i = 0; i < pts.length; i++) {
    const x = toX(logX[i]);
    const y = toY(pts[i].cheb);
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.stroke();
  ctx.setLineDash([]);

  // Empirical tail probability in #0055FF
  ctx.strokeStyle = "#0055FF";
  ctx.lineWidth = 2;
  ctx.beginPath();
  for (let i = 0; i < pts.length; i++) {
    const x = toX(logX[i]);
    const y = toY(pts[i].empTail);
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.stroke();

  ctx.fillStyle = "#0055FF";
  for (let i = 0; i < pts.length; i++) {
    const x = toX(logX[i]);
    const y = toY(pts[i].empTail);
    ctx.fillRect(x - 3, y - 3, 6, 6);
  }

  // X labels
  ctx.fillStyle = "#555555";
  for (const p of pts) {
    if ([1, 10, 100, 1000].includes(p.n)) {
      const x = toX(Math.log10(p.n));
      ctx.fillText(`n=${p.n}`, x - 12, h - 8);
    }
  }
}

function drawVarScalingPlot(pts) {
  const ctx = varScaleCanvas.getContext("2d");
  const w = varScaleCanvas.width;
  const h = varScaleCanvas.height;
  ctx.fillStyle = "#FFFFFF";
  ctx.fillRect(0, 0, w, h);

  const padL = 48, padR = 18, padT = 24, padB = 28;
  const plotW = w - padL - padR;
  const plotH = h - padT - padB;

  const logX = pts.map((p) => Math.log10(p.n));
  const allLogY = [];
  for (const p of pts) {
    allLogY.push(Math.log10(Math.max(1e-8, p.varS)));
    allLogY.push(Math.log10(Math.max(1e-8, p.varAvg)));
    allLogY.push(Math.log10(Math.max(1e-8, p.varNorm)));
  }

  const minX = Math.min(...logX);
  const maxX = Math.max(...logX);
  const minY = Math.min(...allLogY) - 0.3;
  const maxY = Math.max(...allLogY) + 0.3;

  const toX = (lx) => padL + ((lx - minX) / Math.max(1e-6, maxX - minX)) * plotW;
  const toY = (ly) => padT + plotH - ((ly - minY) / (maxY - minY)) * plotH;

  ctx.strokeStyle = "#CCCCCC";
  ctx.strokeRect(padL, padT, plotW, plotH);

  // Plot helper
  function plotSeries(getVal, color, dashed) {
    ctx.strokeStyle = color;
    ctx.lineWidth = dashed ? 1.4 : 2;
    if (dashed) ctx.setLineDash([4, 3]);
    ctx.beginPath();
    for (let i = 0; i < pts.length; i++) {
      const x = toX(logX[i]);
      const y = toY(Math.log10(Math.max(1e-8, getVal(pts[i]))));
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();
    ctx.setLineDash([]);
  }

  // S_n (+1 slope), S_n/sqrt(n) (0 slope), S_n/n (-1 slope)
  plotSeries((p) => p.varS, "#111111", false);
  plotSeries((p) => p.varNorm, "#0055FF", false);
  plotSeries((p) => p.varAvg, "#666666", true);

  // Points
  for (let i = 0; i < pts.length; i++) {
    const x = toX(logX[i]);
    ctx.fillStyle = "#111111";
    ctx.fillRect(x - 2.5, toY(Math.log10(pts[i].varS)) - 2.5, 5, 5);
    ctx.fillStyle = "#0055FF";
    ctx.fillRect(x - 2.5, toY(Math.log10(pts[i].varNorm)) - 2.5, 5, 5);
    ctx.fillStyle = "#666666";
    ctx.fillRect(x - 2.5, toY(Math.log10(pts[i].varAvg)) - 2.5, 5, 5);
  }

  // Labels inside plot
  ctx.font = "bold 10px monospace";
  ctx.fillStyle = "#111111";
  ctx.fillText("Var(S_n) ∝ n¹", padL + 8, padT + 14);
  ctx.fillStyle = "#0055FF";
  ctx.fillText("Var(S_n/√n) = σ² (const)", padL + 8, padT + 28);
  ctx.fillStyle = "#666666";
  ctx.fillText("Var(S_n/n) ∝ n⁻¹", padL + 8, padT + 42);

  // Axis ticks
  ctx.fillStyle = "#555555";
  ctx.font = "10px monospace";
  for (const p of pts) {
    if ([1, 10, 100, 1000].includes(p.n)) {
      const x = toX(Math.log10(p.n));
      ctx.fillText(`n=${p.n}`, x - 12, h - 8);
    }
  }
  for (let i = 0; i <= 3; i++) {
    const ly = minY + (i / 3) * (maxY - minY);
    const y = toY(ly);
    ctx.fillText(`10^${ly.toFixed(1)}`, 2, y + 3);
  }
}

// Event Listeners
btnSimulate.addEventListener("click", runSimulation);
modelSelect.addEventListener("change", runSimulation);
pInput.addEventListener("change", runSimulation);
mSelect.addEventListener("change", runSimulation);
nMaxSelect.addEventListener("change", runSimulation);

btnNewSeed.addEventListener("click", () => {
  baseSeed = (baseSeed + 104729) >>> 0;
  runSimulation();
});

// Initialize on load
runSimulation();
