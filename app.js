/* Aviation Analytics & Defect Detection Dashboard Logic */

// Tab Navigation
function switchTab(tabId) {
  document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
  document.querySelectorAll('.nav-btn').forEach(btn => btn.classList.remove('active'));
  
  const targetTab = document.getElementById(tabId);
  if (targetTab) targetTab.classList.add('active');
  
  // Highlight nav button
  const activeBtn = Array.from(document.querySelectorAll('.nav-btn')).find(b => b.getAttribute('onclick').includes(tabId));
  if (activeBtn) activeBtn.classList.add('active');

  if (tabId === 'eda-studio') {
    initCharts();
  }
}

// Sample Inspection Defect Data
const SAMPLES = {
  fuselage: {
    name: "Rivet Corrosion & Structural Crack",
    region: "Fuselage Upper Panel (Bay 4)",
    defects: [
      { type: "Rivet Corrosion", bbox: [120, 150, 90, 80], color: "#f59e0b" },
      { type: "Fuselage Crack", bbox: [320, 240, 140, 60], color: "#ef4444" }
    ],
    conf: "94.8%",
    sev: "Critical",
    yolo: "[x_center: 0.482, y_center: 0.315, width: 0.185, height: 0.142]",
    action: "Immediate non-destructive ultrasonic testing required before flight sign-off. High probability of stress corrosion cracking."
  },
  turbine: {
    name: "Turbine Blade Erosion & Thermal Oxidation",
    region: "High-Pressure Turbine Rotor 2",
    defects: [
      { type: "Turbine Blade Erosion", bbox: [200, 180, 160, 130], color: "#38bdf8" }
    ],
    conf: "96.2%",
    sev: "Moderate",
    yolo: "[x_center: 0.437, y_center: 0.382, width: 0.250, height: 0.203]",
    action: "Borescope inspection required within 50 flight hours. Perform thermal barrier coating re-application."
  },
  wing: {
    name: "Wing Surface Dent & Delamination",
    region: "Left Wing Composite Panel (Rib 12)",
    defects: [
      { type: "Wing Surface Dent", bbox: [150, 290, 110, 110], color: "#818cf8" },
      { type: "Composite Delamination", bbox: [380, 120, 130, 95], color: "#10b981" }
    ],
    conf: "92.5%",
    sev: "Moderate",
    yolo: "[x_center: 0.632, y_center: 0.261, width: 0.203, height: 0.148]",
    action: "Tap testing recommended to map internal debonding boundaries. Schedule composite patch repair."
  },
  engine: {
    name: "Engine Cowling Hairline Fractures",
    region: "Nacelle Intake Lip",
    defects: [
      { type: "Fuselage Crack", bbox: [220, 200, 200, 50], color: "#ef4444" }
    ],
    conf: "97.1%",
    sev: "Minor",
    yolo: "[x_center: 0.500, y_center: 0.351, width: 0.312, height: 0.078]",
    action: "Monitor crack propagation rate. Apply stop-drill holes at fracture tips if length exceeds 15mm."
  }
};

// Canvas Bounding Box Visualizer
function drawCanvas(sampleKey) {
  const canvas = document.getElementById('defectCanvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const sample = SAMPLES[sampleKey] || SAMPLES.fuselage;

  // Background Metallic Component Texture
  ctx.fillStyle = "#1e293b";
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  // Panel structural lines
  ctx.strokeStyle = "rgba(255, 255, 255, 0.15)";
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(0, 320); ctx.lineTo(640, 320);
  ctx.moveTo(320, 0); ctx.lineTo(320, 640);
  ctx.stroke();

  // Grid rivets
  for (let x = 40; x < 640; x += 80) {
    for (let y of [40, 600]) {
      ctx.fillStyle = "#475569";
      ctx.beginPath();
      ctx.arc(x, y, 6, 0, Math.PI * 2);
      ctx.fill();
    }
  }

  // Draw Bounding Boxes and Labels
  sample.defects.forEach(defect => {
    const [x, y, w, h] = defect.bbox;
    ctx.strokeStyle = defect.color;
    ctx.lineWidth = 3;
    ctx.strokeRect(x, y, w, h);

    // Fill semi-transparent box
    ctx.fillStyle = defect.color + "22";
    ctx.fillRect(x, y, w, h);

    // Label banner
    ctx.fillStyle = defect.color;
    ctx.fillRect(x, y - 24, ctx.measureText(defect.type).width + 70, 24);
    ctx.fillStyle = "#ffffff";
    ctx.font = "bold 12px Outfit, sans-serif";
    ctx.fillText(`${defect.type} (${sample.conf})`, x + 6, y - 8);
  });
}

function loadSampleDefect(sampleKey) {
  document.querySelectorAll('.thumb-btn').forEach(btn => btn.classList.remove('active'));
  const activeBtn = Array.from(document.querySelectorAll('.thumb-btn')).find(b => b.getAttribute('onclick').includes(sampleKey));
  if (activeBtn) activeBtn.classList.add('active');

  const s = SAMPLES[sampleKey];
  if (s) {
    document.getElementById('defectName').innerText = s.name;
    document.getElementById('compRegion').innerText = s.region;
    document.getElementById('defectConf').innerText = s.conf;
    document.getElementById('defectSev').innerText = s.sev;
    document.getElementById('bboxCode').innerText = s.yolo;
    document.getElementById('maintAction').innerText = s.action;
    drawCanvas(sampleKey);
  }
}

// Dynamic Flight Delay Inference Engine
function runDelayPrediction(e) {
  e.preventDefault();
  const weather = document.getElementById('weather').value;
  const traffic = parseFloat(document.getElementById('traffic').value) || 5.0;
  const priorDelay = parseInt(document.getElementById('priorDelay').value) || 0;
  const maint = parseInt(document.getElementById('maintOverdue').value) || 0;

  let weatherWeight = { "Clear": 2.0, "Rain": 15.0, "Fog": 25.0, "Thunderstorm": 45.0 }[weather] || 10.0;
  let estimatedDelay = Math.round((weatherWeight * 0.5) + (traffic * 3.8) + (priorDelay * 0.45) + (maint * 22.0));

  const delayElem = document.getElementById('delayMins');
  const badgeElem = document.getElementById('riskBadge');

  if (delayElem) delayElem.innerHTML = `${estimatedDelay} <span style="font-size: 1.2rem; color: var(--text-secondary);">Mins Delay</span>`;

  if (badgeElem) {
    if (estimatedDelay < 15) {
      badgeElem.className = "risk-badge risk-low";
      badgeElem.innerText = "Low Risk / On-Time";
    } else if (estimatedDelay <= 45) {
      badgeElem.className = "risk-badge risk-moderate";
      badgeElem.innerText = "Moderate Delay Risk";
    } else {
      badgeElem.className = "risk-badge risk-high";
      badgeElem.innerText = "Severe Delay Risk";
    }
  }
}

// Initialize Charts
let weatherChartInst = null;
let defectChartInst = null;

function initCharts() {
  if (weatherChartInst) return;

  const wCtx = document.getElementById('weatherChart')?.getContext('2d');
  if (wCtx) {
    weatherChartInst = new Chart(wCtx, {
      type: 'bar',
      data: {
        labels: ['Clear', 'Overcast', 'Rain', 'Fog', 'High Winds', 'Snow', 'Thunderstorm'],
        datasets: [{
          label: 'Average Delay (Minutes)',
          data: [4.2, 8.5, 18.3, 29.1, 34.0, 38.6, 52.4],
          backgroundColor: 'rgba(56, 189, 248, 0.7)',
          borderColor: '#38bdf8',
          borderWidth: 1
        }]
      },
      options: {
        responsive: true,
        plugins: { legend: { display: false } },
        scales: { y: { ticks: { color: '#94a3b8' } }, x: { ticks: { color: '#94a3b8' } } }
      }
    });
  }

  const dCtx = document.getElementById('defectChart')?.getContext('2d');
  if (dCtx) {
    defectChartInst = new Chart(dCtx, {
      type: 'doughnut',
      data: {
        labels: ['Rivet Corrosion', 'Fuselage Crack', 'Turbine Erosion', 'Wing Dent', 'Delamination'],
        datasets: [{
          data: [18, 12, 15, 9, 7],
          backgroundColor: ['#f59e0b', '#ef4444', '#38bdf8', '#818cf8', '#10b981']
        }]
      },
      options: {
        responsive: true,
        plugins: { legend: { labels: { color: '#f8fafc' } } }
      }
    });
  }
}

// Initial Window Load
window.addEventListener('DOMContentLoaded', () => {
  drawCanvas('fuselage');
});
