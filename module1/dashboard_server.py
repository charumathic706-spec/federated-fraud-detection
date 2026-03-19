#!/usr/bin/env python3
# =============================================================================
# dashboard_server.py
# Real-time FL training dashboard — HTTP server + WebSocket push
#
# HOW IT WORKS:
#   1. Reads trust_training_log.json every second as FL training runs
#   2. Serves the dashboard HTML on http://localhost:5000
#   3. Pushes live updates to the browser via Server-Sent Events (SSE)
#      — no WebSocket library needed, works in every browser natively
#
# HOW TO RUN:
#   Terminal 1: python -m split2.main --data_path ../data/creditcard.csv \
#                   --attack label_flip --malicious 1
#   Terminal 2: python dashboard_server.py --log logs_split2/trust_training_log.json
#   Browser:    http://localhost:5000
#
# FOR TWO-MACHINE SETUP:
#   Run dashboard_server.py on System 1 (same machine as server)
#   Open browser on EITHER machine: http://<System1_IP>:5000
#   System 2 attacker sees the dashboard too — you can show both
#   terminals AND the dashboard simultaneously during the demo
#
# DEPENDENCIES: none beyond Python standard library
#   (no Flask, no Django, no WebSockets — pure stdlib)
# =============================================================================

import argparse
import http.server
import json
import os
import sys
import threading
import time
from http import HTTPStatus
from urllib.parse import urlparse


# =============================================================================
# CONFIGURATION
# =============================================================================

DEFAULT_LOG  = "logs_split2/trust_training_log.json"
POLL_SECS    = 1.0      # how often to check log file for new rounds
NUM_CLIENTS  = 5


# =============================================================================
# LOG READER — watches the JSON log file written by split2/main.py
# =============================================================================

class LogWatcher:
    """
    Watches trust_training_log.json and extracts latest state.
    The FL training loop writes this file after every round.
    """
    def __init__(self, log_path: str):
        self.log_path  = log_path
        self.last_mtime = 0
        self.rounds     = []
        self.lock       = threading.Lock()

    def refresh(self):
        """Read log file if it changed since last read."""
        if not os.path.exists(self.log_path):
            return False
        mtime = os.path.getmtime(self.log_path)
        if mtime <= self.last_mtime:
            return False
        try:
            with open(self.log_path, "r") as f:
                data = json.load(f)
            with self.lock:
                self.rounds     = data if isinstance(data, list) else []
                self.last_mtime = mtime
            return True
        except (json.JSONDecodeError, IOError):
            return False

    def get_state(self) -> dict:
        """Return current dashboard state from latest rounds."""
        with self.lock:
            rounds = list(self.rounds)

        if not rounds:
            return {"status": "waiting", "rounds": [], "latest": None}

        latest  = rounds[-1]
        best_f1 = max((r.get("global_f1", 0) for r in rounds), default=0)
        best_auc = max((r.get("global_auc", 0) for r in rounds), default=0)

        # Build per-client trust trajectory
        trust_history = {str(i): [] for i in range(NUM_CLIENTS)}
        alpha_history = {str(i): [] for i in range(NUM_CLIENTS)}
        f1_history    = []
        auc_history   = []
        round_nums    = []

        for r in rounds:
            round_nums.append(r.get("round", 0))
            f1_history.append(round(r.get("global_f1", 0), 4))
            auc_history.append(round(r.get("global_auc", 0), 4))
            ts = r.get("trust_scores", {})
            as_ = r.get("anomaly_scores", {})
            for i in range(NUM_CLIENTS):
                trust_history[str(i)].append(
                    round(float(ts.get(str(i), ts.get(i, 1.0))), 4)
                )
                alpha_history[str(i)].append(
                    round(float(as_.get(str(i), as_.get(i, 0.0))), 4)
                )

        flagged   = latest.get("flagged_clients", [])
        trusted   = latest.get("trusted_clients", [])
        ts_latest = latest.get("trust_scores", {})
        as_latest = latest.get("anomaly_scores", {})

        # Detect who is attacking (lowest trust score)
        attacker = None
        if flagged:
            attacker = flagged[0]

        # Detection stats
        detected_rounds = sum(
            1 for r in rounds if r.get("flagged_clients")
        )

        return {
            "status":          "running" if len(rounds) < 25 else "complete",
            "current_round":   latest.get("round", 0),
            "total_rounds":    25,
            "global_f1":       round(latest.get("global_f1", 0), 4),
            "global_auc":      round(latest.get("global_auc", 0), 4),
            "best_f1":         round(best_f1, 4),
            "best_auc":        round(best_auc, 4),
            "flagged":         flagged,
            "trusted":         trusted,
            "attacker":        attacker,
            "detected_rounds": detected_rounds,
            "false_positives": sum(
                1 for r in rounds
                for c in r.get("flagged_clients", [])
                if c != attacker
            ),
            "trust_latest":    {
                str(i): round(float(ts_latest.get(str(i),
                              ts_latest.get(i, 1.0))), 4)
                for i in range(NUM_CLIENTS)
            },
            "alpha_latest":    {
                str(i): round(float(as_latest.get(str(i),
                              as_latest.get(i, 0.0))), 4)
                for i in range(NUM_CLIENTS)
            },
            "round_nums":      round_nums,
            "f1_history":      f1_history,
            "auc_history":     auc_history,
            "trust_history":   trust_history,
            "alpha_history":   alpha_history,
            "model_hash":      latest.get("model_hash", ""),
            "blockchain_round": latest.get("round", 0),
        }


# =============================================================================
# DASHBOARD HTML — complete single-file dashboard
# =============================================================================

DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>FL Trust Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<style>
  :root {
    --navy: #1E3A5F; --teal: #028090; --teal2: #02C39A;
    --amber: #F4A261; --red: #E24B4A; --green: #2ECC71;
    --bg: #0F1923; --card: #162233; --border: #1E3A5F;
    --text: #E2E8F0; --muted: #94A3B8;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: var(--bg); color: var(--text); font-family: system-ui, sans-serif; }

  .header {
    background: var(--card); border-bottom: 1px solid var(--border);
    padding: 14px 24px; display: flex; align-items: center;
    justify-content: space-between;
  }
  .header h1 { font-size: 16px; font-weight: 600; color: var(--teal2); }
  .header .sub { font-size: 12px; color: var(--muted); margin-top: 2px; }

  .status-dot { width: 8px; height: 8px; border-radius: 50%;
    display: inline-block; margin-right: 6px; }
  .status-dot.running  { background: var(--green); animation: pulse 1.5s infinite; }
  .status-dot.waiting  { background: var(--amber); }
  .status-dot.complete { background: var(--teal2); }
  @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.4} }

  .main { padding: 20px 24px; display: grid; gap: 16px; }

  /* Metric cards row */
  .metrics { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; }
  .mc { background: var(--card); border: 1px solid var(--border);
    border-radius: 10px; padding: 16px; }
  .mc .label { font-size: 11px; color: var(--muted); text-transform: uppercase;
    letter-spacing: .5px; margin-bottom: 6px; }
  .mc .value { font-size: 28px; font-weight: 700; color: var(--teal2); }
  .mc .sub   { font-size: 11px; color: var(--muted); margin-top: 4px; }

  /* Round progress bar */
  .progress-wrap { background: var(--card); border: 1px solid var(--border);
    border-radius: 10px; padding: 16px; }
  .progress-label { display: flex; justify-content: space-between;
    font-size: 13px; margin-bottom: 8px; }
  .progress-bar-bg { height: 8px; background: var(--border); border-radius: 4px; }
  .progress-bar { height: 8px; border-radius: 4px;
    background: linear-gradient(90deg, var(--teal), var(--teal2));
    transition: width .5s ease; }

  /* Charts grid */
  .charts { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
  .chart-card { background: var(--card); border: 1px solid var(--border);
    border-radius: 10px; padding: 16px; }
  .chart-card h3 { font-size: 13px; color: var(--muted); margin-bottom: 12px; }

  /* Client cards */
  .clients { display: grid; grid-template-columns: repeat(5, 1fr); gap: 10px; }
  .client-card { background: var(--card); border: 1px solid var(--border);
    border-radius: 10px; padding: 14px; text-align: center; transition: border-color .3s; }
  .client-card.flagged  { border-color: var(--red); background: rgba(226,75,74,.08); }
  .client-card.trusted  { border-color: var(--teal); }
  .client-card .cname { font-size: 12px; font-weight: 600; margin-bottom: 8px; }
  .client-card.flagged .cname { color: var(--red); }
  .client-card.trusted .cname { color: var(--teal2); }
  .tau-label { font-size: 10px; color: var(--muted); margin-bottom: 4px; }
  .tau-bar-bg { height: 5px; background: var(--border); border-radius: 3px; margin-bottom: 6px; }
  .tau-bar { height: 5px; border-radius: 3px; transition: width .5s; }
  .tau-val { font-size: 18px; font-weight: 700; }
  .alpha-val { font-size: 11px; color: var(--muted); margin-top: 2px; }
  .attack-badge { font-size: 10px; background: rgba(226,75,74,.2);
    color: var(--red); padding: 2px 8px; border-radius: 20px; margin-top: 6px;
    display: inline-block; }

  /* Blockchain card */
  .chain-card { background: var(--card); border: 1px solid var(--border);
    border-radius: 10px; padding: 16px; }
  .chain-card h3 { font-size: 13px; color: var(--muted); margin-bottom: 10px; }
  .hash-display { font-family: monospace; font-size: 11px; color: var(--teal2);
    background: rgba(2,128,144,.08); padding: 8px 12px; border-radius: 6px;
    word-break: break-all; }

  /* Log */
  .log-card { background: var(--card); border: 1px solid var(--border);
    border-radius: 10px; padding: 16px; }
  .log-card h3 { font-size: 13px; color: var(--muted); margin-bottom: 10px; }
  .log-box { font-family: monospace; font-size: 11px; max-height: 120px;
    overflow-y: auto; line-height: 1.8; }
  .log-line { padding: 1px 0; }
  .log-line.flagged { color: var(--red); }
  .log-line.ok      { color: var(--teal2); }
  .log-line.info    { color: var(--muted); }

  /* Waiting state */
  .waiting-msg { text-align: center; padding: 60px; color: var(--muted); font-size: 14px; }
</style>
</head>
<body>

<div class="header">
  <div>
    <h1>Federated Learning — Trust & Attack Dashboard</h1>
    <div class="sub">Blockchain-Based Dynamic Trust Modeling · Flower gRPC · PyTorch</div>
  </div>
  <div style="text-align:right">
    <div style="font-size:13px">
      <span class="status-dot waiting" id="status-dot"></span>
      <span id="status-text">Waiting for training...</span>
    </div>
    <div style="font-size:11px; color:var(--muted); margin-top:4px" id="last-update">—</div>
  </div>
</div>

<div class="main" id="main-content">
  <div class="waiting-msg" id="waiting-msg">
    Waiting for FL training to start...<br>
    <span style="font-size:12px;margin-top:8px;display:block">
      Run: <code style="color:var(--teal2)">python -m split2.main --attack label_flip --malicious 1</code>
    </span>
  </div>
</div>

<template id="dashboard-template">
  <!-- Round progress -->
  <div class="progress-wrap">
    <div class="progress-label">
      <span>Round <strong id="cur-round">0</strong> / <strong id="tot-rounds">25</strong></span>
      <span id="status-label" style="color:var(--teal2)">Running</span>
    </div>
    <div class="progress-bar-bg">
      <div class="progress-bar" id="progress-bar" style="width:0%"></div>
    </div>
  </div>

  <!-- Metrics -->
  <div class="metrics">
    <div class="mc">
      <div class="label">Global F1</div>
      <div class="value" id="m-f1">—</div>
      <div class="sub">current round</div>
    </div>
    <div class="mc">
      <div class="label">Best F1</div>
      <div class="value" id="m-best-f1">—</div>
      <div class="sub">across all rounds</div>
    </div>
    <div class="mc">
      <div class="label">AUC-ROC</div>
      <div class="value" id="m-auc">—</div>
      <div class="sub">current round</div>
    </div>
    <div class="mc">
      <div class="label">Detection Rate</div>
      <div class="value" id="m-det">0/0</div>
      <div class="sub">rounds flagged correctly</div>
    </div>
  </div>

  <!-- Client cards -->
  <div class="clients" id="client-cards"></div>

  <!-- Charts -->
  <div class="charts">
    <div class="chart-card">
      <h3>Global F1 &amp; AUC over rounds</h3>
      <div style="position:relative;height:200px">
        <canvas id="perf-chart"></canvas>
      </div>
    </div>
    <div class="chart-card">
      <h3>Trust scores (τ) — attacker shown dashed</h3>
      <div style="position:relative;height:200px">
        <canvas id="trust-chart"></canvas>
      </div>
    </div>
  </div>

  <div class="charts">
    <div class="chart-card">
      <h3>Anomaly scores (α) — threshold 0.45</h3>
      <div style="position:relative;height:160px">
        <canvas id="alpha-chart"></canvas>
      </div>
    </div>
    <div class="chain-card">
      <h3>Blockchain — latest block hash</h3>
      <div style="font-size:12px;color:var(--muted);margin-bottom:8px">
        Round <span id="chain-round">—</span> · SHA-256 model fingerprint
      </div>
      <div class="hash-display" id="hash-display">Waiting...</div>
      <div style="font-size:11px;color:var(--muted);margin-top:8px">
        False positives: <span id="fp-count" style="color:var(--teal2)">0</span> &nbsp;|&nbsp;
        Detected: <span id="det-count" style="color:var(--teal2)">0</span> rounds
      </div>
    </div>
  </div>

  <!-- Log -->
  <div class="log-card">
    <h3>Live round log</h3>
    <div class="log-box" id="log-box"></div>
  </div>
</template>

<script>
const COLORS = ['#378ADD','#E24B4A','#639922','#BA7517','#7F77DD'];
let perfChart = null, trustChart = null, alphaChart = null;
let initialized = false;
let lastRound = -1;

function initCharts() {
  const baseOpts = {
    responsive: true, maintainAspectRatio: false,
    plugins: { legend: { display: false } },
    animation: { duration: 300 },
  };

  perfChart = new Chart(document.getElementById('perf-chart'), {
    type: 'line',
    data: {
      labels: [],
      datasets: [
        { label: 'F1',  data: [], borderColor: '#378ADD', backgroundColor: '#378ADD22',
          tension: 0.3, fill: true, pointRadius: 3, borderWidth: 2 },
        { label: 'AUC', data: [], borderColor: '#02C39A', backgroundColor: '#02C39A11',
          tension: 0.3, fill: true, pointRadius: 3, borderWidth: 2 },
      ]
    },
    options: { ...baseOpts, scales: {
      x: { ticks: { color: '#64748b', font: { size: 10 } }, grid: { color: '#1E3A5F' } },
      y: { min: 0.5, max: 1.0, ticks: { color: '#64748b', font: { size: 10 } },
           grid: { color: '#1E3A5F' } }
    }}
  });

  const trustDatasets = Array.from({ length: 5 }, (_, i) => ({
    label: `Bank ${i}`,
    data: [],
    borderColor: COLORS[i],
    borderDash: [],
    tension: 0.3,
    pointRadius: 2,
    borderWidth: i === 1 ? 2.5 : 1.5,
  }));
  trustChart = new Chart(document.getElementById('trust-chart'), {
    type: 'line',
    data: { labels: [], datasets: trustDatasets },
    options: { ...baseOpts, scales: {
      x: { ticks: { color: '#64748b', font: { size: 10 } }, grid: { color: '#1E3A5F' } },
      y: { min: 0, max: 1, ticks: { color: '#64748b', font: { size: 10 } },
           grid: { color: '#1E3A5F' } }
    }}
  });

  alphaChart = new Chart(document.getElementById('alpha-chart'), {
    type: 'bar',
    data: {
      labels: ['Bank 0','Bank 1','Bank 2','Bank 3','Bank 4'],
      datasets: [{
        data: [0,0,0,0,0],
        backgroundColor: COLORS.map(c => c + 'BB'),
        borderRadius: 6,
      }]
    },
    options: {
      ...baseOpts,
      scales: {
        x: { ticks: { color: '#64748b', font: { size: 11 } }, grid: { color: '#1E3A5F' } },
        y: { min: 0, max: 1, ticks: { color: '#64748b', font: { size: 10 } },
             grid: { color: '#1E3A5F' } }
      }
    }
  });
}

function renderClientCards(state) {
  const container = document.getElementById('client-cards');
  container.innerHTML = '';
  for (let i = 0; i < 5; i++) {
    const isFlagged = state.flagged.includes(i);
    const tau   = (state.trust_latest[i] || 1.0).toFixed(3);
    const alpha = (state.alpha_latest[i] || 0.0).toFixed(3);
    const barColor = isFlagged ? '#E24B4A' : '#028090';
    const card = document.createElement('div');
    card.className = 'client-card ' + (isFlagged ? 'flagged' : 'trusted');
    card.innerHTML = `
      <div class="cname">Bank ${i}${isFlagged ? ' ⚠' : ''}</div>
      <div class="tau-label">Trust score τ</div>
      <div class="tau-bar-bg">
        <div class="tau-bar" style="width:${(tau*100).toFixed(0)}%;background:${barColor}"></div>
      </div>
      <div class="tau-val" style="color:${barColor}">${tau}</div>
      <div class="alpha-val">α = ${alpha}</div>
      ${isFlagged ? '<div class="attack-badge">MALICIOUS</div>' : ''}
    `;
    container.appendChild(card);
  }
}

function updateCharts(state) {
  const lbl = String(state.current_round);
  perfChart.data.labels = state.round_nums.map(String);
  perfChart.data.datasets[0].data = state.f1_history;
  perfChart.data.datasets[1].data = state.auc_history;
  perfChart.update('none');

  trustChart.data.labels = state.round_nums.map(String);
  for (let i = 0; i < 5; i++) {
    trustChart.data.datasets[i].data = state.trust_history[i] || [];
    // Dashed line for attacker
    if (state.attacker === i) {
      trustChart.data.datasets[i].borderDash = [5, 3];
      trustChart.data.datasets[i].borderWidth = 2.5;
    }
  }
  trustChart.update('none');

  alphaChart.data.datasets[0].data = Array.from({length:5}, (_,i) =>
    parseFloat(state.alpha_latest[i] || 0)
  );
  alphaChart.data.datasets[0].backgroundColor = Array.from({length:5}, (_,i) =>
    state.flagged.includes(i) ? '#E24B4ABB' : COLORS[i] + 'BB'
  );
  alphaChart.update('none');
}

function appendLog(state) {
  const box = document.getElementById('log-box');
  if (state.current_round <= lastRound) return;
  const isFlagged = state.flagged.length > 0;
  const div = document.createElement('div');
  div.className = 'log-line ' + (isFlagged ? 'flagged' : 'ok');
  div.textContent =
    `Round ${state.current_round}: F1=${state.global_f1} AUC=${state.global_auc}` +
    (isFlagged
      ? ` | FLAGGED: Client ${state.flagged.join(',')} α=${state.alpha_latest[state.attacker]}`
      : ` | All clients trusted`);
  box.appendChild(div);
  box.scrollTop = box.scrollHeight;
  lastRound = state.current_round;
}

function applyState(state) {
  // First time — swap waiting msg for dashboard
  if (!initialized) {
    const main = document.getElementById('main-content');
    const tmpl = document.getElementById('dashboard-template');
    main.innerHTML = '';
    main.appendChild(tmpl.content.cloneNode(true));
    initCharts();
    initialized = true;
    document.getElementById('waiting-msg') && document.getElementById('waiting-msg').remove();
  }

  // Status dot
  const dot  = document.getElementById('status-dot');
  const stxt = document.getElementById('status-text');
  dot.className  = 'status-dot ' + state.status;
  stxt.textContent = state.status === 'running' ? 'Training in progress'
                   : state.status === 'complete' ? 'Training complete'
                   : 'Waiting...';

  // Progress
  const pct = Math.round((state.current_round / state.total_rounds) * 100);
  document.getElementById('progress-bar').style.width = pct + '%';
  document.getElementById('cur-round').textContent   = state.current_round;
  document.getElementById('tot-rounds').textContent  = state.total_rounds;
  document.getElementById('status-label').textContent =
    state.status === 'complete' ? '✓ Complete' : 'Running...';

  // Metrics
  document.getElementById('m-f1').textContent      = state.global_f1;
  document.getElementById('m-best-f1').textContent  = state.best_f1;
  document.getElementById('m-auc').textContent      = state.global_auc;
  document.getElementById('m-det').textContent      =
    `${state.detected_rounds}/${state.current_round}`;

  // Blockchain
  document.getElementById('chain-round').textContent  = state.blockchain_round;
  document.getElementById('hash-display').textContent =
    state.model_hash || 'Not yet recorded';
  document.getElementById('fp-count').textContent  = state.false_positives;
  document.getElementById('det-count').textContent = state.detected_rounds;

  // Last update time
  document.getElementById('last-update').textContent =
    'Updated ' + new Date().toLocaleTimeString();

  renderClientCards(state);
  updateCharts(state);
  appendLog(state);
}

// Server-Sent Events — real-time push from the server
const evtSource = new EventSource('/stream');
evtSource.onmessage = (e) => {
  const state = JSON.parse(e.data);
  if (state.status !== 'waiting') {
    applyState(state);
  }
};
evtSource.onerror = () => {
  document.getElementById('status-text').textContent = 'Connection lost — reconnecting...';
};
</script>
</body>
</html>
"""


# =============================================================================
# HTTP REQUEST HANDLER
# =============================================================================

class DashboardHandler(http.server.BaseHTTPRequestHandler):
    """Serves dashboard HTML and SSE stream."""

    watcher: LogWatcher = None   # set by main()
    clients_lock = threading.Lock()
    sse_clients  = []

    def log_message(self, fmt, *args):
        pass  # suppress default access log noise

    def do_GET(self):
        path = urlparse(self.path).path

        if path == "/" or path == "/index.html":
            self._serve_html()
        elif path == "/stream":
            self._serve_sse()
        elif path == "/state":
            self._serve_json()
        else:
            self.send_error(HTTPStatus.NOT_FOUND)

    def _serve_html(self):
        data = DASHBOARD_HTML.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type",   "text/html; charset=utf-8")
        self.send_header("Content-Length", len(data))
        self.end_headers()
        self.wfile.write(data)

    def _serve_json(self):
        state = DashboardHandler.watcher.get_state()
        data  = json.dumps(state).encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type",   "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", len(data))
        self.end_headers()
        self.wfile.write(data)

    def _serve_sse(self):
        """Server-Sent Events — keep connection open, push updates."""
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type",  "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection",    "keep-alive")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

        try:
            while True:
                state = DashboardHandler.watcher.get_state()
                data  = json.dumps(state)
                self.wfile.write(f"data: {data}\n\n".encode("utf-8"))
                self.wfile.flush()
                time.sleep(POLL_SECS)
        except (BrokenPipeError, ConnectionResetError):
            pass


# =============================================================================
# BACKGROUND WATCHER THREAD
# =============================================================================

def watch_loop(watcher: LogWatcher):
    """Continuously refresh the log file in background."""
    while True:
        watcher.refresh()
        time.sleep(POLL_SECS)


# =============================================================================
# MAIN
# =============================================================================

def build_parser():
    p = argparse.ArgumentParser(description="FL Real-time Dashboard Server")
    p.add_argument("--log",  type=str, default=DEFAULT_LOG,
                   help="Path to trust_training_log.json")
    p.add_argument("--port", type=int, default=5000,
                   help="HTTP port (default: 5000)")
    p.add_argument("--host", type=str, default="0.0.0.0",
                   help="Host to bind (0.0.0.0 = accessible from any machine)")
    return p


def main():
    args   = build_parser().parse_args()
    watcher = LogWatcher(log_path=args.log)
    DashboardHandler.watcher = watcher

    # Start background file watcher
    t = threading.Thread(target=watch_loop, args=(watcher,), daemon=True)
    t.start()

    # Start HTTP server
    server = http.server.ThreadingHTTPServer(
        (args.host, args.port), DashboardHandler
    )

    print(f"\n{'='*55}")
    print(f"  FL Trust Dashboard")
    print(f"  Watching : {args.log}")
    print(f"  Open in browser:")
    print(f"    http://localhost:{args.port}           (this machine)")

    # Try to get WiFi IP for cross-machine access
    try:
        import socket
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        print(f"    http://{ip}:{args.port}   (any machine on your WiFi)")
    except Exception:
        pass

    print(f"\n  Dashboard auto-updates every {POLL_SECS}s as training runs.")
    print(f"  Press Ctrl+C to stop.")
    print(f"{'='*55}\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nDashboard stopped.")


if __name__ == "__main__":
    main()
