from __future__ import annotations

import json
import traceback
from html import escape
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from web3 import Web3


RPC_URL = "http://127.0.0.1:8545"
HOST = "127.0.0.1"
PORT = 8000
THIS_DIR = Path(__file__).resolve().parent
DEPLOY_FILE = THIS_DIR / "eth_deployment.json"


def load_contract():
    with DEPLOY_FILE.open("r", encoding="utf-8") as f:
        deployment = json.load(f)

    w3 = Web3(Web3.HTTPProvider(RPC_URL))
    if not w3.is_connected():
        raise RuntimeError(f"Ganache is not reachable at {RPC_URL}")

    address = Web3.to_checksum_address(deployment["address"])
    code = w3.eth.get_code(address)
    if code == b"":
        raise RuntimeError(
            f"No contract code found at {address}. Ganache may have been restarted."
        )

    contract = w3.eth.contract(address=address, abi=deployment["abi"])
    return w3, address, contract


def fetch_contract_state():
    w3, address, contract = load_contract()

    round_count = contract.functions.getRoundCount().call()
    alert_count = contract.functions.getAlertCount().call()
    audit_count = contract.functions.getAuditCount().call()
    intact, broken_at = contract.functions.verifyFullChain().call()

    rounds = []
    for round_no in range(1, round_count + 1):
        model = contract.functions.getModel(round_no).call()
        rounds.append(
            {
                "round": round_no,
                "modelHash": model[0],
                "blockHash": model[1],
                "prevBlockHash": model[2],
                "globalF1": model[3],
                "globalAUC": model[4],
                "timestamp": model[5],
                "exists": model[6],
            }
        )

    alerts = []
    for alert_id in range(1, alert_count + 1):
        alert = contract.functions.getAlert(alert_id).call()
        alerts.append(
            {
                "id": alert_id,
                "round": alert[0],
                "alertType": alert[1],
                "severity": alert[2],
                "timestamp": alert[3],
            }
        )

    audits = []
    for audit_id in range(1, audit_count + 1):
        audit = contract.functions.getAuditEvent(audit_id).call()
        audits.append(
            {
                "id": audit_id,
                "eventType": audit[0],
                "round": audit[1],
                "actor": audit[2],
                "dataJson": audit[3],
                "timestamp": audit[4],
            }
        )

    return {
        "rpcUrl": RPC_URL,
        "chainId": w3.eth.chain_id,
        "address": address,
        "owner": contract.functions.owner().call(),
        "roundCount": round_count,
        "alertCount": alert_count,
        "auditCount": audit_count,
        "fullChain": {
            "intact": intact,
            "brokenAtRound": broken_at,
        },
        "rounds": rounds,
        "alerts": alerts,
        "audits": audits,
    }


def render_html(state: dict) -> str:
    cards = [
        ("RPC URL", state["rpcUrl"]),
        ("Chain ID", str(state["chainId"])),
        ("Contract", state["address"]),
        ("Owner", state["owner"]),
        ("Rounds", str(state["roundCount"])),
        ("Alerts", str(state["alertCount"])),
        ("Audits", str(state["auditCount"])),
        (
            "Full Chain",
            f'intact={state["fullChain"]["intact"]}, '
            f'brokenAtRound={state["fullChain"]["brokenAtRound"]}',
        ),
    ]

    round_rows = "".join(
        f"""
        <tr>
          <td>{row["round"]}</td>
          <td title="{escape(row["modelHash"])}">{escape(shorten(row["modelHash"]))}</td>
          <td title="{escape(row["blockHash"])}">{escape(shorten(row["blockHash"]))}</td>
          <td title="{escape(row["prevBlockHash"])}">{escape(shorten(row["prevBlockHash"]))}</td>
          <td>{row["globalF1"]}</td>
          <td>{row["globalAUC"]}</td>
          <td>{row["timestamp"]}</td>
          <td>{row["exists"]}</td>
        </tr>
        """
        for row in state["rounds"]
    )

    alert_rows = "".join(
        f"""
        <tr>
          <td>{row["id"]}</td>
          <td>{row["round"]}</td>
          <td>{escape(row["alertType"])}</td>
          <td>{escape(row["severity"])}</td>
          <td>{row["timestamp"]}</td>
        </tr>
        """
        for row in state["alerts"]
    ) or '<tr><td colspan="5">No alerts recorded.</td></tr>'

    audit_rows = "".join(
        f"""
        <tr>
          <td>{row["id"]}</td>
          <td>{escape(row["eventType"])}</td>
          <td>{row["round"]}</td>
          <td>{escape(row["actor"])}</td>
          <td title="{escape(row["dataJson"])}">{escape(shorten(row["dataJson"], 80))}</td>
          <td>{row["timestamp"]}</td>
        </tr>
        """
        for row in state["audits"]
    ) or '<tr><td colspan="6">No audit events recorded.</td></tr>'

    card_html = "".join(
        f"""
        <div class="card">
          <div class="label">{escape(label)}</div>
          <div class="value">{escape(value)}</div>
        </div>
        """
        for label, value in cards
    )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Ganache Contract Viewer</title>
  <style>
    :root {{
      --bg: #f6f2e8;
      --panel: #fffdf7;
      --ink: #1f1a14;
      --muted: #6b6258;
      --line: #d7cdbf;
      --accent: #0d7a5f;
      --accent-soft: #d9efe8;
      --warn: #9b3d2f;
      --shadow: 0 18px 40px rgba(40, 28, 10, 0.10);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Georgia, "Times New Roman", serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(13,122,95,0.10), transparent 30%),
        linear-gradient(180deg, #f3ede0 0%, var(--bg) 100%);
    }}
    .wrap {{
      max-width: 1200px;
      margin: 0 auto;
      padding: 32px 20px 48px;
    }}
    h1 {{
      margin: 0 0 10px;
      font-size: clamp(2rem, 4vw, 3.5rem);
      line-height: 1;
    }}
    p {{
      margin: 0 0 24px;
      color: var(--muted);
      max-width: 760px;
      font-size: 1.05rem;
    }}
    .toolbar {{
      display: flex;
      gap: 12px;
      flex-wrap: wrap;
      margin-bottom: 26px;
    }}
    .button {{
      text-decoration: none;
      color: var(--ink);
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 10px 16px;
      box-shadow: var(--shadow);
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 14px;
      margin-bottom: 26px;
    }}
    .card, section {{
      background: rgba(255, 253, 247, 0.92);
      border: 1px solid var(--line);
      border-radius: 18px;
      box-shadow: var(--shadow);
    }}
    .card {{
      padding: 18px;
    }}
    .label {{
      font-size: 0.82rem;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      color: var(--muted);
      margin-bottom: 8px;
    }}
    .value {{
      font-size: 1rem;
      overflow-wrap: anywhere;
    }}
    section {{
      padding: 18px;
      margin-top: 18px;
    }}
    h2 {{
      margin: 0 0 14px;
      font-size: 1.4rem;
    }}
    .table-wrap {{
      overflow-x: auto;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-family: "Courier New", monospace;
      font-size: 0.92rem;
    }}
    th, td {{
      padding: 10px 12px;
      border-bottom: 1px solid var(--line);
      text-align: left;
      vertical-align: top;
    }}
    th {{
      color: var(--muted);
      font-weight: 700;
    }}
    tr:hover td {{
      background: rgba(13, 122, 95, 0.05);
    }}
    .footer {{
      margin-top: 20px;
      color: var(--muted);
      font-size: 0.95rem;
    }}
    .status-ok {{
      color: var(--accent);
      font-weight: 700;
    }}
    .status-warn {{
      color: var(--warn);
      font-weight: 700;
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <h1>Ganache Contract Viewer</h1>
    <p>
      This page reads your local <strong>ModelRegistry</strong> contract directly from Ganache and
      shows rounds, alerts, and audit events without needing Remix.
    </p>
    <div class="toolbar">
      <a class="button" href="/">Refresh page</a>
      <a class="button" href="/api/state">Open JSON API</a>
      <span class="button">Chain integrity:
        <span class="{('status-ok' if state['fullChain']['intact'] else 'status-warn')}">
          {escape(str(state["fullChain"]["intact"]))}
        </span>
      </span>
    </div>
    <div class="grid">{card_html}</div>

    <section>
      <h2>Rounds</h2>
      <div class="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Round</th>
              <th>Model Hash</th>
              <th>Block Hash</th>
              <th>Prev Block Hash</th>
              <th>F1</th>
              <th>AUC</th>
              <th>Timestamp</th>
              <th>Exists</th>
            </tr>
          </thead>
          <tbody>{round_rows}</tbody>
        </table>
      </div>
    </section>

    <section>
      <h2>Alerts</h2>
      <div class="table-wrap">
        <table>
          <thead>
            <tr>
              <th>ID</th>
              <th>Round</th>
              <th>Type</th>
              <th>Severity</th>
              <th>Timestamp</th>
            </tr>
          </thead>
          <tbody>{alert_rows}</tbody>
        </table>
      </div>
    </section>

    <section>
      <h2>Audit Events</h2>
      <div class="table-wrap">
        <table>
          <thead>
            <tr>
              <th>ID</th>
              <th>Event Type</th>
              <th>Round</th>
              <th>Actor</th>
              <th>Data</th>
              <th>Timestamp</th>
            </tr>
          </thead>
          <tbody>{audit_rows}</tbody>
        </table>
      </div>
    </section>

    <div class="footer">
      Local viewer URL: http://{HOST}:{PORT}
    </div>
  </div>
</body>
</html>"""


def shorten(value: str, limit: int = 28) -> str:
    if len(value) <= limit:
        return value
    return f"{value[:12]}...{value[-12:]}"


class ContractViewerHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        try:
            state = fetch_contract_state()
            if parsed.path == "/api/state":
                body = json.dumps(state, indent=2).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return

            if parsed.path == "/":
                body = render_html(state).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return

            self.send_error(404, "Not Found")
        except Exception as exc:
            message = (
                "<h1>Contract Viewer Error</h1>"
                f"<p>{escape(str(exc))}</p>"
                "<pre>"
                f"{escape(traceback.format_exc())}"
                "</pre>"
            ).encode("utf-8")
            self.send_response(500)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(message)))
            self.end_headers()
            self.wfile.write(message)

    def log_message(self, format, *args):
        return


def main():
    server = ThreadingHTTPServer((HOST, PORT), ContractViewerHandler)
    print(f"Viewer running at http://{HOST}:{PORT}")
    print(f"JSON API available at http://{HOST}:{PORT}/api/state")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping viewer.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
