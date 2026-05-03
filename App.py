"""
Fraud Detection Web App - Flask Backend
Place this file in: D:\credit_fraud_detection\fraud_detection\app.py
Run with: python app.py
"""

from flask import Flask, request, jsonify, render_template_string
import numpy as np
import joblib
import os
import json
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

app = Flask(__name__)

# ── Global model state ──────────────────────────────────────────────────────
lstm_model = None
iso_model  = None
scaler     = None
models_loaded = False
load_error    = ""

MERCHANT_CATS = [
    "food_dining", "gas_transport", "grocery_pos", "health_fitness",
    "home", "kids_pets", "misc_net", "misc_pos", "personal_care",
    "shopping_net", "shopping_pos", "travel", "entertainment", "education"
]
STATES = [
    "AL","AK","AZ","AR","CA","CO","CT","DE","FL","GA","HI","ID","IL","IN",
    "IA","KS","KY","LA","ME","MD","MA","MI","MN","MS","MO","MT","NE","NV",
    "NH","NJ","NM","NY","NC","ND","OH","OK","OR","PA","RI","SC","SD","TN",
    "TX","UT","VT","VA","WA","WV","WI","WY"
]

def load_models():
    global lstm_model, iso_model, scaler, models_loaded, load_error
    try:
        import tensorflow as tf
        lstm_path = "outputs/lstm_model.keras"
        iso_path  = "outputs/isolation_forest.pkl"
        scaler_path = "outputs/scaler.pkl"

        if not os.path.exists(lstm_path):
            load_error = f"LSTM model not found at {lstm_path}"
            return
        if not os.path.exists(iso_path):
            load_error = f"IsoForest model not found at {iso_path}"
            return

        lstm_model = tf.keras.models.load_model(lstm_path)
        iso_model  = joblib.load(iso_path)

        if os.path.exists(scaler_path):
            scaler = joblib.load(scaler_path)

        models_loaded = True
        print("✅ Models loaded successfully.")
    except Exception as e:
        load_error = str(e)
        print(f"❌ Model load error: {e}")


def build_feature_vector(data):
    """Convert form data into the 12-feature vector used during training."""
    dt = datetime.strptime(data["datetime"], "%Y-%m-%dT%H:%M")
    hour     = dt.hour
    dow      = dt.weekday()
    month    = dt.month
    amt      = float(data["amount"])

    # Merchant category encoding
    cat = data.get("category", "misc_pos")
    cat_idx = MERCHANTS_IDX.get(cat, 0)

    # State encoding
    state = data.get("state", "CA")
    state_idx = STATES_IDX.get(state, 0)

    # City population (log-scaled estimate)
    city_pop = float(data.get("city_pop", 50000))
    city_pop_log = np.log1p(city_pop)

    # Lat/long (normalised)
    lat  = float(data.get("lat",  37.0)) / 90.0
    long = float(data.get("long", -95.0)) / 180.0
    merch_lat  = float(data.get("merch_lat",  37.0)) / 90.0
    merch_long = float(data.get("merch_long", -95.0)) / 180.0

    feat = np.array([
        amt / 1000.0,   # normalised amount
        hour / 23.0,
        dow  / 6.0,
        month / 12.0,
        cat_idx / max(len(MERCHANT_CATS)-1, 1),
        state_idx / max(len(STATES)-1, 1),
        city_pop_log / 15.0,
        lat, long,
        merch_lat, merch_long,
        0.0             # placeholder (distance feature)
    ], dtype=np.float32)

    return feat


# Build lookup dicts once
MERCHANTS_IDX = {cat: i for i, cat in enumerate(MERCHANT_CATS)}
STATES_IDX    = {s: i for i, s in enumerate(STATES)}

# ─────────────────────────── HTML TEMPLATE ──────────────────────────────────
HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>FraudShield — Detection System</title>
<link href="https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Mono:wght@300;400;500&family=DM+Sans:wght@300;400;500;600&display=swap" rel="stylesheet">
<style>
:root {
  --bg: #0a0e1a;
  --surface: #111827;
  --surface2: #1a2235;
  --border: #1e2d45;
  --teal: #00d4b4;
  --teal-dim: #00856f;
  --red: #ff4d6d;
  --gold: #fbbf24;
  --text: #e2e8f0;
  --muted: #64748b;
  --safe: #22c55e;
}
* { margin:0; padding:0; box-sizing:border-box; }
body {
  background: var(--bg);
  color: var(--text);
  font-family: 'DM Sans', sans-serif;
  min-height: 100vh;
  overflow-x: hidden;
}
/* background grid */
body::before {
  content:'';
  position:fixed; inset:0;
  background-image:
    linear-gradient(rgba(0,212,180,.03) 1px, transparent 1px),
    linear-gradient(90deg, rgba(0,212,180,.03) 1px, transparent 1px);
  background-size: 40px 40px;
  pointer-events:none; z-index:0;
}

header {
  position: relative; z-index:10;
  border-bottom: 1px solid var(--border);
  padding: 1.2rem 2rem;
  display: flex; align-items: center; gap: 1rem;
  background: rgba(10,14,26,.9);
  backdrop-filter: blur(10px);
}
.logo-mark {
  width:36px; height:36px;
  background: var(--teal);
  clip-path: polygon(50% 0%, 100% 25%, 100% 75%, 50% 100%, 0% 75%, 0% 25%);
  display:flex; align-items:center; justify-content:center;
  font-size:.9rem; font-weight:700; color:#000;
}
header h1 {
  font-family: 'DM Serif Display', serif;
  font-size:1.5rem;
  letter-spacing:.5px;
}
header h1 span { color: var(--teal); }
.status-pill {
  margin-left:auto;
  padding:.3rem .8rem;
  border-radius:999px;
  font-size:.75rem;
  font-family:'DM Mono', monospace;
  border: 1px solid;
}
.status-pill.ok  { border-color:var(--safe); color:var(--safe); background:rgba(34,197,94,.1); }
.status-pill.err { border-color:var(--red);  color:var(--red);  background:rgba(255,77,109,.1); }

main {
  position:relative; z-index:1;
  max-width: 1200px;
  margin: 0 auto;
  padding: 2.5rem 1.5rem;
  display: grid;
  grid-template-columns: 1fr 420px;
  gap: 2rem;
}

.section-label {
  font-family:'DM Mono', monospace;
  font-size:.65rem;
  letter-spacing:2px;
  text-transform:uppercase;
  color: var(--teal);
  margin-bottom:.75rem;
}

/* Form card */
.form-card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 16px;
  padding: 2rem;
}
.form-card h2 {
  font-family:'DM Serif Display', serif;
  font-size:1.4rem;
  margin-bottom:1.5rem;
  color: var(--text);
}

.form-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 1rem;
}
.form-group { display:flex; flex-direction:column; gap:.4rem; }
.form-group.full { grid-column: 1/-1; }
label {
  font-size:.75rem;
  font-family:'DM Mono', monospace;
  color: var(--muted);
  text-transform:uppercase;
  letter-spacing:.8px;
}
input, select {
  background: var(--surface2);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: .65rem .9rem;
  color: var(--text);
  font-size: .9rem;
  font-family: 'DM Sans', sans-serif;
  transition: border-color .2s, box-shadow .2s;
  outline: none;
}
input:focus, select:focus {
  border-color: var(--teal-dim);
  box-shadow: 0 0 0 3px rgba(0,212,180,.08);
}
select option { background: var(--surface2); }

.coord-row {
  display:grid; grid-template-columns:1fr 1fr; gap:.6rem;
}

.txn-history-label {
  margin-top:1.5rem;
  padding: .8rem 1rem;
  background: rgba(0,212,180,.06);
  border: 1px solid rgba(0,212,180,.15);
  border-radius:8px;
  font-size:.8rem;
  color: var(--muted);
  line-height:1.6;
}
.txn-history-label strong { color: var(--teal); }

.btn-predict {
  margin-top:1.5rem;
  width:100%;
  padding:1rem;
  background: var(--teal);
  color:#000;
  font-size:1rem;
  font-weight:600;
  font-family:'DM Sans', sans-serif;
  border:none;
  border-radius:10px;
  cursor:pointer;
  letter-spacing:.3px;
  transition: transform .15s, box-shadow .15s, background .2s;
  display:flex; align-items:center; justify-content:center; gap:.6rem;
}
.btn-predict:hover {
  background: #00f0ce;
  transform:translateY(-1px);
  box-shadow: 0 8px 24px rgba(0,212,180,.25);
}
.btn-predict:active { transform:translateY(0); }
.btn-predict:disabled { background:var(--muted); cursor:not-allowed; transform:none; box-shadow:none; }
.spinner {
  width:16px; height:16px;
  border:2px solid rgba(0,0,0,.3);
  border-top-color:#000;
  border-radius:50%;
  animation:spin .7s linear infinite;
  display:none;
}
@keyframes spin { to { transform:rotate(360deg); } }

/* Right panel */
.right-panel { display:flex; flex-direction:column; gap:1.5rem; }

/* Result card */
.result-card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 16px;
  padding: 1.5rem;
  min-height: 280px;
  display:flex; flex-direction:column;
  transition: border-color .4s;
}
.result-card.fraud  { border-color: var(--red); }
.result-card.legit  { border-color: var(--safe); }
.result-card.warn   { border-color: var(--gold); }

.result-placeholder {
  flex:1; display:flex; flex-direction:column;
  align-items:center; justify-content:center;
  gap:.6rem; color:var(--muted);
}
.result-placeholder svg { opacity:.3; }
.result-placeholder p { font-size:.9rem; }

.verdict {
  display:none;
  flex-direction:column;
  gap:1rem;
}
.verdict-badge {
  display:flex; align-items:center; gap:.8rem;
}
.verdict-icon {
  width:52px; height:52px; border-radius:12px;
  display:flex; align-items:center; justify-content:center;
  font-size:1.6rem;
}
.verdict-icon.fraud { background:rgba(255,77,109,.15); }
.verdict-icon.legit { background:rgba(34,197,94,.15); }
.verdict-icon.warn  { background:rgba(251,191,36,.15); }
.verdict-title {
  font-family:'DM Serif Display', serif;
  font-size:1.6rem;
}
.verdict-title.fraud { color:var(--red); }
.verdict-title.legit { color:var(--safe); }
.verdict-title.warn  { color:var(--gold); }
.verdict-sub { font-size:.8rem; color:var(--muted); }

.score-bars { display:flex; flex-direction:column; gap:.7rem; }
.score-row { display:flex; flex-direction:column; gap:.3rem; }
.score-row-head { display:flex; justify-content:space-between; align-items:center; }
.score-name { font-size:.75rem; font-family:'DM Mono',monospace; color:var(--muted); }
.score-val  { font-size:.8rem;  font-family:'DM Mono',monospace; font-weight:500; }
.bar-track {
  height:6px; background:var(--surface2);
  border-radius:999px; overflow:hidden;
}
.bar-fill {
  height:100%; border-radius:999px;
  transition: width .8s cubic-bezier(.16,1,.3,1);
}
.bar-lstm    { background: var(--teal); }
.bar-iso     { background: var(--gold); }
.bar-hybrid  { background: var(--red); }
.bar-safe    { background: var(--safe); }

.meta-grid {
  display:grid; grid-template-columns:1fr 1fr;
  gap:.6rem;
  padding-top:.8rem;
  border-top:1px solid var(--border);
}
.meta-item { display:flex; flex-direction:column; gap:.2rem; }
.meta-key  { font-size:.65rem; font-family:'DM Mono',monospace; color:var(--muted); text-transform:uppercase; letter-spacing:.8px; }
.meta-val  { font-size:.85rem; font-weight:500; }

/* History panel */
.history-card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 16px;
  padding: 1.5rem;
  flex:1;
  overflow:hidden;
}
.history-card h3 {
  font-family:'DM Serif Display',serif;
  font-size:1.1rem;
  margin-bottom:1rem;
}
.history-list { display:flex; flex-direction:column; gap:.5rem; max-height:280px; overflow-y:auto; }
.history-list::-webkit-scrollbar { width:4px; }
.history-list::-webkit-scrollbar-track { background:transparent; }
.history-list::-webkit-scrollbar-thumb { background:var(--border); border-radius:2px; }
.hist-item {
  display:flex; align-items:center; gap:.8rem;
  padding:.6rem .8rem;
  background:var(--surface2);
  border-radius:8px;
  font-size:.82rem;
  animation: fadeIn .3s ease;
}
@keyframes fadeIn { from{opacity:0;transform:translateY(-4px)} to{opacity:1;transform:none} }
.hist-dot {
  width:8px; height:8px; border-radius:50%; flex-shrink:0;
}
.hist-dot.fraud { background:var(--red); }
.hist-dot.legit { background:var(--safe); }
.hist-dot.warn  { background:var(--gold); }
.hist-info { flex:1; }
.hist-name { color:var(--text); }
.hist-time { color:var(--muted); font-size:.72rem; }
.hist-score {
  font-family:'DM Mono',monospace; font-size:.78rem;
  margin-left:auto;
}
.hist-score.fraud { color:var(--red); }
.hist-score.legit { color:var(--safe); }
.hist-score.warn  { color:var(--gold); }

.empty-hist {
  color:var(--muted); font-size:.85rem; text-align:center;
  padding:1.5rem 0;
}

/* Tooltip error */
.error-msg {
  display:none;
  margin-top:.8rem;
  padding:.7rem 1rem;
  background:rgba(255,77,109,.1);
  border:1px solid rgba(255,77,109,.3);
  border-radius:8px;
  color:var(--red);
  font-size:.82rem;
}

@media (max-width:900px){
  main { grid-template-columns:1fr; }
}
</style>
</head>
<body>

<header>
  <div class="logo-mark">F</div>
  <h1>Fraud<span>Shield</span></h1>
  <div class="status-pill {{ 'ok' if models_ok else 'err' }}" id="statusPill">
    {{ '● MODELS READY' if models_ok else '● ' + error_msg }}
  </div>
</header>

<main>
  <!-- LEFT: Input Form -->
  <div class="form-card">
    <div class="section-label">Transaction Input</div>
    <h2>Analyse a Transaction</h2>

    <div class="form-grid">
      <div class="form-group">
        <label>Date &amp; Time</label>
        <input type="datetime-local" id="datetime" value="2024-06-15T14:30">
      </div>
      <div class="form-group">
        <label>Amount (USD)</label>
        <input type="number" id="amount" placeholder="e.g. 124.50" min="0" step="0.01" value="124.50">
      </div>
      <div class="form-group">
        <label>Merchant Category</label>
        <select id="category">
          <option value="food_dining">Food &amp; Dining</option>
          <option value="gas_transport">Gas / Transport</option>
          <option value="grocery_pos">Grocery</option>
          <option value="health_fitness">Health &amp; Fitness</option>
          <option value="home">Home</option>
          <option value="kids_pets">Kids &amp; Pets</option>
          <option value="misc_net">Misc (Online)</option>
          <option value="misc_pos" selected>Misc (In-store)</option>
          <option value="personal_care">Personal Care</option>
          <option value="shopping_net">Shopping (Online)</option>
          <option value="shopping_pos">Shopping (In-store)</option>
          <option value="travel">Travel</option>
          <option value="entertainment">Entertainment</option>
          <option value="education">Education</option>
        </select>
      </div>
      <div class="form-group">
        <label>State</label>
        <select id="state">
          {% for s in states %}
          <option value="{{ s }}" {% if s=='TX' %}selected{% endif %}>{{ s }}</option>
          {% endfor %}
        </select>
      </div>
      <div class="form-group">
        <label>City Population</label>
        <input type="number" id="city_pop" value="50000" min="100">
      </div>
      <div class="form-group">
        <label>Card Holder ID</label>
        <input type="text" id="user_id" placeholder="e.g. 4532015112830366" value="4532015112830366">
      </div>

      <div class="form-group full">
        <label>Cardholder Location (Lat / Long)</label>
        <div class="coord-row">
          <input type="number" id="lat"  placeholder="Latitude"  value="33.4484"  step="0.0001">
          <input type="number" id="long" placeholder="Longitude" value="-112.0740" step="0.0001">
        </div>
      </div>
      <div class="form-group full">
        <label>Merchant Location (Lat / Long)</label>
        <div class="coord-row">
          <input type="number" id="merch_lat"  placeholder="Lat"  value="33.5000"  step="0.0001">
          <input type="number" id="merch_long" placeholder="Long" value="-112.1000" step="0.0001">
        </div>
      </div>
    </div>

    <div class="txn-history-label">
      ℹ️ The LSTM model uses a <strong>window of 10 transactions</strong>.
      For a single transaction, the system pads the sequence with neutral values.
      For best results, submit multiple transactions from the same card holder in sequence.
    </div>

    <div class="error-msg" id="errMsg"></div>

    <button class="btn-predict" id="predictBtn" onclick="predict()">
      <div class="spinner" id="spinner"></div>
      <span id="btnText">Analyse Transaction</span>
    </button>
  </div>

  <!-- RIGHT PANEL -->
  <div class="right-panel">
    <!-- Result card -->
    <div class="result-card" id="resultCard">
      <div class="section-label">Detection Result</div>
      <div class="result-placeholder" id="placeholder">
        <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
          <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>
        </svg>
        <p>Submit a transaction to analyse</p>
      </div>
      <div class="verdict" id="verdictDiv">
        <div class="verdict-badge">
          <div class="verdict-icon" id="verdictIcon">🔍</div>
          <div>
            <div class="verdict-title" id="verdictTitle">—</div>
            <div class="verdict-sub" id="verdictSub">—</div>
          </div>
        </div>
        <div class="score-bars">
          <div class="score-row">
            <div class="score-row-head">
              <span class="score-name">LSTM Score</span>
              <span class="score-val" id="lstmVal">—</span>
            </div>
            <div class="bar-track"><div class="bar-fill bar-lstm" id="lstmBar" style="width:0%"></div></div>
          </div>
          <div class="score-row">
            <div class="score-row-head">
              <span class="score-name">IsoForest Score</span>
              <span class="score-val" id="isoVal">—</span>
            </div>
            <div class="bar-track"><div class="bar-fill bar-iso" id="isoBar" style="width:0%"></div></div>
          </div>
          <div class="score-row">
            <div class="score-row-head">
              <span class="score-name">Hybrid Score</span>
              <span class="score-val" id="hybVal">—</span>
            </div>
            <div class="bar-track"><div class="bar-fill bar-hybrid" id="hybBar" style="width:0%"></div></div>
          </div>
        </div>
        <div class="meta-grid">
          <div class="meta-item">
            <span class="meta-key">Amount</span>
            <span class="meta-val" id="metaAmt">—</span>
          </div>
          <div class="meta-item">
            <span class="meta-key">Category</span>
            <span class="meta-val" id="metaCat">—</span>
          </div>
          <div class="meta-item">
            <span class="meta-key">Time</span>
            <span class="meta-val" id="metaTime">—</span>
          </div>
          <div class="meta-item">
            <span class="meta-key">State</span>
            <span class="meta-val" id="metaState">—</span>
          </div>
        </div>
      </div>
    </div>

    <!-- History -->
    <div class="history-card">
      <div class="section-label">Session History</div>
      <h3>Recent Analyses</h3>
      <div class="history-list" id="historyList">
        <div class="empty-hist" id="emptyHist">No transactions analysed yet</div>
      </div>
    </div>
  </div>
</main>

<script>
const history = [];

async function predict() {
  const btn = document.getElementById('predictBtn');
  const spinner = document.getElementById('spinner');
  const btnText = document.getElementById('btnText');
  const errMsg = document.getElementById('errMsg');

  // Collect form data
  const payload = {
    datetime:   document.getElementById('datetime').value,
    amount:     document.getElementById('amount').value,
    category:   document.getElementById('category').value,
    state:      document.getElementById('state').value,
    city_pop:   document.getElementById('city_pop').value,
    user_id:    document.getElementById('user_id').value,
    lat:        document.getElementById('lat').value,
    long:       document.getElementById('long').value,
    merch_lat:  document.getElementById('merch_lat').value,
    merch_long: document.getElementById('merch_long').value,
  };

  if (!payload.amount || !payload.datetime) {
    showError('Please fill in Amount and Date/Time.');
    return;
  }

  errMsg.style.display = 'none';
  btn.disabled = true;
  spinner.style.display = 'block';
  btnText.textContent = 'Analysing…';

  try {
    const res = await fetch('/predict', {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify(payload)
    });
    const data = await res.json();
    if (data.error) { showError(data.error); return; }
    showResult(data, payload);
  } catch(e) {
    showError('Connection error: ' + e.message);
  } finally {
    btn.disabled = false;
    spinner.style.display = 'none';
    btnText.textContent = 'Analyse Transaction';
  }
}

function showResult(data, payload) {
  const card  = document.getElementById('resultCard');
  const ph    = document.getElementById('placeholder');
  const vd    = document.getElementById('verdictDiv');
  const hybrid = data.hybrid_score;

  let cls, icon, title, sub;
  if (hybrid >= 0.7) {
    cls='fraud'; icon='🚨'; title='FRAUD DETECTED';
    sub='High probability of fraudulent activity';
  } else if (hybrid >= 0.4) {
    cls='warn'; icon='⚠️'; title='SUSPICIOUS';
    sub='Elevated risk — manual review recommended';
  } else {
    cls='legit'; icon='✅'; title='LEGITIMATE';
    sub='Transaction appears normal';
  }

  card.className = 'result-card ' + cls;
  document.getElementById('verdictIcon').className = 'verdict-icon ' + cls;
  document.getElementById('verdictIcon').textContent = icon;
  document.getElementById('verdictTitle').className  = 'verdict-title ' + cls;
  document.getElementById('verdictTitle').textContent = title;
  document.getElementById('verdictSub').textContent   = sub;

  // Scores
  const lstm = data.lstm_score, iso = data.iso_score, hyb = data.hybrid_score;
  document.getElementById('lstmVal').textContent = (lstm*100).toFixed(1)+'%';
  document.getElementById('isoVal').textContent  = (iso*100).toFixed(1)+'%';
  document.getElementById('hybVal').textContent  = (hyb*100).toFixed(1)+'%';

  setTimeout(()=>{
    document.getElementById('lstmBar').style.width = (lstm*100)+'%';
    document.getElementById('isoBar').style.width  = (iso*100)+'%';
    document.getElementById('hybBar').style.width  = (hyb*100)+'%';
  }, 50);

  // Meta
  const dt = new Date(payload.datetime);
  document.getElementById('metaAmt').textContent   = '$'+parseFloat(payload.amount).toFixed(2);
  document.getElementById('metaCat').textContent   = payload.category.replace(/_/g,' ');
  document.getElementById('metaTime').textContent  = dt.toLocaleTimeString('en-US',{hour:'2-digit',minute:'2-digit'});
  document.getElementById('metaState').textContent = payload.state;

  ph.style.display = 'none';
  vd.style.display = 'flex';

  // Add to history
  addHistory(payload, cls, hyb);
}

function addHistory(payload, cls, score) {
  const list  = document.getElementById('historyList');
  const empty = document.getElementById('emptyHist');
  if (empty) empty.remove();

  const dt   = new Date(payload.datetime);
  const time = dt.toLocaleTimeString('en-US',{hour:'2-digit',minute:'2-digit'});
  const item = document.createElement('div');
  item.className = 'hist-item';
  item.innerHTML = `
    <div class="hist-dot ${cls}"></div>
    <div class="hist-info">
      <div class="hist-name">$${parseFloat(payload.amount).toFixed(2)} — ${payload.category.replace(/_/g,' ')}</div>
      <div class="hist-time">${time} · ${payload.state}</div>
    </div>
    <div class="hist-score ${cls}">${(score*100).toFixed(1)}%</div>
  `;
  list.insertBefore(item, list.firstChild);
}

function showError(msg) {
  const e = document.getElementById('errMsg');
  e.textContent = '⚠ ' + msg;
  e.style.display = 'block';
}
</script>
</body>
</html>
"""

# ─────────────────────────── ROUTES ─────────────────────────────────────────
@app.route("/")
def index():
    from flask import render_template_string
    return render_template_string(HTML,
        models_ok=models_loaded,
        error_msg=load_error[:60] if load_error else "",
        states=STATES
    )


@app.route("/predict", methods=["POST"])
def predict():
    if not models_loaded:
        return jsonify({"error": "Models not loaded: " + load_error})

    data = request.get_json()
    try:
        feat = build_feature_vector(data)

        # Build a sequence of 10 (pad with this transaction repeated)
        seq = np.tile(feat, (10, 1)).astype(np.float32)   # (10, 12)
        seq_batch = seq[np.newaxis, ...]                    # (1, 10, 12)

        import tensorflow as tf
        lstm_score = float(lstm_model.predict(seq_batch, verbose=0)[0][0])

        # IsoForest on single feature vector
        iso_raw   = iso_model.decision_function(feat.reshape(1, -1))[0]
        iso_score = float(np.clip(0.5 - iso_raw * 0.5, 0, 1))

        hybrid = 0.7 * lstm_score + 0.3 * iso_score

        return jsonify({
            "lstm_score":   round(lstm_score, 4),
            "iso_score":    round(iso_score, 4),
            "hybrid_score": round(hybrid, 4),
        })

    except Exception as e:
        import traceback
        return jsonify({"error": str(e), "trace": traceback.format_exc()})


if __name__ == "__main__":
    print("Loading models...")
    load_models()
    print("Starting FraudShield on http://localhost:5000")
    app.run(debug=False, port=5000)