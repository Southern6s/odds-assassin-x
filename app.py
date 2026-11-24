from flask import Flask, jsonify, render_template_string, request
import requests

app = Flask(__name__)
BASE = "https://external-api.kalshi.com/trade-api/v2"

PAGE = r"""
<!doctype html>
<html lang="en">
<head>
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ODDS ASSASSIN X</title>
<style>
:root{color-scheme:dark}*{box-sizing:border-box}
body{margin:0;background:#0b1020;color:#eef2ff;font:15px -apple-system,BlinkMacSystemFont,Segoe UI,sans-serif}
header{padding:22px 18px;background:#121a30;border-bottom:1px solid #26324d}
h1{margin:0;font-size:23px;letter-spacing:1px}header p{color:#9aa8c7;margin:7px 0 0}
main{padding:16px;max-width:1100px;margin:auto}
.notice{background:#28213a;border:1px solid #5d477e;padding:12px;border-radius:12px;color:#e5d8ff;font-size:13px}
.controls{display:flex;gap:8px;margin:14px 0;flex-wrap:wrap}
input,select,button{border:1px solid #34415f;background:#141e35;color:#fff;border-radius:9px;padding:11px;font-size:14px}
button{background:#315de8;border:0;font-weight:700}
.statgrid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:10px;margin:14px 0}
.stat{background:#141e35;padding:13px;border:1px solid #26324d;border-radius:12px}
.stat b{display:block;font-size:21px;margin-top:5px}.muted{color:#9aa8c7;font-size:12px}
.market{background:#141e35;border:1px solid #26324d;border-radius:12px;padding:13px;margin:9px 0}
.market h3{font-size:15px;margin:0 0 8px}.prices{display:flex;gap:14px;flex-wrap:wrap;font-size:13px}
.tag{font-size:11px;color:#f5c978}.empty{color:#aab5ce;padding:20px;text-align:center}
footer{padding:20px;color:#8492b2;font-size:12px;text-align:center}
@media(min-width:650px){.statgrid{grid-template-columns:repeat(4,minmax(0,1fr))}}
</style>
</head>
<body>
<header><h1>ODDS ASSASSIN X</h1><p>Quantitative market research · Kalshi · v0.1</p></header>
<main>
<div class="notice"><b>PAPER / RESEARCH MODE ONLY.</b> This dashboard reads public market data. It does not connect to your account, predict winners, or place orders. Market prices are not independent forecasts.</div>
<div class="statgrid">
 <div class="stat"><span class="muted">Markets loaded</span><b id="count">—</b></div>
 <div class="stat"><span class="muted">Data status</span><b id="status" style="font-size:16px">Waiting</b></div>
 <div class="stat"><span class="muted">Live orders</span><b style="font-size:16px;color:#f5c978">DISABLED</b></div>
 <div class="stat"><span class="muted">Model edge</span><b style="font-size:16px">Not calculated</b></div>
</div>
<div class="controls">
 <input id="search" placeholder="Search markets…" oninput="filterMarkets()">
 <button onclick="loadMarkets()">Refresh markets</button>
</div>
<div id="markets"><div class="empty">Tap “Refresh markets” to load current public listings.</div></div>
</main>
<footer>ODDS ASSASSIN X · Free prototype · Verify all market rules and prices at Kalshi before making decisions.</footer>
<script>
let allMarkets=[];
function money(v){if(v===null||v===undefined||v==="")return "—";let n=Number(v);return Number.isFinite(n)?(n>1?(n/100).toFixed(2):n.toFixed(2)):"—";}
function render(items){
 const host=document.getElementById('markets');
 document.getElementById('count').textContent=items.length;
 if(!items.length){host.innerHTML='<div class="empty">No matching markets returned.</div>';return;}
 host.innerHTML=items.map(m=>`<article class="market">
 <h3>${esc(m.title||m.ticker||"Untitled market")}</h3>
 <div class="muted">${esc(m.ticker||"")} · ${esc(m.category||"Uncategorized")} · ${esc(m.status||"")}</div>
 <div class="prices" style="margin-top:10px">
 <span>YES bid <b>${money(m.yes_bid_dollars??m.yes_bid)}</b></span>
 <span>YES ask <b>${money(m.yes_ask_dollars??m.yes_ask)}</b></span>
 <span>NO bid <b>${money(m.no_bid_dollars??m.no_bid)}</b></span>
 <span>NO ask <b>${money(m.no_ask_dollars??m.no_ask)}</b></span>
 </div>
 <div class="muted" style="margin-top:9px">24h volume: ${esc(String(m.volume_24h_fp??m.volume_24h??"—"))} · Close: ${esc(m.close_time||"—")}</div>
 <div class="tag" style="margin-top:8px">No independent probability / no edge signal</div>
 </article>`).join('');
}
function esc(s){return String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));}
function filterMarkets(){let q=document.getElementById('search').value.toLowerCase();render(allMarkets.filter(m=>(m.title+" "+m.ticker+" "+m.category).toLowerCase().includes(q)));}
async function loadMarkets(){
 const status=document.getElementById('status');status.textContent="Loading…";
 try{const r=await fetch('/api/markets?limit=100');if(!r.ok)throw new Error("API request failed ("+r.status+")");const d=await r.json();allMarkets=d.markets||[];status.textContent="Connected";filterMarkets();}
 catch(e){status.textContent="Unavailable";document.getElementById('markets').innerHTML='<div class="empty">Could not load public market data. Try again later. '+esc(e.message)+'</div>';}
}
</script>
</body></html>
"""

@app.get("/")
def home():
    return render_template_string(PAGE)

@app.get("/health")
def health():
    return jsonify({"ok": True, "mode": "read-only"})

@app.get("/api/markets")
def markets():
    try:
        limit = max(1, min(int(request.args.get("limit", "100")), 200))
        r = requests.get(
            f"{BASE}/markets",
            params={"status": "open", "limit": limit},
            headers={"Accept": "application/json", "User-Agent": "OddsAssassinX/0.1"},
            timeout=15,
        )
        r.raise_for_status()
        payload = r.json()
        return jsonify({"markets": payload.get("markets", []), "cursor": payload.get("cursor")})
    except requests.RequestException as e:
        return jsonify({"error": "Kalshi public market-data request failed", "detail": str(e)}), 502
    except (ValueError, TypeError):
        return jsonify({"error": "Invalid limit"}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
