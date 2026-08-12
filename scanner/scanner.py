#!/usr/bin/env python3
"""
SwingSense AI - scanner engine v2 (event-driven).

v2 upgrade: instead of ranking on filters alone, every stock gets a
100-point Breakout Probability Score built from the pre-breakout evidence
professionals look for: proximity to 52-week high, consolidation (tight
range + falling ATR + Bollinger squeeze), volume dry-up before expansion,
EMA alignment, ADX, relative strength vs the market, sector strength,
delivery, accumulation (OBV/CMF) and simple candle patterns.

Pipeline: fetch NSE data -> indicators -> breakout scoring -> hard filters
          -> rank -> AI reasoning -> write data.json -> review open positions

Spawned by server.js. Config lives in ../data.json (editable in Settings).
All stdout ASCII (Windows safe), all file writes UTF-8.
Env: GEMINI_API_KEY (optional), SWINGSENSE_SKIP_FETCH=1 (testing).
"""
import io
import json
import os
import sqlite3
import sys
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import requests

HERE = Path(__file__).resolve().parent
DATA_JSON = HERE.parent / "data.json"
DB_PATH = HERE / "market.db"
CACHE_DIR = HERE / "cache"
LOOKBACK_DAYS = 260                    # ~1 year, enables true 52-week high
MAX_HOLD_DAYS = 10                     # fallback for old records

# horizon definitions live in the scoring section below

# NSE moves its archive hosts from time to time - try these in order.
UNIVERSE_URLS = [
    "https://nsearchives.nseindia.com/content/indices/ind_nifty500list.csv",
    "https://archives.nseindia.com/content/indices/ind_nifty500list.csv",
    "https://www.niftyindices.com/IndexConstituent/ind_nifty500list.csv",
]
BHAV_HOSTS = ["nsearchives.nseindia.com", "archives.nseindia.com"]
BHAV_PATH = "https://{host}/products/content/sec_bhavdata_full_{d}.csv"
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                         "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
           "Referer": "https://www.nseindia.com/"}

DEFAULT_CFG = dict(
    # core hard filters
    minPrice=30, maxPrice=1200, minVolumeMultiplier=1.5,
    minRsi=50, maxRsi=70, trendEma=True,
    # breakout model
    minBreakoutScore=70,          # A-grade needs score >= this (80 = very strict)
    consolidationDays=15,         # window for tight-range detection
    resistanceLookback=20,        # days for resistance = highest high
    maxDistToBreakoutPct=3.0,     # "near breakout" if within this % of resistance
    requireEma200=False,          # optionally demand EMA50 > EMA200 too
    minAdx=0,                     # optionally demand ADX >= this (0 = off)
    minRsRank=60,                 # only stocks outperforming 60%+ of NIFTY 500
    minAvgVolume=500000,          # absolute liquidity floor: 20d avg shares/day
    maxOpenPositions=4,           # hard cap on concurrent A-grade tracked trades
    maxDrawdownFrom20dHigh=0,     # skip if fallen more than this % from 20d high (0 = off)
    # risk
    minRiskReward=1.2,            # A-grade needs T1 risk:reward >= this
    # minimum potential move (entry -> Target 2) required per horizon, in %
    minMovePct={"2d": 3.0, "1w": 5.0, "2w": 7.0, "1m": 12.0},
    # position sizing (used by the app to compute rupee amounts per card)
    capital=100000, riskPerTradePct=1.0,
    # output
    maxRecommendations=10, fillWithNearMisses=True,
)


def log(msg):
    print(msg, flush=True)


# ------------------------------------------------------------- live preview
IST_OFFSET = dt.timedelta(hours=5, minutes=30)

def ist_now():
    return dt.datetime.now(dt.timezone.utc).replace(tzinfo=None) + IST_OFFSET


def fetch_upcoming_board_meetings(symbols, days_ahead=5):
    """Real, forward-looking company events - NSE's own public board-meetings
    listing, which is how Indian companies formally announce their results
    date in advance. NOT tested against a live response (this sandbox has
    no NSE network access) - built to fail gracefully to {} on ANY problem
    (wrong response shape, blocked request, endpoint moved), exactly like
    fetch_nifty_snapshot and fetch_recent_news already do. First real test
    of whether NSE actually answers this endpoint happens on your machine,
    not here - if it breaks, the symptom will be an empty result, not a
    crash, and the rest of the scan is unaffected either way.

    Returns: {symbol: {"date": "YYYY-MM-DD", "purpose": "..."}} for any
    cached symbol with a real board meeting in the next `days_ahead` days.
    """
    today = dt.date.today()
    to_date = today + dt.timedelta(days=days_ahead)
    url = ("https://www.nseindia.com/api/corporate-board-meetings"
           f"?index=equities&from_date={today.strftime('%d-%m-%Y')}"
           f"&to_date={to_date.strftime('%d-%m-%Y')}")
    try:
        with requests.Session() as s:
            s.headers.update(HEADERS)
            try:
                s.get("https://www.nseindia.com", timeout=15)  # primes cookies, same as bhavcopy fetch
            except requests.RequestException:
                pass
            resp = s.get(url, timeout=15)
            if resp.status_code != 200:
                log(f"  ! board meetings fetch returned {resp.status_code} - skipping")
                return {}
            data = resp.json()
        wanted = set(symbols)
        out = {}
        for row in data if isinstance(data, list) else []:
            sym = str(row.get("symbol", "")).strip()
            if sym not in wanted:
                continue
            date_raw = row.get("bm_date") or row.get("bm_timestamp") or ""
            purpose = str(row.get("bm_purpose", "")).strip() or "Board meeting"
            try:
                parsed = dt.datetime.strptime(date_raw.split()[0], "%d-%b-%Y").date()
                iso_date = parsed.isoformat()
            except Exception:
                iso_date = None
            if iso_date:
                out[sym] = {"date": iso_date, "purpose": purpose}
        return out
    except Exception as e:
        log(f"  ! board meetings fetch failed ({type(e).__name__}) - "
            f"continuing without it, this is a bonus signal not a dependency")
        return {}


def fetch_recent_news(company_name, symbol, max_items=3, max_age_days=10):
    """Real recent news headlines for one stock, via Google News RSS - no
    API key, no fabrication. Only called for the handful of stocks that
    actually made today's final recommendation list, never the full
    universe, so this stays cheap even though it's a live web fetch per
    stock. Returns [] on any failure - news is a bonus, never blocks a scan."""
    import xml.etree.ElementTree as ET
    from urllib.parse import quote
    try:
        query = quote(f'"{company_name}" OR {symbol} share NSE')
        url = f"https://news.google.com/rss/search?q={query}&hl=en-IN&gl=IN&ceid=IN:en"
        resp = requests.get(url, headers=HEADERS, timeout=8)
        if resp.status_code != 200:
            return []
        root = ET.fromstring(resp.content)
        cutoff = dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=max_age_days)
        items = []
        for item in root.findall(".//item"):
            title = (item.findtext("title") or "").strip()
            link = (item.findtext("link") or "").strip()
            pub = (item.findtext("pubDate") or "").strip()
            src = item.find("source")
            source = src.text.strip() if src is not None and src.text else ""
            if not title or not link:
                continue
            try:
                pub_dt = dt.datetime.strptime(pub, "%a, %d %b %Y %H:%M:%S %Z").replace(tzinfo=dt.timezone.utc)
                if pub_dt < cutoff:
                    continue
                pub_date = pub_dt.date().isoformat()
            except Exception:
                pub_date = None
            # strip the " - Source Name" suffix Google News appends to titles
            if source and title.endswith(f" - {source}"):
                title = title[:-(len(source) + 3)]
            items.append({"title": title, "link": link, "source": source, "date": pub_date})
            if len(items) >= max_items:
                break
        return items
    except Exception as e:
        log(f"  ! news fetch failed for {symbol} ({type(e).__name__})")
        return []


def fetch_nifty_snapshot():
    """Current Nifty 50 level, today's change, and a plain-English read of
    where it sits vs its own real EMAs - genuinely current data, not a
    forecast. No invented probability anywhere in this function."""
    try:
        import yfinance as yf
    except ImportError:
        return None
    try:
        hist = yf.Ticker("^NSEI").history(period="4mo", interval="1d")
        if hist is None or len(hist) < 55:
            return None
        closes = hist["Close"].dropna()
        level = float(closes.iloc[-1])
        prev = float(closes.iloc[-2])
        chg_pts = round(level - prev, 2)
        chg_pct = round(chg_pts / prev * 100, 2)
        ema20 = float(closes.ewm(span=20, adjust=False).mean().iloc[-1])
        ema50 = float(closes.ewm(span=50, adjust=False).mean().iloc[-1])
        above20 = level > ema20
        above50 = level > ema50
        # plain description of REAL current state, not a prediction of
        # tomorrow - explicitly framed as "what today looks like", never
        # as a probability or forecast
        if above20 and above50:
            read = "Trading above both its 20-day and 50-day average - the near-term trend is constructive."
        elif above20 and not above50:
            read = "Above its 20-day average but still under the 50-day - a recovery still proving itself."
        elif not above20 and above50:
            read = "Slipped below its 20-day average while still above the 50-day - short-term wobble inside a longer uptrend."
        else:
            read = "Below both its 20-day and 50-day average - the near-term trend is under pressure."
        return dict(level=round(level, 2), changePts=chg_pts, changePct=chg_pct,
                    aboveEma20=above20, aboveEma50=above50, technicalRead=read)
    except Exception as e:
        log(f"  ! Nifty 50 fetch failed ({type(e).__name__})")
        return None


def next_fo_expiry(from_date=None):
    """F&O monthly expiry: last Thursday of the month. Pure calendar math,
    no data fetch, cannot be wrong or stale."""
    d = from_date or dt.date.today()
    def last_thursday(year, month):
        if month == 12:
            nxt = dt.date(year + 1, 1, 1)
        else:
            nxt = dt.date(year, month + 1, 1)
        day = nxt - dt.timedelta(days=1)
        while day.weekday() != 3:  # Thursday = 3
            day -= dt.timedelta(days=1)
        return day
    this_month_expiry = last_thursday(d.year, d.month)
    if this_month_expiry >= d:
        return this_month_expiry
    nm = d.month + 1 if d.month < 12 else 1
    ny = d.year if d.month < 12 else d.year + 1
    return last_thursday(ny, nm)


def fetch_current_quotes(symbols):
    """{symbol: dict(open,high,low,close,volume,asOf)} for TODAY, from the
    most recent 1-minute bar available. Works the same way whether the
    market is open (that bar is the live/current price, ~15min delayed)
    or closed (that bar is the day's final close, which by definition
    won't change again). Symbols with no bar dated today are skipped -
    that's a per-symbol data gap, not treated as a global failure."""
    try:
        import yfinance as yf
    except ImportError:
        log("  ! yfinance not installed - run: pip install yfinance")
        return {}
    out, skipped = {}, 0
    today = dt.date.today()
    syms = sorted(symbols)
    try:
        for i in range(0, len(syms), 60):
            batch = [f"{x}.NS" for x in syms[i:i+60]]
            df = yf.download(batch, period="1d", interval="1m",
                             progress=False, threads=True, group_by="ticker")
            for x in syms[i:i+60]:
                t = f"{x}.NS"
                try:
                    row = (df[t] if len(batch) > 1 else df).dropna(how="all")
                    if row.empty:
                        skipped += 1; continue
                    bar_time = row.index[-1]
                    bar_ist = (bar_time.tz_convert("Asia/Kolkata") if bar_time.tzinfo
                              else bar_time.tz_localize("UTC").tz_convert("Asia/Kolkata"))
                    if bar_ist.date() != today:
                        skipped += 1; continue        # no data for today yet
                    out[x] = dict(
                        open=float(row["Open"].dropna().iloc[0]),
                        high=float(row["High"].max()), low=float(row["Low"].min()),
                        close=float(row["Close"].iloc[-1]),
                        volume=float(row["Volume"].sum()),
                        asOf=bar_ist.strftime("%H:%M"))
                except Exception:
                    skipped += 1; continue
    except Exception as e:
        log(f"  ! quote fetch failed ({type(e).__name__})")
        return {}
    if skipped:
        log(f"  ! {skipped} symbol(s) had no data for today - kept prior close")
    return out


def overlay_today(hist, universe):
    """Always overlay TODAY's current price for every symbol we can get one
    for - the app's whole point per your requirement: every scan, the
    latest available price, no waiting on any end-of-day file. Volume is
    projected to a fair full-day comparison while the market is still
    open; once closed, today's volume-so-far already IS the day's total."""
    quotes = fetch_current_quotes(set(universe) & set(hist["symbol"].unique()))
    if not quotes:
        log("  ! could not fetch current prices - showing last available close")
        return hist, None
    today = dt.date.today().isoformat()
    now = ist_now()
    mkt_open = now.replace(hour=9, minute=15, second=0, microsecond=0)
    mkt_close = now.replace(hour=15, minute=30, second=0, microsecond=0)
    if mkt_open <= now <= mkt_close:
        frac = max((now - mkt_open).total_seconds() /
                   (mkt_close - mkt_open).total_seconds(), 0.02)
        proj = min(1.0 / frac, 4.0)
    else:
        proj = 1.0                                    # session over - volume is final
    rows = [dict(symbol=sym, date=today, open=q["open"], high=q["high"],
                low=q["low"], close=q["close"], volume=int(q["volume"] * proj))
            for sym, q in quotes.items()]
    live_df = pd.DataFrame(rows)
    hist = hist[hist["date"] != today]
    hist = pd.concat([hist, live_df], ignore_index=True).sort_values(
        ["symbol", "date"]).reset_index(drop=True)
    sample_time = next(iter(quotes.values()))["asOf"]
    log(f"  {len(rows)} symbols updated to current price, latest tick "
        f"~{sample_time} IST")
    return hist, sample_time


# ---------------------------------------------------------------- data.json
def load_backtest_results():
    """Read the pooled, whole-market evidence saved by backtest.py, if it
    exists. Returns None if you haven't run a backtest yet - this is
    optional evidence layered on top of the per-stock historical replay,
    never a dependency of a normal scan."""
    path = HERE / "backtest_results.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def read_db_json():
    return json.loads(DATA_JSON.read_text(encoding="utf-8"))


def write_db_json(obj):
    """Atomic write: write to a temp file first, then os.replace() into
    place. The OS guarantees replace() either fully succeeds or fully
    fails - never leaves data.json half-written if the process dies
    mid-save (power loss, a killed scan, etc)."""
    tmp = DATA_JSON.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, DATA_JSON)


# ---------------------------------------------------------------- fetching
def _sql():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""CREATE TABLE IF NOT EXISTS prices (
        symbol TEXT, date TEXT, open REAL, high REAL, low REAL, close REAL,
        volume INTEGER, delivery_pct REAL, PRIMARY KEY (symbol, date))""")
    return conn


def load_universe():
    CACHE_DIR.mkdir(exist_ok=True)
    cache = CACHE_DIR / "nifty500.csv"
    last_err = None
    for url in UNIVERSE_URLS:
        try:
            r = requests.get(url, headers=HEADERS, timeout=30)
            r.raise_for_status()
            if b"Symbol" not in r.content[:2000]:
                raise ValueError("unexpected content")
            cache.write_bytes(r.content)
            log(f"  universe list from {url.split('/')[2]}")
            last_err = None
            break
        except Exception as e:
            last_err = e
    if last_err is not None:
        if not cache.exists():
            raise RuntimeError(
                "Cannot fetch the NIFTY 500 list from any known NSE host "
                f"(last error: {last_err}). Check your internet/DNS/firewall, "
                "or if NSE moved again, update UNIVERSE_URLS in scanner.py")
        log(f"  universe fetch failed on all hosts, using cached list")
    df = pd.read_csv(cache)
    ind_col = next((c for c in df.columns if "industry" in c.lower()), None)
    meta = {}
    for _, r in df.iterrows():
        meta[str(r["Symbol"]).strip()] = {
            "companyName": str(r.get("Company Name", "")).strip(),
            "sector": str(r[ind_col]).strip() if ind_col else "",
        }
    return meta


_bhav_host = None            # locked to the first host that answers

def fetch_bhavcopy(session, date):
    global _bhav_host
    ddmmyyyy = date.strftime("%d%m%Y")
    cache = CACHE_DIR / f"bhav_{ddmmyyyy}.csv"
    if cache.exists():
        raw = cache.read_bytes()
    else:
        hosts = [_bhav_host] if _bhav_host else BHAV_HOSTS
        raw = None
        for host in hosts:
            try:
                r = session.get(BHAV_PATH.format(host=host, d=ddmmyyyy),
                                timeout=30)
                if r.status_code == 200 and b"SYMBOL" in r.content[:200]:
                    raw = r.content
                    if _bhav_host != host:
                        _bhav_host = host
                        log(f"  using bhavcopy host: {host}")
                    break
            except requests.RequestException:
                continue
        if raw is None:
            return None
        cache.write_bytes(raw)
    df = pd.read_csv(io.BytesIO(raw))
    df.columns = [c.strip() for c in df.columns]
    df = df[df["SERIES"].str.strip() == "EQ"]
    out = pd.DataFrame({
        "symbol": df["SYMBOL"].str.strip(),
        "open": pd.to_numeric(df["OPEN_PRICE"], errors="coerce"),
        "high": pd.to_numeric(df["HIGH_PRICE"], errors="coerce"),
        "low": pd.to_numeric(df["LOW_PRICE"], errors="coerce"),
        "close": pd.to_numeric(df["CLOSE_PRICE"], errors="coerce"),
        "volume": pd.to_numeric(df["TTL_TRD_QNTY"], errors="coerce"),
        "delivery_pct": pd.to_numeric(df["DELIV_PER"], errors="coerce"),
    })
    out["date"] = date.isoformat()
    return out.dropna(subset=["close"])


def sync_history(universe):
    conn = _sql()
    if os.environ.get("SWINGSENSE_SKIP_FETCH") == "1":
        log("  SWINGSENSE_SKIP_FETCH=1 -> using cached history only")
        return conn
    have = {r[0] for r in conn.execute("SELECT DISTINCT date FROM prices")}
    added, checked = 0, 0
    d = dt.date.today()
    with requests.Session() as s:
        s.headers.update(HEADERS)
        try:
            s.get("https://www.nseindia.com", timeout=15)
        except requests.RequestException:
            pass
        while checked < LOOKBACK_DAYS:
            if d.weekday() < 5:
                checked += 1
                if d.isoformat() not in have:
                    df = fetch_bhavcopy(s, d)
                    if df is not None:
                        df = df[df["symbol"].isin(universe)]
                        df.to_sql("prices", conn, if_exists="append",
                                  index=False, method="multi", chunksize=500)
                        conn.commit()
                        added += 1
                        if added % 20 == 0 or added < 5:
                            log(f"  + {d} : {len(df)} symbols")
                    elif d == dt.date.today():
                        log(f"  ! today's ({d}) bhavcopy not published by NSE yet - "
                            f"this usually appears ~6:30-7:30 PM IST. "
                            f"Prices below are from the last available session.")
            d -= dt.timedelta(days=1)
    log(f"  history sync complete ({added} new day(s))")
    return conn


# ---------------------------------------------------------------- indicators
def ema(s, n): return s.ewm(span=n, adjust=False).mean()


def rsi(close, n=14):
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1/n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1/n, adjust=False).mean()
    rs = up / dn.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def williams_r(df, n=14):
    """Standard Williams %R: where today's close sits within the recent
    high-low range, scaled -100 (at the low) to 0 (at the high)."""
    hh = df["high"].rolling(n).max()
    ll = df["low"].rolling(n).min()
    rng = (hh - ll).replace(0, np.nan)
    return -100 * (hh - df["close"]) / rng


def atr_series(df, n=14):
    pc = df["close"].shift(1)
    tr = pd.concat([df["high"]-df["low"], (df["high"]-pc).abs(),
                    (df["low"]-pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/n, adjust=False).mean()


def adx(df, n=14):
    up = df["high"].diff()
    dn = -df["low"].diff()
    plus_dm = np.where((up > dn) & (up > 0), up, 0.0)
    minus_dm = np.where((dn > up) & (dn > 0), dn, 0.0)
    tr = pd.concat([df["high"]-df["low"],
                    (df["high"]-df["close"].shift()).abs(),
                    (df["low"]-df["close"].shift()).abs()], axis=1).max(axis=1)
    atr_ = tr.ewm(alpha=1/n, adjust=False).mean()
    pdi = 100 * pd.Series(plus_dm, index=df.index).ewm(alpha=1/n, adjust=False).mean() / atr_
    mdi = 100 * pd.Series(minus_dm, index=df.index).ewm(alpha=1/n, adjust=False).mean() / atr_
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    return dx.ewm(alpha=1/n, adjust=False).mean()


def snapshot(df):
    """Latest per-symbol state, including all pre-breakout evidence."""
    if len(df) < 60:
        return None
    df = df.reset_index(drop=True)
    n = len(df)
    close, high, low, vol = df["close"], df["high"], df["low"], df["volume"]

    e20, e50 = ema(close, 20), ema(close, 50)
    e200 = ema(close, 200) if n >= 200 else e50
    macd_l = ema(close, 12) - ema(close, 26)
    macd_s = macd_l.ewm(span=9, adjust=False).mean()
    hist = macd_l - macd_s
    r = rsi(close)
    wr = williams_r(df)
    a = atr_series(df)
    ax = adx(df)
    avg_vol = vol.rolling(20).mean()

    # Bollinger bandwidth + its percentile over the last 120 bars (squeeze)
    mid = close.rolling(20).mean()
    sd = close.rolling(20).std()
    bw = ((mid + 2*sd) - (mid - 2*sd)) / mid
    bw_hist = bw.dropna().tail(120)
    bw_pctile = float((bw_hist <= bw_hist.iloc[-1]).mean() * 100) if len(bw_hist) > 20 else 50.0

    # OBV and CMF (accumulation)
    obv = (np.sign(close.diff().fillna(0)) * vol).cumsum()
    rng = (high - low).replace(0, np.nan)
    mfm = ((close - low) - (high - close)) / rng
    cmf = float((mfm * vol).rolling(20).sum().iloc[-1] /
                max(vol.rolling(20).sum().iloc[-1], 1))

    # candle patterns (yesterday -> today)
    o1, c1, h1, l1 = df["open"].iloc[-2], close.iloc[-2], high.iloc[-2], low.iloc[-2]
    o0, c0, h0, l0 = df["open"].iloc[-1], close.iloc[-1], high.iloc[-1], low.iloc[-1]
    ranges7 = (high - low).tail(7)
    patterns = []
    if (h0 - l0) == ranges7.min(): patterns.append("NR7")
    if h0 < h1 and l0 > l1: patterns.append("Inside bar")
    if c0 > o0 and c1 < o1 and c0 >= o1 and o0 <= c1: patterns.append("Bullish engulfing")

    i = n - 1
    look52 = min(n - 1, 250)
    hi52 = float(high.iloc[-look52-1:-1].max())
    ret20 = float(close.pct_change(20).iloc[-1] * 100) if n > 20 else 0.0
    ret63 = float(close.pct_change(63).iloc[-1] * 100) if n > 63 else None    # ~3 months
    ret126 = float(close.pct_change(126).iloc[-1] * 100) if n > 126 else None  # ~6 months
    ret63 = float(close.pct_change(min(63, n-1)).iloc[-1] * 100)
    ret126 = float(close.pct_change(min(126, n-1)).iloc[-1] * 100)
    ret252 = float(close.pct_change(min(252, n-1)).iloc[-1] * 100)

    # close location value: where in today's range did we close
    clv = float((c0 - l0) / max(h0 - l0, 1e-9))
    # today's volume percentile vs last 120 sessions
    v120 = vol.tail(120)
    vol_pctile = float((v120 <= vol.iloc[-1]).mean() * 100)
    # opening gap vs yesterday's close
    gap_pct = float((o0 - c1) / max(c1, 1e-9) * 100)
    # range expansion day (true range above ATR = trend igniting)
    tr_today = max(h0 - l0, abs(h0 - c1), abs(l0 - c1))
    tr_expand = bool(tr_today > a.iloc[-1])
    # weekly timeframe alignment
    weekly_up = False
    try:
        wk = df.set_index(pd.to_datetime(df["date"]))["close"].resample("W-FRI").last().dropna()
        if len(wk) >= 30:
            we10 = wk.ewm(span=10, adjust=False).mean()
            we30 = wk.ewm(span=30, adjust=False).mean()
            weekly_up = bool(we10.iloc[-1] > we30.iloc[-1] and wk.iloc[-1] > we10.iloc[-1])
    except Exception:
        pass
    # anchored VWAP from the lowest low of the last 120 sessions
    above_avwap = False
    try:
        a_idx = int(low.tail(120).idxmin())
        tp = (high + low + close) / 3
        seg_tp, seg_v = tp.iloc[a_idx:], vol.iloc[a_idx:]
        avwap = float((seg_tp * seg_v).sum() / max(seg_v.sum(), 1))
        above_avwap = bool(close.iloc[-1] > avwap)
    except Exception:
        pass
    # swing structure: higher highs AND higher lows (5-bar pivots, last 90 bars)
    hhhl = False
    try:
        hh, ll = high.tail(90).reset_index(drop=True), low.tail(90).reset_index(drop=True)
        ph = [hh[j] for j in range(5, len(hh)-5)
              if hh[j] == hh[j-5:j+6].max()]
        pl = [ll[j] for j in range(5, len(ll)-5)
              if ll[j] == ll[j-5:j+6].min()]
        hhhl = bool(len(ph) >= 2 and len(pl) >= 2
                    and ph[-1] > ph[-2] and pl[-1] > pl[-2])
    except Exception:
        pass
    pivot20 = float(high.iloc[-21:-1].max())
    # resistance test count AND real rejection magnitude at each test -
    # a level tested 5x with rejections shrinking from 3% to 0.4% is a very
    # different story than 5x all rejecting hard at 3%, and that's only
    # visible if we measure the actual pullback after each touch, not just
    # count the touches.
    res_tests = 0
    in_touch = False
    touch_high = None
    touch_start = None
    rejection_pcts = []
    last_touch_day = None
    for j in range(max(i - 60, 0), i):
        touched = high.iloc[j] >= pivot20 * 0.99
        if touched and not in_touch:
            res_tests += 1
            touch_high = high.iloc[j]
            touch_start = j
        elif touched and in_touch:
            touch_high = max(touch_high, high.iloc[j])
        elif not touched and in_touch:
            window_end = min(touch_start + 5, i)
            pullback_low = low.iloc[touch_start:window_end].min()
            rejection_pcts.append((touch_high - pullback_low) / touch_high * 100)
        if touched:
            last_touch_day = j
        in_touch = touched
    avg_rejection_pct = float(np.mean(rejection_pcts)) if rejection_pcts else None
    days_since_last_touch = (i - 1 - last_touch_day) if last_touch_day is not None else None

    # time spent pressing the pivot: consecutive recent closes within 3% below
    days_near_res = 0
    for j in range(i - 1, max(i - 31, 0), -1):
        if close.iloc[j] >= pivot20 * 0.97:
            days_near_res += 1
        else:
            break

    # ascending triangle: rising swing lows into a flat, repeatedly tested pivot
    asc_triangle = False
    try:
        ll60 = low.tail(60).reset_index(drop=True)
        pls = [ll60[j] for j in range(5, len(ll60) - 5)
               if ll60[j] == ll60[j-5:j+6].min()]
        asc_triangle = bool(res_tests >= 2 and len(pls) >= 3
                            and pls[-1] > pls[-2] > pls[-3])
    except Exception:
        pass

    # volume building into the pivot (alternative accumulation signature to dry-up)
    vol_building = bool(vol.iloc[-4:-1].mean() > vol.iloc[-9:-4].mean() * 1.15)

    # delivery rising while price consolidates
    # circuit-prone: frequent near-circuit moves = untradeable risk
    big_moves = int((close.pct_change().abs().tail(120) > 0.094).sum())

    # historical breakout behavior on available history:
    # every past close above the prior 20d high -> did +4% arrive before -3%
    # within 5 sessions, or did the stop hit first, or did neither resolve?
    # three REAL counted outcomes, not modeled/invented probabilities. Also
    # track the REAL magnitude reached that day, not just whether the 4%/3%
    # threshold was crossed - a "win" day might have actually moved +7%, a
    # "stop" day might have actually dropped -5%, and that's the number
    # worth knowing, not the threshold itself.
    bh_n = bh_wins = bh_stops = 0
    bh_rets = []
    bh_win_moves = []
    bh_stop_moves = []
    bh_win_days = []
    bh_individual_events = []   # real (date, outcome, movePct) per past event, not just aggregates
    try:
        hi20 = high.rolling(20).max().shift(1)
        events = [j for j in range(30, n - 6)
                  if close.iloc[j] > hi20.iloc[j]
                  and close.iloc[j - 1] <= hi20.iloc[j - 1]]
        for j in events:
            e = close.iloc[j]
            win = lose = False
            ev_move = None
            for k in range(j + 1, min(j + 6, n)):
                if low.iloc[k] <= e * 0.97:
                    lose = True
                    ev_move = low.iloc[k] / e - 1
                    bh_stop_moves.append(ev_move)   # real drop, not capped at -3%
                    break
                if high.iloc[k] >= e * 1.04:
                    win = True
                    ev_move = high.iloc[k] / e - 1
                    bh_win_moves.append(ev_move)   # real gain, not capped at +4%
                    bh_win_days.append(k - j)                    # real trading days to that win
                    break
            bh_n += 1
            if win: bh_wins += 1
            elif lose: bh_stops += 1
            end = min(j + 5, n - 1)
            close_ret = close.iloc[end] / e - 1
            bh_rets.append(close_ret)
            bh_individual_events.append({
                "date": str(df["date"].iloc[j]),
                "outcome": "win" if win else "stop" if lose else "neither",
                "movePct": round((ev_move if ev_move is not None else close_ret) * 100, 2),
            })
    except Exception:
        pass
    bh_win_rate = (bh_wins / bh_n * 100) if bh_n else 0.0
    bh_stop_rate = (bh_stops / bh_n * 100) if bh_n else 0.0
    bh_neither_rate = max(0.0, 100.0 - bh_win_rate - bh_stop_rate) if bh_n else 0.0
    bh_avg5 = (float(np.mean(bh_rets)) * 100) if bh_rets else 0.0
    bh_avg_win_move = (float(np.mean(bh_win_moves)) * 100) if bh_win_moves else 0.0
    bh_avg_stop_move = (float(np.mean(bh_stop_moves)) * 100) if bh_stop_moves else 0.0
    bh_median_win_move = (float(np.median(bh_win_moves)) * 100) if bh_win_moves else 0.0
    bh_best_win_move = (float(np.max(bh_win_moves)) * 100) if bh_win_moves else 0.0
    bh_worst_stop_move = (float(np.min(bh_stop_moves)) * 100) if bh_stop_moves else 0.0
    # middle 50% of real winning moves (interquartile range) - a genuinely
    # richer picture than one average number, from the exact same data
    # already collected, no new sampling or fabrication.
    if len(bh_win_moves) >= 4:
        bh_win_q25 = float(np.percentile(bh_win_moves, 25)) * 100
        bh_win_q75 = float(np.percentile(bh_win_moves, 75)) * 100
    else:
        bh_win_q25 = bh_win_q75 = None
    bh_fastest_win_days = int(np.min(bh_win_days)) if bh_win_days else None
    bh_typical_win_days = int(round(np.median(bh_win_days))) if bh_win_days else None
    bh_slowest_win_days = int(np.max(bh_win_days)) if bh_win_days else None
    bh_period_start = str(df["date"].iloc[0]) if n > 0 else None
    bh_period_end = str(df["date"].iloc[i]) if n > 0 else None
    # stability: how consistent were the winning moves, not just their
    # average - coefficient of variation (std/mean) of the real win
    # magnitudes. Low = wins cluster tightly around the average (reliable),
    # high = wins swing wildly (less predictable even if the average looks good).
    if len(bh_win_moves) >= 3:
        wm = np.array(bh_win_moves)
        cv = float(np.std(wm) / abs(np.mean(wm))) if np.mean(wm) != 0 else None
        bh_stability = ("Stable" if cv is not None and cv < 0.4 else
                        "Variable" if cv is not None and cv < 0.8 else
                        "Highly variable") if cv is not None else None
    else:
        bh_stability = None

    # base length: sessions spent within 15% under the 20d pivot
    base_len = 0
    for j in range(i - 1, max(i - 121, 0), -1):
        if low.iloc[j] > pivot20 * 0.85 and high.iloc[j] <= pivot20 * 1.005:
            base_len += 1
        else:
            break

    return {
        "close": float(close.iloc[i]), "volume": float(vol.iloc[i]),
        "delivery_pct": float(df["delivery_pct"].iloc[i] or 0),
        "ema20": float(e20.iloc[-1]), "ema50": float(e50.iloc[-1]),
        "ema200": float(e200.iloc[-1]),
        "ema_spread_pct": float((max(e20.iloc[-1], e50.iloc[-1], e200.iloc[-1])
                                 - min(e20.iloc[-1], e50.iloc[-1], e200.iloc[-1]))
                                / close.iloc[i] * 100) if close.iloc[i] else 0.0,
        "rsi": float(r.iloc[-1]),
        "williams_r": float(wr.iloc[-1]) if not pd.isna(wr.iloc[-1]) else -50.0,
        "macd": float(macd_l.iloc[-1]), "macd_signal": float(macd_s.iloc[-1]),
        "macd_hist": float(hist.iloc[-1]), "macd_hist_prev": float(hist.iloc[-2]),
        "macd_cross_up": bool(macd_l.iloc[-1] > macd_s.iloc[-1]
                              and macd_l.iloc[-2] <= macd_s.iloc[-2]),
        "atr": float(a.iloc[-1]),
        "atr_falling": bool(a.iloc[-1] < a.iloc[-11]) if n > 11 else False,
        "adx": float(ax.iloc[-1]) if pd.notna(ax.iloc[-1]) else 0.0,
        "adx_rising": bool(ax.iloc[-1] > ax.iloc[-6]) if n > 6 and pd.notna(ax.iloc[-6]) else False,
        "avg_volume_20": float(avg_vol.iloc[-1]) if pd.notna(avg_vol.iloc[-1]) else 0.0,
        "avg_volume_5_prior": float(vol.iloc[-6:-1].mean()),
        "avg_volume_30_prior": float(vol.iloc[-31:-1].mean()),
        "ret63": ret63, "ret126": ret126, "ret252": ret252,
        "clv": clv, "vol_pctile": vol_pctile, "gap_pct": gap_pct,
        "tr_expand": tr_expand, "weekly_up": weekly_up,
        "above_avwap": above_avwap, "hhhl": hhhl, "base_len": base_len,
        "res_tests": res_tests, "days_near_res": days_near_res,
        "avg_rejection_pct": avg_rejection_pct, "days_since_last_touch": days_since_last_touch,
        "asc_triangle": asc_triangle, "vol_building": vol_building,
        "big_moves": big_moves,
        "bh_n": bh_n, "bh_win_rate": bh_win_rate, "bh_stop_rate": bh_stop_rate,
        "bh_neither_rate": bh_neither_rate, "bh_avg5": bh_avg5,
        "bh_avg_win_move": bh_avg_win_move, "bh_avg_stop_move": bh_avg_stop_move,
        "bh_median_win_move": bh_median_win_move, "bh_best_win_move": bh_best_win_move,
        "bh_worst_stop_move": bh_worst_stop_move,
        "bh_fastest_win_days": bh_fastest_win_days, "bh_typical_win_days": bh_typical_win_days,
        "bh_slowest_win_days": bh_slowest_win_days,
        "bh_period_start": bh_period_start, "bh_period_end": bh_period_end,
        "bh_win_q25": bh_win_q25, "bh_win_q75": bh_win_q75,
        "bh_individual_events": bh_individual_events,
        "bh_stability": bh_stability,
        "bw_pctile": bw_pctile,
        "obv_rising": bool(obv.iloc[-1] > obv.iloc[-21]) if n > 21 else False,
        "cmf": cmf,
        "patterns": patterns,
        "hi52": hi52, "yr_bars": look52,
        "cons_low": float(low.iloc[-16:-1].min()),
        "dd20": float((close.iloc[-1] - high.iloc[-21:-1].max())
                     / high.iloc[-21:-1].max() * 100) if n > 21 else 0.0,
        "chg1": float(close.pct_change().iloc[-1] * 100),
        "ret20": ret20, "ret63": ret63, "ret126": ret126,
        # filled later (need full df window per config):
        "_df_tail": df.tail(60)[["open", "high", "low", "close"]],
    }


# ---------------------------------------------------------------- scoring
HORIZONS = {
    "2d": dict(label="2 days",  hold=2,  stopATR=1.2, t1ATR=1.5, t2ATR=2.5, watch=2, dayLo=1,  dayHi=3),
    "1w": dict(label="1 week",  hold=5,  stopATR=2.0, t1ATR=2.5, t2ATR=4.0, watch=5, dayLo=2,  dayHi=6),
    "2w": dict(label="2 weeks", hold=10, stopATR=2.5, t1ATR=3.5, t2ATR=5.5, watch=5, dayLo=4,  dayHi=11),
    "1m": dict(label="1 month", hold=22, stopATR=3.0, t1ATR=5.0, t2ATR=8.0, watch=7, dayLo=9,  dayHi=25),
}

# what each horizon cares about (weights sum to 100)
H_WEIGHTS = {
    "2d": dict(volq=20, pivot=18, atrexp=8, adx=8, fresh=9, candle=8,
               rsrank=10, notext=5, restest=7, history=7),
    "1w": dict(pivot=13, volq=13, vcp=11, rsrank=11, dryup=7, ema=7, weekly=6,
               rsi=5, adx=5, candle=4, accum=4,
               restest=6, asctri=4, nearres=4),
    "2w": dict(vcp=16, rsrank=13, prox52=11, ema=10, volq=9, weekly=7,
               dryup=6, adx=5, sector=5, accum=5,
               restest=6, asctri=4, nearres=3),
    "1m": dict(rsrank=17, ema=14, weekly=11, prox52=11, accum=12, sector=9,
               hhhl=8, vcp=6, rsi=4, history=8),
}


def components(s, cfg, rel_strength, sector_strong, res_dist, rs_rank=50):
    """Normalized 0-1 evidence fractions + human labels, computed once."""
    c = {}
    close = s["close"]

    prox = close / s["hi52"] * 100 if s["hi52"] else 0
    yr = "52w" if s["yr_bars"] >= 240 else f"{s['yr_bars']}d"
    c["prox52"] = ((1.0, f"At/near new {yr} high ({prox:.0f}%)") if prox >= 95 else
                   (0.6, f"Near {yr} high ({prox:.0f}%)") if prox >= 90 else
                   (0.3, f"{prox:.0f}% of {yr} high") if prox >= 85 else
                   (0.0, f"Far from {yr} high ({prox:.0f}%)"))

    tail = s["_df_tail"]; cd = int(cfg["consolidationDays"])
    win = tail.tail(cd + 1).iloc[:-1]
    rng_pct = (win["high"].max() - win["low"].min()) / close * 100 if len(win) >= 5 else 99
    base = 0.7 if rng_pct <= 8 else 0.45 if rng_pct <= 12 else 0.2 if rng_pct <= 16 else 0
    if base and s["atr_falling"]: base += 0.15
    if base and s["bw_pctile"] <= 25: base += 0.15
    c["consol"] = (min(base, 1.0),
                   (f"Consolidating {cd}d (range {rng_pct:.0f}%"
                    + (", ATR falling" if s["atr_falling"] else "")
                    + (", BB squeeze" if s["bw_pctile"] <= 25 else "") + ")")
                   if base else f"No consolidation (range {rng_pct:.0f}%)")
    c["_rng_pct"] = rng_pct

    dry = s["avg_volume_5_prior"] / max(s["avg_volume_20"], 1)
    c["dryup"] = ((1.0, "Volume dry-up before move") if dry <= 0.75 else
                  (0.5, "Mild volume dry-up") if dry <= 0.9 else
                  (0.0, "No volume dry-up"))

    vx = s["volume"] / max(s["avg_volume_20"], 1)
    c["volexp"] = ((1.0, f"Volume expansion {vx:.1f}x") if vx >= 2 else
                   (0.65, f"Volume {vx:.1f}x average") if vx >= 1.5 else
                   (0.3, f"Volume {vx:.1f}x average") if vx >= 1.2 else
                   (0.0, f"Volume only {vx:.1f}x average"))

    c["ema"] = ((1.0, "EMA20 > EMA50 > EMA200 aligned")
                if s["ema20"] > s["ema50"] > s["ema200"] and close > s["ema20"] else
                (0.6, "EMA20 above EMA50") if s["ema20"] > s["ema50"] else
                (0.0, "EMAs not aligned"))


    c["rsi"] = ((1.0, f"RSI {s['rsi']:.0f} in sweet spot")
                if cfg["minRsi"] <= s["rsi"] <= cfg["maxRsi"] else
                (0.5, f"RSI {s['rsi']:.0f} acceptable")
                if cfg["minRsi"]-5 <= s["rsi"] <= cfg["maxRsi"]+4 else
                (0.0, f"RSI {s['rsi']:.0f} outside range"))

    c["adx"] = ((1.0, f"ADX {s['adx']:.0f} and rising")
                if s["adx"] >= 25 and s["adx_rising"] else
                (0.6, f"ADX {s['adx']:.0f}") if s["adx"] >= 20 else
                (0.0, f"ADX weak ({s['adx']:.0f})"))

    c["rel"] = ((1.0, f"Outperforming market by {rel_strength:.1f}% (20d)")
                if rel_strength > 0 else (0.0, "Underperforming market (20d)"))
    c["sector"] = ((1.0, "Sector among market leaders") if sector_strong
                   else (0.0, "Sector not leading"))
    c["accum"] = ((1.0, "Steady buying pressure building")
                  if s["obv_rising"] and s["cmf"] > 0 else
                  (0.5, "OBV rising") if s["obv_rising"] else
                  (0.0, "No accumulation signature"))
    c["fresh"] = ((1.0, "Fresh MACD bullish crossover") if s["macd_cross_up"] else
                  (0.5, "MACD improving")
                  if s["macd_hist"] > s["macd_hist_prev"] else (0.0, "MACD flat"))
    c["notext"] = ((1.0, f"Not extended (today {s['chg1']:+.1f}%)")
                   if 0 < s["chg1"] <= 6 else
                   (0.4, f"Today {s['chg1']:+.1f}%") if -1 <= s["chg1"] <= 0 else
                   (0.0, f"Extended/weak today ({s['chg1']:+.1f}%)"))
    c["res"] = ((1.0, "Broke resistance today") if res_dist <= 0 else
                (0.8, f"Within {res_dist:.1f}% of resistance")
                if res_dist <= cfg["maxDistToBreakoutPct"] else
                (0.0, f"{res_dist:.1f}% below resistance"))
    # --- upgraded evidence (institutional additions) ---
    c["rsrank"] = ((1.0, f"RS Rank {rs_rank:.0f} - market leader") if rs_rank >= 90 else
                   (0.75, f"RS Rank {rs_rank:.0f}") if rs_rank >= 80 else
                   (0.5, f"RS Rank {rs_rank:.0f}") if rs_rank >= 70 else
                   (0.25, f"RS Rank {rs_rank:.0f}") if rs_rank >= 60 else
                   (0.0, f"RS Rank {rs_rank:.0f} - laggard"))

    # VCP: successive range contraction 30d -> 20d -> 10d (Minervini)
    t = s["_df_tail"]
    def _rng(nn):
        w = t.tail(nn + 1).iloc[:-1]
        return (w["high"].max() - w["low"].min()) / close * 100 if len(w) >= 5 else 99
    r30, r20, r10 = _rng(30), _rng(20), _rng(10)
    shrink = r30 > r20 > r10
    vcp = (1.0 if shrink and r10 <= 5 else
           0.75 if shrink and r10 <= 8 else
           0.45 if shrink else
           0.25 if r10 <= 6 else 0.0)
    bl = s["base_len"]
    if vcp and 25 <= bl <= 80: vcp = min(vcp + 0.15, 1.0)
    elif vcp and bl < 10: vcp = max(vcp - 0.25, 0.0)      # too-short base: fake-out risk
    c["vcp"] = (vcp,
                (f"Base tightening nicely ({r30:.0f}% down to {r10:.0f}% range)"
                 + (f", quiet for {bl} sessions" if bl else "")) if vcp >= 0.45 else
                (f"Trading in a tight range ({r10:.0f}%)" if vcp > 0 else
                 "No real tightening yet"))

    # pivot proximity: ideal is -1% to +0.5% around the pivot (dist>0 = below)
    d = res_dist
    c["pivot"] = ((1.0, "Right at the breakout level") if -0.5 <= d <= 1.0 else
                  (0.7, f"{d:.1f}% below the breakout level") if 1.0 < d <= 2.0 else
                  (0.45, f"{d:.1f}% below the breakout level")
                  if 2.0 < d <= cfg["maxDistToBreakoutPct"] else
                  (0.3, f"Already broke out, {-d:.1f}% above") if -2.0 <= d < -0.5 else
                  (0.0, f"Extended {-d:.1f}% too far past breakout") if d < -2.0 else
                  (0.0, "Not close enough to its breakout level yet"))

    # volume quality: percentile beats fixed multiples
    vp = s["vol_pctile"]
    c["volq"] = ((1.0, f"Heaviest volume in months ({vx:.1f}x normal)") if vp >= 95 else
                 (0.7, f"Strong volume ({vx:.1f}x normal)") if vp >= 85 else
                 (0.45, f"Volume {vx:.1f}x normal") if vx >= 1.5 else
                 (0.2, f"Volume {vx:.1f}x normal") if vx >= 1.2 else
                 (0.0, f"Volume is quiet ({vx:.1f}x normal)"))

    # dry-up quality vs 30d baseline (institutional bases go very quiet)
    dq = s["avg_volume_5_prior"] / max(s["avg_volume_30_prior"], 1)
    c["dryup"] = ((1.0, "Trading went very quiet before this move") if dq <= 0.4 else
                  (0.7, "Trading quieted down before this move") if dq <= 0.6 else
                  (0.4, "Slightly quieter than usual before this move") if dq <= 0.75 else
                  (0.0, "No unusual quiet period before this"))

    c["weekly"] = ((1.0, "Weekly chart also trending up") if s["weekly_up"]
                   else (0.0, "Weekly chart not trending up yet"))
    c["hhhl"] = ((1.0, "Each pullback held higher than the last") if s["hhhl"]
                 else (0.0, "No clear rising pattern yet"))
    c["atrexp"] = ((1.0, "Today's move is bigger than normal") if s["tr_expand"]
                   else (0.0, "Today's move is unremarkable"))

    # candle quality: strong close + pattern
    cl = s["clv"]
    cq = (0.6 if cl >= 0.75 else 0.3 if cl >= 0.6 else 0.0)
    if s["patterns"]: cq = min(cq + 0.4, 1.0)
    c["candle"] = (cq, ("Closed strong, near the day's high"
                        + (" — " + ", ".join(s["patterns"]) if s["patterns"] else ""))
                   if cq else "Closed weak, near the day's low")

    # accumulation upgraded: OBV + CMF + anchored VWAP
    acc = (0.3 if s["obv_rising"] else 0) + (0.3 if s["cmf"] > 0 else 0)           + (0.4 if s["above_avwap"] else 0)
    lbl = [x for x, ok in [("buying volume rising", s["obv_rising"]),
                           ("money flowing in", s["cmf"] > 0),
                           ("trading above its average cost basis", s["above_avwap"])] if ok]
    c["accum"] = (acc, ("Steady buying: " + ", ".join(lbl)) if lbl
                  else "No sign of steady accumulation")

    rt = s["res_tests"]
    c["restest"] = ((1.0, f"Resistance tested {rt}x - well-defined level") if 4 <= rt <= 6 else
                    (0.6, f"Resistance tested {rt}x") if 2 <= rt <= 3 else
                    (0.2, "Resistance tested once") if rt == 1 else
                    (0.0, "Untested level" if rt == 0 else
                     f"Resistance tested {rt}x - possibly too strong"))

    c["asctri"] = ((1.0, "Ascending triangle: higher lows into flat resistance")
                   if s["asc_triangle"] else (0.0, "No ascending triangle"))

    dn = s["days_near_res"]
    c["nearres"] = ((1.0, f"Holding at resistance {dn} sessions - sellers tiring") if dn >= 5 else
                    (0.6, f"Pressing resistance {dn} sessions") if dn >= 3 else
                    (0.0, "Not holding at resistance"))

    c["volbuild"] = ((1.0, "Volume building into the level")
                     if s["vol_building"] else (0.0, "Volume flat"))


    # historical breakout behavior - only credited with a real sample
    if s["bh_n"] >= 5:
        wr, sr = s["bh_win_rate"], s["bh_stop_rate"]
        move_bits = []
        if s["bh_avg_win_move"]: move_bits.append(f"averaged {s['bh_avg_win_move']:+.1f}% when it worked")
        if s["bh_avg_stop_move"]: move_bits.append(f"{s['bh_avg_stop_move']:+.1f}% when it stopped out")
        move_txt = f" ({', '.join(move_bits)})" if move_bits else ""
        lbl = (f"History: {s['bh_n']} breakouts past year - {wr:.0f}% reached "
               f"+4% before -3%, {sr:.0f}% hit -3% first{move_txt}")
        c["history"] = ((1.0, lbl) if wr >= 70 else
                        (0.6, lbl) if wr >= 55 else (0.15, lbl))
    else:
        c["history"] = (0.0, f"History: only {s['bh_n']} past breakout(s) - too few to judge")

    if s["patterns"]:
        c["pattern"] = (1.0, "Pattern: " + ", ".join(s["patterns"]))
    return c


def horizon_scores(c):
    """Score all 4 horizons from the shared components."""
    out = {}
    for h, w in H_WEIGHTS.items():
        pts, br = 0.0, []
        for key, weight in w.items():
            frac, label = c.get(key, (0.0, key))
            got = frac * weight
            pts += got
            br.append({"label": label, "points": round(got, 1), "max": weight, "key": key})
        if "pattern" in c:
            pts += 2
            br.append({"label": c["pattern"][1], "points": 2, "max": 2})
        out[h] = (min(round(pts), 100), br)
    return out


def resistance_info(s, cfg):
    tail = s["_df_tail"]
    lb = int(cfg["resistanceLookback"])
    res = float(tail["high"].tail(lb + 1).iloc[:-1].max())
    dist = (res - s["close"]) / s["close"] * 100
    return res, dist


def hard_filter_fails(s, cfg, rs_rank=50):
    """The user's core filters - still enforced for A-grade."""
    fails = []
    if not (cfg["minPrice"] <= s["close"] <= cfg["maxPrice"]):
        fails.append(f"price Rs{s['close']:.0f} outside {cfg['minPrice']}-{cfg['maxPrice']}")
    if cfg.get("trendEma", True) and not s["ema20"] > s["ema50"]:
        fails.append("EMA20 below EMA50")
    if cfg.get("requireEma200") and not s["ema50"] > s["ema200"]:
        fails.append("EMA50 below EMA200")
    if not (cfg["minRsi"] <= s["rsi"] <= cfg["maxRsi"]):
        fails.append(f"RSI {s['rsi']:.0f} outside {cfg['minRsi']}-{cfg['maxRsi']}")
    macd_ok = (s["macd"] > s["macd_signal"]
               or s["macd_hist"] > s["macd_hist_prev"])   # histogram turning up
    if not macd_ok:
        fails.append("MACD not bullish or improving")
    if not (s["avg_volume_20"] > 0
            and s["volume"] > cfg["minVolumeMultiplier"] * s["avg_volume_20"]):
        fails.append(f"volume below {cfg['minVolumeMultiplier']}x average")
    if cfg.get("minAvgVolume", 0) and s["avg_volume_20"] < cfg["minAvgVolume"]:
        fails.append(f"avg volume {s['avg_volume_20']/1e5:.1f}L below "
                     f"{cfg['minAvgVolume']/1e5:.1f}L shares/day - too illiquid")
    if cfg.get("minAdx", 0) and s["adx"] < cfg["minAdx"]:
        fails.append(f"ADX {s['adx']:.0f} below {cfg['minAdx']}")
    if cfg.get("minRsRank", 0) and rs_rank < cfg["minRsRank"]:
        fails.append(f"RS Rank {rs_rank:.0f} below {cfg['minRsRank']}")
    if s.get("gap_pct", 0) > 5:
        fails.append(f"gapped up {s['gap_pct']:.1f}% at open - poor R:R, chase risk")
    if cfg.get("maxDrawdownFrom20dHigh", 0) and s.get("dd20", 0) < -cfg["maxDrawdownFrom20dHigh"]:
        fails.append(f"{-s['dd20']:.1f}% below 20d high - showing weakness")
    if s.get("big_moves", 0) >= 3:
        fails.append(f"{s['big_moves']} near-circuit moves in 120d - erratic, skip")
    return fails


# ---------------------------------------------------------------- AI
def gemini_rank(cands, mood, api_key):
    if not api_key:
        return
    payload = [{"symbol": c["symbol"], "sector": c["sector"],
                "breakout_score": c["breakoutScore"],
                "evidence": [b["label"] for b in c["scoreBreakdown"] if b["points"] > 0],
                "risk_reward": c["riskReward"]} for c in cands]
    prompt = (f"Rank these NSE pre-breakout swing candidates (5-10 day holds). "
              f"Market mood: {mood}. Return ONLY a JSON array of "
              '{"symbol","score" (0-100),"reason" (one short line)} - no fences.\n'
              + json.dumps(payload))
    try:
        url = ("https://generativelanguage.googleapis.com/v1beta/models/"
               f"gemini-1.5-flash:generateContent?key={api_key}")
        r = requests.post(url, json={"contents": [{"parts": [{"text": prompt}]}]},
                          timeout=60)
        r.raise_for_status()
        body = r.json()
        u = body.get("usageMetadata", {})
        if u:
            log(f"  Gemini tokens: {u.get('promptTokenCount',0)} in + "
                f"{u.get('candidatesTokenCount',0)} out = "
                f"{u.get('totalTokenCount',0)} total")
        gemini_rank.last_usage = u
        text = body["candidates"][0]["content"]["parts"][0]["text"].strip()
        text = text.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        ai = {x["symbol"]: x for x in json.loads(text)}
        for c in cands:
            if c["symbol"] in ai:
                c["aiExplanation"] = ai[c["symbol"]]["reason"] + "\n" + c["aiExplanation"]
        log("  AI commentary applied (Gemini)")
    except Exception as e:
        log(f"  AI commentary skipped ({type(e).__name__})")


# ---------------------------------------------------------------- review
PARTIAL_BOOK_PCT = 0.6      # book 60% at T1, let 40% ride toward T2

def update_signal_history(db, conn, todays_recs):
    """Track every signal's actual price path for 5 trading days from the
    day it was flagged - A-grade AND Watch alike. This is what lets you
    check whether a Watch rejection (e.g. 'RSI 47 outside 50-70') would
    have moved anyway, or whether the filter that blocked it was earning
    its keep. Independent of review_open/positions: this tracks every
    signal, not just the ones that became tracked A-grade trades."""
    hist = db.setdefault("signalHistory", [])
    have_ids = {h["id"] for h in hist}
    for r in todays_recs:
        if r["id"] not in have_ids:
            hist.append({
                "id": r["id"], "symbol": r["symbol"], "date": r["date"],
                "grade": r["grade"], "gradeLabel": r.get("gradeLabel", r["grade"]),
                "breakoutScore": r.get("breakoutScore"),
                "failedFilters": r.get("failedFilters", []),
                "entryClose": r["closePrice"], "days": [],
            })

    today = dt.date.today()
    for h in hist:
        if len(h["days"]) >= 5:
            continue                                   # tracking window complete
        sig_date = dt.date.fromisoformat(h["date"])
        px = pd.read_sql(
            "SELECT date, close FROM prices WHERE symbol=? AND date>? "
            "ORDER BY date", conn, params=(h["symbol"], h["date"]))
        have_dates = {d["date"] for d in h["days"]}
        for _, row in px.iterrows():
            if row["date"] in have_dates or len(h["days"]) >= 5:
                continue
            chg = round((row["close"] - h["entryClose"]) / h["entryClose"] * 100, 2)
            h["days"].append({"date": row["date"], "close": round(float(row["close"]), 2),
                              "changePct": chg})

    # keep the list from growing forever: drop fully-tracked entries older
    # than ~45 days (plenty of time to have reviewed them)
    cutoff = (today - dt.timedelta(days=45)).isoformat()
    db["signalHistory"] = [h for h in hist if h["date"] >= cutoff or len(h["days"]) < 5]
    log(f"  signal history: tracking {len(db['signalHistory'])} signals "
        f"(5-trading-day performance window)")


def apply_score_change_tracking(db, recs):
    """Day-over-day score movement, using ONLY data already computed today
    and what was stored from a genuinely PRIOR scan - no new data source.
    Compares each symbol's real per-factor points against the last time
    it was scored, and attaches the diff to today's rec if there's
    something genuinely prior to compare against. Then updates the stored
    snapshot for tomorrow's comparison. Order matters: look up BEFORE
    overwriting, so a symbol never gets diffed against itself."""
    prior = db.setdefault("priorScores", {})
    today = dt.date.today().isoformat()
    for r in recs:
        sym = r["symbol"]
        prev = prior.get(sym)
        if prev and prev.get("date") != today and r.get("scoreBreakdown"):
            # prev.breakdown may be from an OLDER scan that predates the
            # "key" field existing at all - use safe .get() so stale stored
            # data can never crash a live scan, and skip any item that
            # genuinely has no key rather than let several None-keyed
            # items collide into one dict entry
            prev_by_key = {b["key"]: b for b in prev.get("breakdown", []) if b.get("key")}
            factor_deltas = []
            for b in r["scoreBreakdown"]:
                pb = prev_by_key.get(b.get("key"))
                if pb is not None:
                    d = round(b.get("points", 0) - pb.get("points", 0), 1)
                    if abs(d) >= 0.5:
                        factor_deltas.append({"label": b.get("label", ""), "delta": d})
            total_delta = round((r.get("breakoutScore", 0) or 0) - (prev.get("score", 0) or 0), 1)
            r["scoreChangeFromPrior"] = {
                "priorScore": prev.get("score"), "priorDate": prev.get("date"),
                "totalDelta": total_delta,
                "factorDeltas": sorted(factor_deltas, key=lambda d: -abs(d["delta"]))[:4],
            }
        if r.get("scoreBreakdown"):
            prior[sym] = {"date": today, "score": r.get("breakoutScore"),
                         "breakdown": r["scoreBreakdown"]}
    # keep this from growing forever - drop anything not seen in 60 days
    cutoff = (dt.date.today() - dt.timedelta(days=60)).isoformat()
    db["priorScores"] = {s: v for s, v in prior.items() if v.get("date", "") >= cutoff}


def apply_watchlist_trend(recs):
    """Same prior-score data, read from the rec's own scoreChangeFromPrior
    (already computed above) to produce a simple Improving/Weakening/
    Unchanged label for Watch candidates - zero new computation, just a
    threshold on the same real total_delta."""
    for r in recs:
        chg = r.get("scoreChangeFromPrior")
        if not chg:
            continue
        d = chg["totalDelta"]
        r["watchTrend"] = ("Improving" if d >= 3 else
                           "Weakening" if d <= -3 else "Unchanged")


def update_signal_health(db, snaps, universe, rs_ranks, cfg):
    """For every confirmed OPEN position, re-run the SAME real checks that
    got it flagged in the first place, using today's actual data - not a
    price/P&L view (that's what your broker app is for), but 'is the reason
    I bought this still true?' Every number here is freshly re-measured,
    nothing predicted or invented."""
    checked = 0
    for rec in db["recommendations"]:
        if rec.get("status") != "OPEN":
            continue
        sym = rec["symbol"]
        snap = snaps.get(sym)
        if not snap:
            continue
        rsr_now = rs_ranks.get(sym, 50)
        fails_now = hard_filter_fails(snap, cfg, rsr_now)
        rec["health"] = {
            "asOf": dt.date.today().isoformat(),
            "rsRank": {"then": rec.get("rsRankAtSignal"), "now": round(rsr_now)},
            "adx": {"then": rec.get("adxAtSignal"), "now": round(snap["adx"], 1)},
            "volRatio": {"then": rec.get("volRatioAtSignal"),
                        "now": round(snap["volume"] / max(snap["avg_volume_20"], 1), 2)},
            "stillPassesFilters": len(fails_now) == 0,
            "newlyFailing": fails_now,
        }
        checked += 1
    if checked:
        log(f"  signal health: re-checked {checked} open position(s) against today's real data")


def review_open(db, conn):
    """Two-stage exit, matching the 'book 50-70% at target, trail the rest'
    rule: T1 hit -> book PARTIAL_BOOK_PCT of the position, move stop to
    breakeven for the remainder. Remainder then exits at T2, breakeven stop,
    or the time limit. Final P&L is the size-weighted blend of both legs."""
    closed = 0
    for rec in db["recommendations"]:
        if rec.get("status") not in ("OPEN", "PARTIAL"):
            continue
        px = pd.read_sql(
            "SELECT date, high, low, close FROM prices "
            "WHERE symbol=? AND date>? ORDER BY date",
            conn, params=(rec["symbol"], rec["date"]))
        if px.empty:
            continue
        entry = rec["closePrice"]
        rec["maxGainPct"] = round((px["high"].max()-entry)/entry*100, 2)
        rec["maxDrawdownPct"] = round((px["low"].min()-entry)/entry*100, 2)
        t1 = rec.get("target1") or rec["target"]
        t2 = rec.get("target2") or t1
        stop = rec["stopLoss"]
        partial_done = rec.get("status") == "PARTIAL"
        runner_stop = rec.get("runnerStop", entry) if partial_done else stop

        for i, day in px.iterrows():
            if not partial_done:
                if day["low"] <= stop:
                    rec.update(status="STOP_LOSS_HIT", exitPrice=round(stop, 2),
                               exitDate=day["date"],
                               pnlPct=round((stop-entry)/entry*100, 2),
                               closedBy="auto")
                    closed += 1
                    break
                if day["high"] >= t1:
                    rec.update(status="PARTIAL", partialDate=day["date"],
                               partialPrice=round(float(t1), 2),
                               partialPct=PARTIAL_BOOK_PCT,
                               runnerStop=round(entry, 2))   # breakeven
                    partial_done = True
                    runner_stop = entry
                    continue
                if i + 1 >= rec.get("maxHoldDays", MAX_HOLD_DAYS):
                    rec.update(status="TIME_EXIT", exitPrice=round(float(day["close"]), 2),
                               exitDate=day["date"],
                               pnlPct=round((day["close"]-entry)/entry*100, 2),
                               closedBy="auto")
                    closed += 1
                    break
            else:
                if day["low"] <= runner_stop:
                    final_ret = (runner_stop - entry) / entry
                    blend = (PARTIAL_BOOK_PCT * (t1 - entry) / entry
                            + (1 - PARTIAL_BOOK_PCT) * final_ret) * 100
                    rec.update(status=("STOP_LOSS_HIT" if runner_stop < entry
                                       else "TARGET_HIT"),
                               exitPrice=round(float(runner_stop), 2),
                               exitDate=day["date"], pnlPct=round(blend, 2),
                               closedBy="auto")
                    closed += 1
                    break
                if day["high"] >= t2:
                    final_ret = (t2 - entry) / entry
                    blend = (PARTIAL_BOOK_PCT * (t1 - entry) / entry
                            + (1 - PARTIAL_BOOK_PCT) * final_ret) * 100
                    rec.update(status="TARGET_HIT", exitPrice=round(float(t2), 2),
                               exitDate=day["date"], pnlPct=round(blend, 2),
                               closedBy="auto")
                    closed += 1
                    break
                if i + 1 >= rec.get("maxHoldDays", MAX_HOLD_DAYS):
                    final_ret = (day["close"] - entry) / entry
                    blend = (PARTIAL_BOOK_PCT * (t1 - entry) / entry
                            + (1 - PARTIAL_BOOK_PCT) * final_ret) * 100
                    rec.update(status="TIME_EXIT", exitPrice=round(float(day["close"]), 2),
                               exitDate=day["date"], pnlPct=round(blend, 2),
                               closedBy="auto")
                    closed += 1
                    break
    if closed:
        log(f"  auto-closed {closed} open recommendation(s)")


# ---------------------------------------------------------------- main
def main():
    log("SwingSense scanner v2 (breakout model) starting")
    db = read_db_json()
    cfg = db.get("config", {}).get("scanner", {})
    for k, v in DEFAULT_CFG.items():
        cfg.setdefault(k, v)
    for k, v in DEFAULT_CFG["minMovePct"].items():
        cfg["minMovePct"].setdefault(k, v)

    log("Step 1/6 syncing NSE data (first run downloads ~1 year, be patient)")
    universe = load_universe()
    conn = sync_history(set(universe))

    log("Step 2/6 computing indicators")
    hist = pd.read_sql("SELECT * FROM prices ORDER BY symbol, date", conn)
    hist, quote_time = overlay_today(hist, universe)
    if hist.empty:
        log("[ERROR] no price data available"); sys.exit(1)
    snaps = {}
    for sym, g in hist.groupby("symbol"):
        s = snapshot(g)
        if s:
            snaps[sym] = s
    log(f"  {len(snaps)} stocks with enough history")

    med_ret20 = float(np.median([s["ret20"] for s in snaps.values()])) if snaps else 0
    # RS Rating: weighted 3/6/12-month return, percentile-ranked across universe
    comp = {sym: 0.4*sn["ret63"] + 0.35*sn["ret126"] + 0.25*sn["ret252"]
            for sym, sn in snaps.items()}
    vals = np.array(sorted(comp.values()))
    rs_ranks = {sym: float((vals <= v).mean() * 100) for sym, v in comp.items()}
    above = sum(1 for s in snaps.values() if s["close"] > s["ema50"]) / max(len(snaps), 1)
    adv = sum(1 for s in snaps.values() if s["chg1"] > 0) / max(len(snaps), 1)
    mood_score = above*0.6 + adv*0.4
    mood = "Bullish" if mood_score > 0.58 else ("Bearish" if mood_score < 0.42 else "Neutral")
    log(f"Step 3/6 market mood: {mood} ({above*100:.0f}% above EMA50)")

    # sector strength: top 40% of sectors by average 20d return
    sec_ret = {}
    for sym, s in snaps.items():
        sec = universe.get(sym, {}).get("sector", "")
        sec_ret.setdefault(sec, []).append(s["ret20"])
    sec_avg = {k: float(np.mean(v)) for k, v in sec_ret.items() if len(v) >= 5}
    strong_cut = np.percentile(list(sec_avg.values()), 60) if sec_avg else 0
    strong_sectors = {k for k, v in sec_avg.items() if v >= strong_cut}
    top_sec = max(sec_avg, key=sec_avg.get) if sec_avg else ""
    weak_sec = min(sec_avg, key=sec_avg.get) if sec_avg else ""
    sector_ranking = [{"sector": k, "avgReturn20d": round(v, 2)}
                      for k, v in sorted(sec_avg.items(), key=lambda kv: kv[1], reverse=True)]
    log(f"  top sector: {top_sec} / weak sector: {weak_sec}")

    log("Step 4/6 horizon scoring (2d / 1w / 2w / 1m)")
    today = dt.date.today().isoformat()
    scored = []
    for sym, s in snaps.items():
        if not (cfg["minPrice"] <= s["close"] <= cfg["maxPrice"]):
            continue
        if s["big_moves"] >= 3:
            continue          # near-circuit regulars: untradeable, never shown
        rel = s["ret20"] - med_ret20
        sec = universe.get(sym, {}).get("sector", "")
        res, dist = resistance_info(s, cfg)
        rsr = rs_ranks.get(sym, 50)
        comps = components(s, cfg, rel, sec in strong_sectors, dist, rsr)
        hs = horizon_scores(comps)
        best_h = max(hs, key=lambda h: hs[h][0])   # single best horizon: no repeats
        score, breakdown = hs[best_h]
        fails = hard_filter_fails(s, cfg, rsr)
        scored.append(dict(sym=sym, s=s, score=score, breakdown=breakdown,
                           horizon=best_h, res=res, dist=dist, fails=fails,
                           sectorStrong=(sec in strong_sectors),
                           rs_rank=rsr))
    log(f"  scored {len(scored)} in-band stocks across 4 horizons")

    # honest percentile AND raw rank: this stock's score vs every OTHER
    # stock actually scored today (not a modeled probability - a straight
    # rank of real numbers). Ties share the same rank (dense ranking).
    all_scores = sorted((x["score"] for x in scored), reverse=True)
    n_scored = len(all_scores)
    def percentile_rank(sc):
        if n_scored < 2:
            return None
        better_or_equal = sum(1 for v in all_scores if v <= sc)
        return round(better_or_equal / n_scored * 100)
    def raw_rank(sc):
        if n_scored < 1:
            return None
        return sum(1 for v in all_scores if v > sc) + 1


    def classify(x):
        s = x["s"]
        if x["dist"] <= 0 and x["vx"] >= 2 and s["chg1"] >= 2:
            return "2d"     # broke resistance today on big volume: momentum burst
        if 0 <= x["dist"] <= 1.5 and x["vx"] >= 1.5:
            return "1w"     # knocking on resistance with participation
        if x["rng_pct"] <= 12:
            return "2w"     # tight consolidation: classic swing breakout
        if (s["ema20"] > s["ema50"] > s["ema200"] and s["adx"] >= 22
                and x["rel"] > 0):
            return "1m"     # aligned trend + relative strength: position ride
        return "2w"

    def make_rec(x, grade):
        s, sym = x["s"], x["sym"]
        hz = HORIZONS[x["horizon"]]
        a = s["atr"]
        current_close = s["close"]
        # A-grade is already at/through its pivot, so current close IS the
        # entry. Watch is, by definition, BELOW its pivot - so the buy zone,
        # stop and targets must be projected off the trigger price (the
        # resistance level it needs to cross), not today's close. Otherwise
        # the card shows a "buy zone" at a price you're explicitly told not
        # to buy at, contradicting its own watch-plan text.
        trigger_price = max(x["res"], current_close)
        entry = current_close if grade == "A" else trigger_price
        # tight base -> stop goes just under the consolidation low
        # (tighter risk is what makes breakout trades pay), else 2 x ATR
        tail = s["_df_tail"]
        cd = int(cfg["consolidationDays"])
        win = tail.tail(cd + 1).iloc[:-1]
        rng_pct = ((win["high"].max() - win["low"].min()) / entry * 100
                   if len(win) >= 5 else 99)
        atr_stop = entry - hz["stopATR"]*a
        base_stop = s["cons_low"] * 0.99
        stop = round(max(atr_stop, base_stop) if rng_pct <= 16 else atr_stop, 2)
        stop = min(stop, round(entry * 0.995, 2))          # never above ~entry
        t1 = round(entry + hz["t1ATR"]*a, 2)
        t2 = round(entry + hz["t2ATR"]*a, 2)
        rr = round((t1 - entry) / max(entry - stop, 0.01), 2)
        t1_pct = round((t1 - entry) / entry * 100, 1)
        t2_pct = round((t2 - entry) / entry * 100, 1)

        # --- transparency fields: all derived from numbers already computed
        # above for scoring/stops, just never surfaced as their own line ---
        rvol = round(s["volume"] / max(s["avg_volume_20"], 1), 2)
        atr_pct = round(a / entry * 100, 2)
        adx_val = s["adx"]
        adx_label = ("Strong trend" if adx_val >= 25 else
                     "Moderate trend" if adx_val >= 20 else "Weak trend")
        dist52_pct = round((entry - s["hi52"]) / s["hi52"] * 100, 1) if s["hi52"] else None
        near52_label = ("At ATH" if dist52_pct is not None and dist52_pct >= -0.5 else
                        "Near ATH" if dist52_pct is not None and dist52_pct >= -5 else
                        "Approaching ATH" if dist52_pct is not None and dist52_pct >= -10 else
                        "Below ATH")
        near52_stars = (5 if dist52_pct is not None and dist52_pct >= -0.5 else
                        4 if dist52_pct is not None and dist52_pct >= -5 else
                        3 if dist52_pct is not None and dist52_pct >= -10 else
                        2 if dist52_pct is not None and dist52_pct >= -20 else 1)
        avg_daily_value_cr = round(s["avg_volume_20"] * entry / 1e7, 1)
        closing_range_pct = round(s["clv"] * 100)
        # estimated days to T1: this stock's own typical daily move (ATR as
        # % of price) sets the pace - a stock with bigger daily swings gets
        # there faster. Clamped to a sane band per horizon so a 2-day setup
        # can't claim 3 weeks and a 1-month trend-ride can't claim 2 days.
        atr_pct = max(a / entry * 100, 0.3)
        raw_days = t1_pct / atr_pct
        est_days = int(round(min(max(raw_days, hz["dayLo"]), hz["dayHi"])))
        near_bo = 0 <= x["dist"] <= cfg["maxDistToBreakoutPct"]  # kept for other uses below

        # --- watchlist progress: only meaningful for stocks that haven't
        # triggered yet (dist > 0). Real numbers only.
        trigger_readiness = None
        volume_gap_text = None
        est_days_to_trigger = None
        if x["dist"] > 0:
            band = max(cfg["maxDistToBreakoutPct"], 0.1)
            trigger_readiness = round(max(0, min(100, (1 - x["dist"] / band) * 100)))
            vx_now = s["volume"] / max(s["avg_volume_20"], 1)
            need = cfg["minVolumeMultiplier"]
            volume_gap_text = (f"Volume at {vx_now:.1f}x - meets the {need}x bar" if vx_now >= need
                               else f"Volume at {vx_now:.1f}x - needs {need}x to confirm")
            days_raw = x["dist"] / max(atr_pct, 0.3)
            est_days_to_trigger = int(round(min(max(days_raw, 1), 10)))
        def add_days(d0, nn):
            d0 = dt.date.fromisoformat(d0)
            while nn > 0:
                d0 += dt.timedelta(days=1)
                if d0.weekday() < 5: nn -= 1
            return d0.isoformat()
        watch_until = add_days(today, hz["watch"]) if grade == "B" else None
        stars = min(5, max(1, int(x["score"] // 20) + (1 if x["score"] >= 90 else 0)))
        glabel = ("A+" if x["score"] >= 90 else "A" if x["score"] >= 80 else "B")
        setup = ("Breakout buy" if x["dist"] <= 0 else
                 "Ascending triangle" if s["asc_triangle"] else
                 "Base breakout setup" if s["base_len"] >= 15 else
                 "Pullback continuation" if s["close"] <= s["ema20"] * 1.02 else
                 "Momentum continuation")
        vx_ = s["volume"] / max(s["avg_volume_20"], 1)
        d_ = x["dist"]
        pos_txt = (f"{-d_:.1f}% above its pivot" if d_ < 0
                   else f"{d_:.1f}% below its pivot Rs{x['res']:.1f}")
        base_txt = (f" after a {s['base_len']}-day base" if s["base_len"] >= 10 else "")
        align_txt = ("weekly and daily trends aligned" if s["weekly_up"]
                     else "daily uptrend")
        summary = (f"{glabel} ({x['score']}/100): trading {pos_txt}{base_txt}. "
                   f"Volume {vx_:.1f}x average (P{s['vol_pctile']:.0f}), "
                   f"RS Rank {x['rs_rank']:.0f}, "
                   f"{align_txt}. About {t2_pct:.1f}% "
                   f"upside to T2 at {rr}:1 risk-reward, target expected in "
                   f"~{est_days} trading days{' once it triggers' if grade=='B' else ''}.")
        # (resistance proximity is already covered by the "pivot" score
        # component above in plain language - no need to restate it here)
        evidence = [b["label"] for b in x["breakdown"] if b["points"] > 0]
        return {
            "id": f"{sym}-{today}", "date": today, "symbol": sym,
            "companyName": universe.get(sym, {}).get("companyName", sym),
            "sector": universe.get(sym, {}).get("sector", ""),
            "closePrice": round(current_close, 2),
            "projectedEntry": round(entry, 2) if grade == "B" else None,
            "buyZone": f"{entry*0.99:.1f} - {entry*1.01:.1f}",
            "stopLoss": stop, "target": t1, "target1": t1, "target2": t2,
            "riskReward": rr,
            "target1Pct": t1_pct, "potentialPct": t2_pct,
            "resistance": round(x["res"], 2),
            "distToResistancePct": round(x["dist"], 2),
            "triggerReadiness": trigger_readiness,
            "emaSpreadPct": round(s["ema_spread_pct"], 2),
            "williamsR": round(s["williams_r"], 1),
            "avgRejectionPct": round(s["avg_rejection_pct"], 1) if s.get("avg_rejection_pct") is not None else None,
            "daysSinceLastTouch": s.get("days_since_last_touch"),
            "williamsRLabel": ("Oversold" if s["williams_r"] <= -80
                               else "Overbought" if s["williams_r"] >= -20
                               else "Neutral"),
            "macdHist": round(s["macd_hist"], 3),
            "macdRising": bool(s["macd_hist"] > s["macd_hist_prev"]),
            "emaConvergenceLabel": ("Very tight - energy building" if s["ema_spread_pct"] < 2
                                    else "Compressing" if s["ema_spread_pct"] < 5
                                    else "Spread out - already trending"),
            "ret1mo": round(s["ret20"], 1) if s.get("ret20") is not None else None,
            "ret3mo": round(s["ret63"], 1) if s.get("ret63") is not None else None,
            "ret6mo": round(s["ret126"], 1) if s.get("ret126") is not None else None,
            "volumeGapText": volume_gap_text,
            "estDaysToTrigger": est_days_to_trigger,
            "nearestSupport": round(s["cons_low"], 2),
            "rvol": rvol,
            "ema20Level": round(s["ema20"], 2),
            "atrValue": round(a, 2), "atrPct": atr_pct,
            "adxValue": round(adx_val, 1), "adxLabel": adx_label,
            "distFrom52wHighPct": dist52_pct, "near52wLabel": near52_label,
            "near52wStars": near52_stars,
            "avgDailyValueCr": avg_daily_value_cr,
            "closingRangePct": closing_range_pct,
            "horizon": x["horizon"],
            "horizonLabel": hz["label"],
            "estDays": est_days,
            "estDaysRange": f"{hz['dayLo']}-{hz['dayHi']}",
            "maxHoldDays": hz["hold"],
            "holdingPeriod": hz["label"],
            "breakoutScore": x["score"],
            "confidenceScore": x["score"],           # backward compat
            "percentile": percentile_rank(x["score"]),
            "rsRankAtSignal": round(x["rs_rank"]),
            "adxAtSignal": round(s["adx"], 1),
            "volRatioAtSignal": round(s["volume"] / max(s["avg_volume_20"], 1), 2),
            "rawRank": raw_rank(x["score"]),
            "scoredCount": n_scored,
            "gradeLabel": glabel,
            "setupType": setup,
            "sectorStrong": x.get("sectorStrong", False),
            "breakoutHistory": ({"n": s["bh_n"],
                                 "winRate": round(s["bh_win_rate"]),
                                 "stopRate": round(s["bh_stop_rate"]),
                                 "neitherRate": round(s["bh_neither_rate"]),
                                 "avg5d": round(s["bh_avg5"], 1),
                                 "avgWinMove": round(s["bh_avg_win_move"], 1),
                                 "avgStopMove": round(s["bh_avg_stop_move"], 1),
                                 "medianWinMove": round(s["bh_median_win_move"], 1),
                                 "bestWinMove": round(s["bh_best_win_move"], 1),
                                 "worstStopMove": round(s["bh_worst_stop_move"], 1),
                                 "fastestWinDays": s["bh_fastest_win_days"],
                                 "typicalWinDays": s["bh_typical_win_days"],
                                 "slowestWinDays": s["bh_slowest_win_days"],
                                 "periodStart": s["bh_period_start"],
                                 "periodEnd": s["bh_period_end"],
                                 "stability": s["bh_stability"],
                                 "winQ25": round(s["bh_win_q25"], 1) if s.get("bh_win_q25") is not None else None,
                                 "winQ75": round(s["bh_win_q75"], 1) if s.get("bh_win_q75") is not None else None,
                                 "recentEvents": s.get("bh_individual_events", [])[-6:][::-1]}
                                if s["bh_n"] >= 5 else None),
            "rsRank": round(x["rs_rank"], 0),
            "summary": summary,
            "scoreBreakdown": x["breakdown"],
            "riskRating": ("Low" if a/entry*100 < 2 else
                           "Medium" if a/entry*100 < 4 else "High"),
            "grade": grade,
            "failedFilters": x["fails"],
            "aiExplanation": "\n".join(evidence),
            "watchUntil": watch_until,
            "triggerAbove": round(max(x["res"], entry), 2),
            "invalidBelow": stop,
            "status": "SIGNAL" if grade == "A" else "WATCH",
            "marketMood": mood,
        }

    max_n = int(cfg["maxRecommendations"])
    if mood == "Bearish":
        max_n = max(3, max_n - 3)
        log("  bearish tape -> recommending fewer stocks")

    # portfolio-level cap: don't recommend new buys beyond your position limit
    n_open_now = sum(1 for r in db["recommendations"] if r.get("status") == "OPEN")
    room = max(0, int(cfg.get("maxOpenPositions", 4)) - n_open_now)

    a_pool = [x for x in scored if not x["fails"]
              and x["score"] >= cfg["minBreakoutScore"]]
    candidates = []
    for h in HORIZONS:
        pool_h = sorted([x for x in a_pool if x["horizon"] == h],
                        key=lambda x: x["score"], reverse=True)
        for x in pool_h[:max_n]:
            r = make_rec(x, "A")
            if (r["riskReward"] >= cfg["minRiskReward"]
                    and r["potentialPct"] >= cfg["minMovePct"][h]):
                candidates.append(r)
    candidates.sort(key=lambda r: r["breakoutScore"], reverse=True)
    recs = candidates[:room]
    overflow = candidates[room:]
    for r in overflow:                    # still shown, capped from tracking
        r["status"], r["grade"] = "WATCH", "A"
        r["capReason"] = (f"Would qualify, but {cfg.get('maxOpenPositions',4)}-position "
                          f"cap reached ({n_open_now} open)")
    recs = recs + overflow
    log(f"  {len(candidates)} A-grade qualified, {len(candidates)-len(overflow)} "
        f"opened ({n_open_now} already open, room for {room}/"
        f"{cfg.get('maxOpenPositions',4)})")

    gemini_rank(recs, mood, os.environ.get("GEMINI_API_KEY")
                or db.get("config", {}).get("geminiApiKey", ""))

    if cfg["fillWithNearMisses"]:
        chosen = {r["symbol"] for r in recs}
        for h in HORIZONS:
            have_h = sum(1 for r in recs if r["horizon"] == h)
            pool_h = sorted([x for x in scored
                             if x["horizon"] == h and x["sym"] not in chosen
                             and x["score"] >= cfg["minBreakoutScore"]],
                            key=lambda x: x["score"], reverse=True)
            added = 0
            for x in pool_h:
                if added >= max(0, max_n - have_h):
                    break
                r = make_rec(x, "B")
                if r["potentialPct"] >= cfg["minMovePct"][h]:
                    recs.append(r)
                    chosen.add(x["sym"])
                    added += 1
        log(f"  watchlist fill: only score >= {cfg['minBreakoutScore']} "
            f"and per-horizon min move (total {len(recs)})")

    log("Step 5/6 saving recommendations")
    db = read_db_json()
    db["recommendations"] = [r for r in db["recommendations"]
                             if not (r["date"] == today and
                                     r.get("closedBy") != "manual")]
    kept_ids = {r["id"] for r in db["recommendations"]}
    db["recommendations"].extend(r for r in recs if r["id"] not in kept_ids)
    apply_score_change_tracking(db, recs)
    apply_watchlist_trend(recs)
    backtest_results = load_backtest_results()
    if backtest_results:
        for r in recs:
            h = backtest_results.get("byHorizon", {}).get(r.get("horizon"))
            if h:
                r["modelEvidence"] = dict(h, generatedAt=backtest_results.get("generatedAt"),
                                          symbolsCovered=backtest_results.get("symbolsCovered"),
                                          criteria=backtest_results.get("criteria", []))

    log("Step 6/6 reviewing open positions")
    review_open(db, conn)
    update_signal_history(db, conn, recs)
    update_signal_health(db, snaps, universe, rs_ranks, cfg)
    log(f"  fetching recent news for {len(recs)} recommended stock(s)")
    for r in recs:
        r["recentNews"] = fetch_recent_news(r.get("companyName", r["symbol"]), r["symbol"])
    log("  checking for upcoming board meetings (results/dividends) in the next 5 days")
    board_meetings = fetch_upcoming_board_meetings([r["symbol"] for r in recs])
    for r in recs:
        bm = board_meetings.get(r["symbol"])
        if bm:
            r["upcomingEvent"] = bm
    data_date = str(hist["date"].max())
    log("  fetching Nifty 50 level")
    nifty = fetch_nifty_snapshot()
    db["lastScan"] = {"date": today, "mood": mood,
                      "quoteTime": quote_time,
                      "dataDate": data_date,
                      "nifty": nifty,
                      "sectorRanking": sector_ranking,
                      "nextFoExpiry": next_fo_expiry().isoformat(),
                      "aiTokens": getattr(gemini_rank, "last_usage", None),
                      "passed": len(a_pool), "shown": len(recs),
                      "topSector": top_sec, "weakSector": weak_sec,
                      "at": dt.datetime.now().isoformat(timespec="seconds")}
    write_db_json(db)
    conn.close()

    a_grade = sum(1 for r in recs if r["grade"] == "A")
    log(f"Done. {a_grade} A-grade signal(s), {len(recs)-a_grade} watchlist.")
    if a_grade == 0:
        log("No A-grade swing opportunities identified today.")


if __name__ == "__main__":
    main()
