"""
Lightweight, twice-daily intraday check - NOT a re-scan.

The full scan at 6:40 PM does real, heavy work: fetches the whole market,
recomputes every indicator, RS Rank, sector strength, all of it. None of
that can change again until tomorrow's end-of-day data exists - re-running
it midday would just repeat the same answer using yesterday's data.

What GENUINELY can change during the day: whether a WAIT FOR TRIGGER or
WATCHLIST stock has crossed its already-computed breakout level. This
script checks exactly that, and only that - one live quote fetch per
watched stock, no re-scoring, no re-fetching the whole market.

Deliberately does NOT flip a stock to official BUY NOW status. A live
intraday price crossing the level is real information, but it is not the
same as a confirmed end-of-day close above it - price can (and does)
reverse before the close. Silently upgrading the status would blur that
distinction the whole rest of this app has been careful to keep honest.
Instead, this attaches a clearly-labeled, separate "intraday check" field
that the frontend shows alongside the official status, not instead of it.
"""
import sys
import datetime as dt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import scanner as sc


def run_intraday_check():
    db = sc.read_db_json()
    today = dt.date.today().isoformat()
    watching = [r for r in db.get("recommendations", [])
                if r.get("date") == today and r.get("status") == "SIGNAL"
                and r.get("distToResistancePct") is not None
                and r["distToResistancePct"] > 0]

    if not watching:
        sc.log("  intraday check: nothing currently waiting to trigger today - nothing to check")
        return

    symbols = [r["symbol"] for r in watching]
    sc.log(f"  intraday check: {len(symbols)} stock(s) still waiting to trigger, "
           f"fetching live quotes (not a full re-scan)")
    quotes = sc.fetch_current_quotes(symbols)
    now_ist = dt.datetime.now(dt.timezone(dt.timedelta(hours=5, minutes=30)))

    checked = crossed = 0
    for r in watching:
        q = quotes.get(r["symbol"])
        if not q or q.get("close") is None:
            continue
        live_price = q["close"]
        resistance = r.get("resistance") or r.get("triggerAbove")
        if not resistance:
            continue
        live_dist_pct = round((resistance - live_price) / resistance * 100, 2)
        checked += 1
        crossed_now = live_price > resistance
        if crossed_now:
            crossed += 1
        r["intradayCheck"] = {
            "asOf": now_ist.strftime("%H:%M"),
            "livePrice": round(live_price, 2),
            "resistance": round(resistance, 2),
            "liveDistPct": live_dist_pct,
            "aboveResistance": crossed_now,
        }

    sc.write_db_json(db)
    sc.log(f"  intraday check: {checked} stock(s) checked, {crossed} currently trading "
           f"above their resistance level (not yet confirmed - only today's close decides that)")


if __name__ == "__main__":
    run_intraday_check()
