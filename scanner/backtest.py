"""
Real backtest of the FULL scoring model - not the simple per-stock
"closed above the 20-day high" replay already shown on cards, but a true
walk-forward replay of the entire pipeline: hard filters, every scored
factor, RS Rank, sector strength, all of it, re-run on every historical
day using only data that would genuinely have been available that day.

This is deliberately a SEPARATE, on-demand tool, not part of the nightly
scan - it's much heavier (potentially tens of thousands of snapshot()
calls) and its whole point is a one-time or occasional trust-check, not
something that needs to run daily.

Honest scope limits, stated up front rather than discovered later:
  - Only replays over whatever price history is already cached in
    market.db (built up from your normal scans). It does NOT go fetch
    additional years of history - if you've only been scanning for a
    few weeks, this backtest only has a few weeks to work with.
  - RS Rank and sector strength are computed cross-sectionally across
    whatever symbols are ALSO cached in market.db that day, not the
    full 500-stock universe unless you've scanned enough for all of
    it to be cached. Smaller cached universe = noisier rank.
  - Only tests whether a day WOULD have graded A (passed every hard
    filter + cleared the score threshold) - matches the "when this
    exact model said A-grade, how often did it work" question this
    was built to answer, not the Watch-grade near-miss logic.

Usage:
    python backtest.py                  # default config, all cached symbols
    python backtest.py --min-events 5   # require at least N events per horizon
"""
import sqlite3
import sys
import time
import argparse
import json
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import scanner as sc


def fetch_nifty_regime_series():
    """Nifty's own daily Bull/Bear classification, going back ~2 years -
    Close above its own 50-day EMA that day = Bull, below = Bear. This is
    a real, simple, defensible regime proxy, not an invented multi-factor
    score. Zero lookahead: EMA50 on any given day only ever uses that
    day's own past prices, same principle already verified for the main
    backtest. Returns None (gracefully, not a crash) if the fetch fails -
    regime segmentation is a bonus layer on the backtest, not a
    dependency of it."""
    try:
        import yfinance as yf
    except ImportError:
        return None
    try:
        hist = yf.Ticker("^NSEI").history(period="2y", interval="1d")
        if hist is None or len(hist) < 60:
            return None
        closes = hist["Close"].dropna()
        ema50 = closes.ewm(span=50, adjust=False).mean()
        regime = (closes > ema50).map({True: "Bull", False: "Bear"})
        regime.index = regime.index.strftime("%Y-%m-%d")
        return regime
    except Exception as e:
        print(f"  ! Nifty regime fetch failed ({type(e).__name__}) - "
              f"continuing without regime segmentation")
        return None


def load_cached_history(conn):
    """Every symbol's full cached price history from market.db, keyed by
    symbol -> DataFrame with columns open/high/low/close/volume/date."""
    df = pd.read_sql(
        "SELECT symbol, date, open, high, low, close, volume, delivery_pct "
        "FROM prices ORDER BY symbol, date", conn)
    if df.empty:
        return {}
    out = {}
    for sym, g in df.groupby("symbol"):
        g = g.reset_index(drop=True)
        if len(g) >= 90:  # need real runway: 60 for snapshot() + room to walk forward
            out[sym] = g
    return out


def precompute_cross_sectional(history, universe):
    """Vectorized, once: for every cached symbol, its date-indexed ret20
    series. From that, build per-date RS percentile rank and per-date
    sector-average-return tables. This is the expensive-if-done-naively
    part (would be O(days x stocks) snapshot calls) done instead as a
    handful of wide-table pandas operations - fast, and still zero
    lookahead, since ret20 on date d only ever looks backward from d."""
    ret20_by_sym = {}
    for sym, df in history.items():
        s = df.set_index("date")["close"]
        ret20_by_sym[sym] = s.pct_change(20) * 100

    wide = pd.DataFrame(ret20_by_sym)  # index=date, columns=symbol
    rank_pct = wide.rank(axis=1, pct=True) * 100   # per-date percentile, 0-100
    median_ret20 = wide.median(axis=1)

    sec_of = {sym: universe.get(sym, {}).get("sector", "") for sym in history}
    sectors = pd.Series(sec_of)
    # per-date, per-sector average ret20 (only over sectors with >=5 cached
    # members that date, matching the real scanner's own threshold)
    sector_avg_by_date = {}
    for date, row in wide.iterrows():
        row = row.dropna()
        if row.empty:
            sector_avg_by_date[date] = set()
            continue
        g = row.groupby(sec_of)
        counts = g.size()
        avgs = g.mean()
        valid = avgs[counts >= 5]
        if len(valid) < 2:
            sector_avg_by_date[date] = set()
            continue
        cut = np.percentile(valid.values, 60)
        sector_avg_by_date[date] = set(valid[valid >= cut].index)

    return rank_pct, median_ret20, sector_avg_by_date, sec_of


def walk_symbol(sym, df, cfg, rank_pct, median_ret20, sector_avg_by_date, sec_of, regime=None):
    """Replay one stock, day by day, using only df.iloc[:k+1] at each step
    (nothing from day k+1 onward exists yet as far as snapshot() is
    concerned) - then, for any day that would have graded A, walk forward
    for real using the SAME win/stop/neither logic already tested and
    shipped for the single-stock replay feature."""
    events = []
    n = len(df)
    for k in range(60, n - 3):
        df_slice = df.iloc[:k + 1]
        s = sc.snapshot(df_slice)
        if s is None:
            continue
        date_k = df.iloc[k]["date"]
        if date_k not in rank_pct.index:
            continue
        rsr = rank_pct.loc[date_k].get(sym, 50.0)
        if pd.isna(rsr):
            rsr = 50.0
        med20 = median_ret20.loc[date_k]
        rel = s["ret20"] - (med20 if not pd.isna(med20) else 0.0)
        sector_strong = sec_of.get(sym, "") in sector_avg_by_date.get(date_k, set())
        res, dist = sc.resistance_info(s, cfg)
        comps = sc.components(s, cfg, rel, sector_strong, dist, rsr)
        hs = sc.horizon_scores(comps)
        best_h = max(hs, key=lambda h: hs[h][0])
        score, _ = hs[best_h]
        fails = sc.hard_filter_fails(s, cfg, rsr)
        if fails or score < cfg["minBreakoutScore"]:
            continue

        # this historical day WOULD have graded A - walk forward for real
        hz = sc.HORIZONS[best_h]
        entry = s["close"]
        a = s["atr"]
        stop = entry - hz["stopATR"] * a
        t1 = entry + hz["t1ATR"] * a
        target_pct = (t1 - entry) / entry * 100
        outcome = "neither"
        days_taken = None
        move_pct = None
        max_reach_pct = 0.0     # real best point reached during the whole window
        max_drawdown_pct = 0.0  # real worst point reached BEFORE the eventual outcome -
                                 # for winners, this is how far they dipped before winning,
                                 # the real minimum safe stop distance
        window_end = min(k + 1 + hz["hold"], n)
        for j in range(k + 1, window_end):
            max_reach_pct = max(max_reach_pct, (df.iloc[j]["high"] - entry) / entry * 100)
            max_drawdown_pct = min(max_drawdown_pct, (df.iloc[j]["low"] - entry) / entry * 100)
            if df.iloc[j]["low"] <= stop:
                outcome = "stop"
                days_taken = j - k
                move_pct = (df.iloc[j]["low"] - entry) / entry * 100  # real, not capped at -stopATR
                break
            if df.iloc[j]["high"] >= t1:
                outcome = "win"
                days_taken = j - k
                move_pct = (df.iloc[j]["high"] - entry) / entry * 100  # real, not capped at +t1ATR
                break
        if outcome == "neither" and window_end > k + 1:
            # resolved neither - still capture the real close-based return,
            # same as the per-stock replay already does for its "neither" bucket
            move_pct = (df.iloc[window_end - 1]["close"] - entry) / entry * 100
        event_regime = regime.get(date_k, "Unknown") if regime is not None else "Unknown"
        events.append(dict(symbol=sym, date=date_k, horizon=best_h, score=score,
                            outcome=outcome, days=days_taken, regime=event_regime,
                            movePct=move_pct, targetPct=round(target_pct, 2),
                            maxReachPct=round(max_reach_pct, 2),
                            maxDrawdownPct=round(max_drawdown_pct, 2)))
    return events


def run_backtest(min_events=5, max_symbols=None, cost_pct=0.15):
    """cost_pct: real round-trip transaction cost estimate, in percent of
    trade value, subtracted from every outcome before computing expectancy.
    Default 0.15% is a reasonable placeholder for NSE delivery equity
    (STT ~0.1% on the sell side + exchange/stamp/GST charges; most discount
    brokers charge ~0 delivery brokerage, so this is mostly statutory
    costs, not commission). Adjust with --cost-pct if your actual broker
    charges differ - this number materially changes whether a thin edge
    is real or not, so don't leave it as an unexamined default."""
    conn = sqlite3.connect(sc.DB_PATH)
    db = sc.read_db_json()
    cfg = db.get("config", {}).get("scanner", {})
    for k, v in sc.DEFAULT_CFG.items():
        cfg.setdefault(k, v)
    for k, v in sc.DEFAULT_CFG["minMovePct"].items():
        cfg["minMovePct"].setdefault(k, v)
    universe = sc.load_universe()

    print("Loading cached price history...")
    history = load_cached_history(conn)
    conn.close()
    if not history:
        print("No cached history found in market.db - run the real scanner "
              "a few times first so there's price history to replay against.")
        return

    if max_symbols:
        history = dict(list(history.items())[:max_symbols])

    print(f"  {len(history)} symbols with enough cached history to test")
    print("Precomputing RS Rank and sector-strength tables (fast, vectorized)...")
    rank_pct, median_ret20, sector_avg_by_date, sec_of = precompute_cross_sectional(history, universe)

    print("Fetching Nifty 50 history for market-regime tagging (Bull/Bear via "
          "its own 50-day EMA)...")
    regime = fetch_nifty_regime_series()
    if regime is None:
        print("  regime fetch unavailable - continuing without regime breakdown")

    print("Walking forward through history (this is the slow part - one real "
          f"snapshot computed per stock per day)...")
    t0 = time.time()
    all_events = []
    for i, (sym, df) in enumerate(history.items()):
        events = walk_symbol(sym, df, cfg, rank_pct, median_ret20, sector_avg_by_date, sec_of, regime)
        all_events.extend(events)
        if (i + 1) % 10 == 0:
            elapsed = time.time() - t0
            per_symbol = elapsed / (i + 1)
            remaining = per_symbol * (len(history) - (i + 1))
            print(f"  ...{i+1}/{len(history)} symbols, {len(all_events)} A-grade "
                  f"events so far ({elapsed:.0f}s elapsed, ~{remaining/60:.1f} min remaining)")
    print(f"Done in {time.time()-t0:.0f}s. {len(all_events)} total historical "
          f"A-grade events found across {len(history)} symbols.\n")

    if not all_events:
        print("No A-grade events found anywhere in the cached history. Either "
              "the cached window is too short, or the current thresholds are "
              "genuinely strict enough that this combination rarely triggers.")
        return

    ev = pd.DataFrame(all_events)

    # the REAL criteria that define "similar setup" for this backtest -
    # built from the actual cfg just used, not a hardcoded mockup, so this
    # stays accurate even if you change Settings and re-run
    criteria = [
        f"Price between Rs{cfg['minPrice']}-{cfg['maxPrice']}",
        f"RSI between {cfg['minRsi']}-{cfg['maxRsi']}",
        "EMA20 above EMA50" + (" and EMA50 above EMA200" if cfg.get("requireEma200") else ""),
        "MACD bullish or improving",
        f"Volume at least {cfg['minVolumeMultiplier']}x its 20-day average",
    ]
    if cfg.get("minRsRank", 0):
        criteria.append(f"RS Rank at least {cfg['minRsRank']} (top {100-cfg['minRsRank']}% vs the rest of the market that day)")
    if cfg.get("minAdx", 0):
        criteria.append(f"ADX at least {cfg['minAdx']}")
    if cfg.get("minAvgVolume", 0):
        criteria.append(f"Average volume at least {cfg['minAvgVolume']/1e5:.1f} lakh shares/day (liquidity floor)")
    criteria.append(f"Overall breakout score at least {cfg['minBreakoutScore']}/100 (the same weighted model used every day)")

    print("=" * 70)
    print("RESULTS BY HORIZON - the REAL, honest question this was built to answer:")
    print("when this exact model said A-grade, what actually happened next?")
    print("=" * 70)
    results_by_horizon = {}
    for h in sorted(ev["horizon"].unique()):
        sub = ev[ev["horizon"] == h]
        n_ev = len(sub)
        if n_ev < min_events:
            print(f"\n{h}: only {n_ev} event(s) - too few to trust, skipping "
                  f"(use --min-events to change this floor)")
            continue
        wins = (sub["outcome"] == "win").sum()
        stops = (sub["outcome"] == "stop").sum()
        neither = (sub["outcome"] == "neither").sum()
        avg_days = sub.loc[sub["outcome"] != "neither", "days"].mean()
        print(f"\n{h} ({sc.HORIZONS[h]['label']}): {n_ev} historical A-grade signals")
        print(f"  Hit target first:  {wins:4d}  ({wins/n_ev*100:5.1f}%)")
        print(f"  Hit stop first:     {stops:4d}  ({stops/n_ev*100:5.1f}%)")
        print(f"  Resolved neither:   {neither:4d}  ({neither/n_ev*100:5.1f}%)")
        if not pd.isna(avg_days):
            print(f"  Avg days to resolve (when it resolved): {avg_days:.1f}")

        result = {"n": int(n_ev), "winRate": round(wins/n_ev*100, 1),
                  "stopRate": round(stops/n_ev*100, 1),
                  "avgDays": round(float(avg_days), 1) if not pd.isna(avg_days) else None}

        # real average gain/loss and expectancy - the actual missing piece
        # from a "prove the strategy" backtest report, using the real move%
        # captured for every event, not the ATR-target threshold
        win_moves = sub.loc[sub["outcome"] == "win", "movePct"].dropna()
        stop_moves = sub.loc[sub["outcome"] == "stop", "movePct"].dropna()
        if len(win_moves) and len(stop_moves):
            avg_win_pct = win_moves.mean()
            avg_loss_pct = stop_moves.mean()
            expectancy = (wins/n_ev * avg_win_pct) + (stops/n_ev * avg_loss_pct)
            worst_dd = sub["movePct"].dropna().min()
            # net-of-cost version: every closed trade pays the round-trip
            # cost regardless of direction, shown separately so neither
            # number silently overwrites the other - the gap between them
            # is itself useful information (how much of the edge is real
            # vs eaten by costs)
            net_expectancy = expectancy - cost_pct
            print(f"  Avg gain when it won:  {avg_win_pct:+.2f}%")
            print(f"  Avg loss when it lost: {avg_loss_pct:+.2f}%")
            print(f"  Expectancy per signal (raw):        {expectancy:+.2f}% "
                  f"(win_rate x avg_win + stop_rate x avg_loss, includes 'neither' as zero weight)")
            print(f"  Expectancy per signal (net of ~{cost_pct}% round-trip cost): {net_expectancy:+.2f}%")
            print(f"  Worst single outcome:  {worst_dd:+.2f}%")

            # Profit Factor and Win/Loss Ratio - standard, well-defined,
            # computed directly from the same real win/stop moves already used above
            gross_profit = win_moves.sum()
            gross_loss = abs(stop_moves.sum())
            profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else None
            win_loss_ratio = (avg_win_pct / abs(avg_loss_pct)) if avg_loss_pct != 0 else None
            if profit_factor is not None:
                print(f"  Profit Factor:         {profit_factor:.2f} (gross profit / gross loss - above 1.0 means the wins outweigh the losses)")
            if win_loss_ratio is not None:
                print(f"  Win/Loss Ratio:        {win_loss_ratio:.2f} (avg win size / avg loss size)")

            # Max Drawdown - a SIMPLIFIED sequential equity curve: every
            # closed event in this horizon, ordered by exit date, one at a
            # time, using its real movePct. This does NOT model running
            # multiple concurrent positions (a real week might have several
            # open at once) - it's the honest single-threaded version, not
            # a full portfolio simulation, and is labeled that way below.
            seq = sub.dropna(subset=["movePct"]).sort_values("date")
            equity = 1.0
            peak = 1.0
            max_dd = 0.0
            for _, row in seq.iterrows():
                equity *= (1 + row["movePct"]/100)
                peak = max(peak, equity)
                dd = (equity - peak) / peak * 100
                max_dd = min(max_dd, dd)
            print(f"  Max drawdown (simplified, one trade at a time, not a full portfolio sim): {max_dd:.2f}%")

            # Sharpe: mean/std of the REAL per-trade returns. This is a
            # per-trade Sharpe, not an annualized one - trades happen at
            # irregular intervals so a clean annualization factor doesn't
            # honestly apply. Labeled as such, not dressed up as more
            # precise than it is.
            trade_returns = sub["movePct"].dropna()
            sharpe = None
            if len(trade_returns) >= 5 and trade_returns.std() > 0:
                sharpe = float(trade_returns.mean() / trade_returns.std())
                print(f"  Per-trade Sharpe (mean/std of trade returns, NOT annualized): {sharpe:.2f}")

            result.update(avgWinPct=round(float(avg_win_pct), 2),
                          avgLossPct=round(float(avg_loss_pct), 2),
                          expectancyPct=round(float(expectancy), 2),
                          netExpectancyPct=round(float(net_expectancy), 2),
                          costPctUsed=cost_pct,
                          medianGainPct=round(float(win_moves.median()), 2),
                          profitFactor=round(float(profit_factor), 2) if profit_factor is not None else None,
                          winLossRatio=round(float(win_loss_ratio), 2) if win_loss_ratio is not None else None,
                          maxDrawdownPct=round(float(max_dd), 2),
                          perTradeSharpe=round(sharpe, 2) if sharpe is not None else None)

        # THE ACTUAL DIAGNOSTIC: for trades that resolved "neither", how
        # close did they really get to target at their best point, versus
        # how far the target actually was? Deliberately OUTSIDE the
        # win/stop block above - a horizon can have zero wins (like this
        # one did) and still have plenty of "neither" trades worth
        # diagnosing; nesting this inside the win-dependent block would
        # have silently skipped exactly the horizon that most needs it.
        neither_sub = sub[sub["outcome"] == "neither"]
        if len(neither_sub) >= 5 and "maxReachPct" in neither_sub.columns and "targetPct" in neither_sub.columns:
            avg_target = neither_sub["targetPct"].mean()
            avg_best_reach = neither_sub["maxReachPct"].mean()
            avg_final = neither_sub["movePct"].mean()
            shortfall_pct = avg_target - avg_best_reach
            reach_ratio = (avg_best_reach / avg_target * 100) if avg_target else None
            print(f"\n  DIAGNOSTIC - the {len(neither_sub)} trades that resolved 'neither':")
            print(f"    Target was aiming for: {avg_target:+.2f}% on average")
            print(f"    They actually reached (best point):  {avg_best_reach:+.2f}% on average"
                  + (f"  ({reach_ratio:.0f}% of the way to target)" if reach_ratio is not None else ""))
            print(f"    Where they ended up (at window close): {avg_final:+.2f}% on average")
            if shortfall_pct > 0:
                print(f"    -> Real shortfall: fell {shortfall_pct:.2f} percentage points short of "
                      f"target at their best point - the target may genuinely be set too far "
                      f"for what these stocks typically do in this window.")
            result["neitherDiagnostic"] = {
                "n": int(len(neither_sub)), "avgTargetPct": round(float(avg_target), 2),
                "avgBestReachPct": round(float(avg_best_reach), 2),
                "avgFinalPct": round(float(avg_final), 2),
                "shortfallPct": round(float(shortfall_pct), 2),
            }

        # THE OTHER HALF OF THE PICTURE: for trades that eventually WON,
        # how far did they dip before winning? Set the stop any tighter
        # than this and you'd shake yourself out of trades that were
        # genuinely going to work. This is the real minimum safe stop
        # distance, not a guess.
        win_sub = sub[sub["outcome"] == "win"]
        if len(win_sub) >= 5 and "maxDrawdownPct" in win_sub.columns:
            # use the 80th percentile of drawdown-before-winning (not the
            # single worst case, which could be one outlier) - covers most
            # real winners without being held hostage by the most extreme one
            winner_dd_80th = abs(win_sub["maxDrawdownPct"].quantile(0.20))  # 20th percentile of (negative) values = 80th percentile of severity
            winner_dd_avg = abs(win_sub["maxDrawdownPct"].mean())
            print(f"\n  DIAGNOSTIC - the {len(win_sub)} trades that eventually WON:")
            print(f"    Average dip before winning: {winner_dd_avg:.2f}%")
            print(f"    80th-percentile dip before winning (covers most real winners): {winner_dd_80th:.2f}%")
            print(f"    -> A stop tighter than ~{winner_dd_80th:.2f}% would have shaken out a meaningful share of these real winners.")
            result["winnerDrawdown"] = {"n": int(len(win_sub)),
                                        "avgDrawdownPct": round(float(winner_dd_avg), 2),
                                        "p80DrawdownPct": round(float(winner_dd_80th), 2)}

            # Propose new stop/target ATR multipliers from these two REAL
            # measurements, then check the result against the real
            # minRiskReward filter before ever suggesting it - the exact
            # check that caught the broken first attempt at this fix.
            if len(neither_sub) >= 5:
                a_est = (sub["targetPct"] / sc.HORIZONS[h]["t1ATR"]).mean() if sc.HORIZONS[h]["t1ATR"] else None
                if a_est:
                    proposed_stop_atr = round(winner_dd_80th / a_est * 1.15, 2)  # 15% margin beyond the 80th-pctile dip
                    proposed_t1_atr = round(max(avg_best_reach, 0.1) / a_est * 1.1, 2)  # slightly above real avg best-reach
                    proposed_rr = round(proposed_t1_atr / proposed_stop_atr, 2) if proposed_stop_atr else None
                    print(f"\n  PROPOSED new values for this horizon (stop widened to survive real winners,")
                    print(f"  target brought to a realistic, achievable level):")
                    print(f"    stopATR: {sc.HORIZONS[h]['stopATR']} -> {proposed_stop_atr}")
                    print(f"    t1ATR:   {sc.HORIZONS[h]['t1ATR']} -> {proposed_t1_atr}")
                    if proposed_rr is not None:
                        min_rr = cfg.get("minRiskReward", 1.2)
                        passes = proposed_rr >= min_rr
                        print(f"    Resulting R:R: {proposed_rr} - "
                              f"{'PASSES' if passes else 'STILL FAILS'} the real minRiskReward >= {min_rr} filter")
                        result["proposedValues"] = {"stopATR": proposed_stop_atr, "t1ATR": proposed_t1_atr,
                                                    "resultingRR": proposed_rr, "passesRRFilter": bool(passes)}

        if (sub["regime"] != "Unknown").any():
            for reg in ["Bull", "Bear"]:
                reg_sub = sub[sub["regime"] == reg]
                if len(reg_sub) < min_events:
                    continue
                reg_wins = (reg_sub["outcome"] == "win").sum()
                print(f"    -> during {reg} regime: {reg_wins}/{len(reg_sub)} "
                      f"({reg_wins/len(reg_sub)*100:.1f}%) hit target, "
                      f"n={len(reg_sub)}")
        results_by_horizon[h] = result

    if results_by_horizon:
        out_path = Path(__file__).parent / "backtest_results.json"
        payload = {"generatedAt": dt.datetime.now().isoformat(timespec="seconds"),
                  "symbolsCovered": len(history), "byHorizon": results_by_horizon,
                  "criteria": criteria}
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nResults saved to {out_path.name} - the main scanner will now show "
              f"this pooled evidence on cards until you re-run this backtest.")

    print("\n" + "=" * 70)
    print("Honest caveats, not fine print - read this before trusting the numbers above:")
    print(f"  - Based on {len(history)} cached symbols, not the full universe -")
    print(f"    RS Rank/sector strength are only as good as what's actually cached.")
    print(f"  - Covers only the date range currently in market.db, whatever that")
    print(f"    happens to be on your machine - check your actual history depth.")
    print(f"  - This tests the CURRENT config thresholds. If you change Settings,")
    print(f"    re-run this to get numbers that match the new thresholds.")
    print(f"  - Market regime is a single real signal (Nifty vs its own 50-day EMA),")
    print(f"    not the fuller volatility/sector/liquidity breakdown - that's more")
    print(f"    scope than this pass covers, and any regime bucket with too few")
    print(f"    events is silently skipped above rather than shown misleadingly.")
    print(f"  - CONSERVATIVE DAILY-BAR EXECUTION ASSUMPTION: with only daily")
    print(f"    open/high/low/close, if a day's range includes both the stop AND")
    print(f"    the target, there's no way to know which was actually hit first.")
    print(f"    This backtest always assumes the stop hit first on such days -")
    print(f"    the conservative assumption, but a real assumption, not a fact.")
    print(f"  - THIS TESTS SIGNALS INDEPENDENTLY, NOT A REAL PORTFOLIO: each event")
    print(f"    asks 'if this stock signaled, what happened' - it does NOT simulate")
    print(f"    running your actual capital with a real position-conflict limit")
    print(f"    (max concurrent positions), and it evaluates stop-vs-target only,")
    print(f"    not the real partial-book-at-T1-then-trail-to-T2 exit lifecycle")
    print(f"    your live positions actually use. A true portfolio-level backtest")
    print(f"    with execution parity is real, valuable future work - not what")
    print(f"    this specific report claims to answer today.")
    print("=" * 70)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--min-events", type=int, default=5)
    p.add_argument("--max-symbols", type=int, default=None,
                    help="limit to first N symbols, for a quick test run")
    p.add_argument("--cost-pct", type=float, default=0.15,
                    help="real round-trip transaction cost estimate (%% of trade value) "
                         "subtracted from expectancy - adjust to match your actual broker/charges")
    args = p.parse_args()
    run_backtest(min_events=args.min_events, max_symbols=args.max_symbols, cost_pct=args.cost_pct)
