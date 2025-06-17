import os
import json
import schedule
import subprocess
import time as time_module
from datetime import datetime, time as dt_time, date

import sys
# Use the running main module instance (loaded via -m) to avoid double initialization
main = sys.modules.get("__main__")


def job_reset_drawdown():
    """
    Reset daily P&L accumulator and halted flag at midnight, persist state.
    """
    main.daily_pnl_accumulated = 0.0
    main.halted = False
    main.last_equity = None
    # Prepare today's directory and state file
    today_str = date.today().isoformat()
    new_log_dir = os.path.join(main.LOG_ROOT, today_str)
    os.makedirs(new_log_dir, exist_ok=True)
    state_file = os.path.join(new_log_dir, "state.json")
    try:
        with open(state_file, "w") as sf:
            json.dump({"daily_pnl_accumulated": main.daily_pnl_accumulated, "halted": main.halted}, sf)
    except Exception as e:
        main.log(f"[reset] Error persisting state: {e}")
    main.log(f"🔄 Reset daily drawdown state for {today_str}")

# Schedule midnight reset at 00:01 ET
schedule.every().day.at("00:01").do(job_reset_drawdown)


def job_flatten() -> None:
    """
    EOD flatten: snapshot PnL for all positions and then close them out.
    """
    main.log("⚠️ Auto-flattening all positions at EOD")
    positions = main.fetch_positions()
    for p in positions:
        # Fetch mid price
        try:
            quote = main.guarded_get_option_latest_quote(
                main.OptionLatestQuoteRequest(symbol_or_symbols=p.symbol)
            ).get(p.symbol)
            mid = (quote.bid_price + quote.ask_price) / 2
        except Exception:
            mid = float(p.avg_entry_price)
        qty = int(p.qty)
        entry = float(p.avg_entry_price)
        contract_size = 100
        pnl_share = (entry - mid) if p.side == main.PositionSide.SHORT else (mid - entry)
        pnl = pnl_share * qty * contract_size
        pnl_pct = None
        if p.usd and p.usd.unrealized_plpc is not None:
            pnl_pct = p.usd.unrealized_plpc
        # Log PnL snapshot for EOD flatten
        total_unrealized = main.sum_unrealized_pnl(positions)
        cumulative_pnl = main.daily_pnl_accumulated + total_unrealized
        main.write_pnl_snapshot_row([
            datetime.now().isoformat(),
            p.symbol,
            p.side.value if hasattr(p.side, 'value') else p.side,
            qty,
            entry,
            mid,
            pnl_share,
            pnl,
            pnl_pct,
            cumulative_pnl,
        ])

        try:
            resp = main.trade_client.close_position(p.symbol)
            status = getattr(resp, 'status', 'closed')
        except Exception as e:
            status = f"error: {e}"
        main.log_exit(
            p.symbol,
            p.side.value if hasattr(p.side, 'value') else p.side,
            qty,
            mid,
            pnl,
            pnl_pct,
            status,
        )
        main.log(f"⚠️ Flattened {p.symbol} EOD: PnL ${pnl:.2f}")
    main.send_email(
        f"EOD Flatten Report {date.today().isoformat()}",
        f"Auto-flatten executed at EOD. {len(positions)} positions closed."
    )

# Schedule end-of-day flatten at 16:00 ET
schedule.every().day.at("16:00").do(job_flatten)


def job_send_eod():
    main.log("📈 Triggering EOD report")
    try:
        subprocess.run(["python3", "scripts/send_eod_report.py"], check=True)
    except Exception as e:
        main.log(f"Error running EOD report: {e}")

# Schedule end-of-day report at 16:05 ET
schedule.every().day.at("16:05").do(job_send_eod)


def main_loop():
    """
    Main trading loop: run scheduled jobs, manage positions, execute trades.
    """
    main.log("🟢 Bot started")
    TICKERS = main.load_tickers()
    while True:
        schedule.run_pending()
        if main.is_market_open():
            prices = main.get_all_underlying_prices(TICKERS)
            # === POSITION MANAGEMENT ===
            positions = main.fetch_positions()
            for p in positions:
                # Only manage option positions
                if getattr(p, 'asset_class', main.AssetClass.US_OPTION) != main.AssetClass.US_OPTION:
                    continue
                try:
                    quote = main.guarded_get_option_latest_quote(
                        main.OptionLatestQuoteRequest(symbol_or_symbols=p.symbol)
                    ).get(p.symbol)
                except Exception as e:
                    main.log(f"[{p.symbol}] Error fetching quote: {e}")
                    continue
                mid = (quote.bid_price + quote.ask_price) / 2
                qty = int(p.qty)
                entry = float(p.avg_entry_price)
                contract_size = 100
                pnl_share = (entry - mid) if p.side == main.PositionSide.SHORT else (mid - entry)
                pnl = pnl_share * qty * contract_size
                pnl_pct = None
                if p.usd and p.usd.unrealized_plpc is not None:
                    pnl_pct = p.usd.unrealized_plpc
                # Log PnL snapshot (including cumulative PnL)
                total_unrealized = main.sum_unrealized_pnl(positions)
                cumulative_pnl = main.daily_pnl_accumulated + total_unrealized
                main.write_pnl_snapshot_row([
                    datetime.now().isoformat(),
                    p.symbol,
                    p.side.value if hasattr(p.side, 'value') else p.side,
                    qty,
                    entry,
                    mid,
                    pnl_share,
                    pnl,
                    pnl_pct,
                    cumulative_pnl,
                ])
                # Check stop-loss or profit-take
                action = main.should_exit(pnl_pct) if pnl_pct is not None else None
                if action:
                    try:
                        resp = main.trade_client.close_position(p.symbol)
                        status = getattr(resp, 'status', 'closed')
                        main.log_exit(
                            p.symbol,
                            p.side.value if hasattr(p.side, 'value') else p.side,
                            qty,
                            mid,
                            pnl,
                            pnl_pct,
                            status,
                        )
                        # enforce realized PnL update even if log_exit was stubbed
                        main.add_to_daily_pnl(pnl)
                        main.check_and_halt_on_drawdown()
                        msg = f"⚠️ Closed {p.symbol} due to {action.replace('_', ' ')}: PnL ${pnl:.2f}"
                        main.log(msg)
                        main.send_email(f"Position Closed: {p.symbol}", msg)
                    except Exception as e:
                        main.log(f"[{p.symbol}] Error closing position: {e}")
                        continue
            # === NEW TRADES ===
            for symbol in TICKERS:
                if symbol in prices:
                    main.trade(symbol, prices[symbol])
            main.log(f"⏱ Waiting {main.SCAN_INTERVAL // 60} minutes for next scan...")
            time_module.sleep(main.SCAN_INTERVAL)
        else:
            main.log("🔴 Market closed. Sleeping 5 minutes...")
            time_module.sleep(300)
