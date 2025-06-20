from datetime import datetime, timedelta

def fetch_and_log_equity():
    """
    Fetch account equity using Alpaca, log equity and daily change based on open,
    update intraday history, and enforce intraday drawdown cap.
    """
    import sys
    main = sys.modules.get("__main__") or sys.modules.get("src.main")
    if main is None or not hasattr(main, 'trade_client'):
        import src.main as main
    try:
        acct = main.trade_client.get_account()
        equity = float(acct.equity)
        # Set today's opening equity on first run
        # and persist to state for cooldown calculations
        # On first run, record opening equity and persist state
        if main.equity_open is None:
            main.equity_open = equity
            try:
                main.save_state(
                    main.LOG_ROOT,
                    main.daily_pnl_accumulated,
                    main.halted,
                    main.last_equity,
                    main.intraday_equity_history,
                    main.daily_halt_time,
                    main.intraday_halt_time,
                    main.equity_open,
                )
            except Exception as e:
                main.log(f"[equity] Error persisting equity_open: {e}")
        # After initial recording, do not modify equity_open again
        main.last_equity = equity
        # Compute change and percentage
        change = equity - main.equity_open
        pct = (change / main.equity_open) if main.equity_open else 0.0
        # Log equity with change%
        entry = f"💰 Account equity: ${equity:,.2f} (Δ ${change:,.2f}, {pct*100:.2f}% today)"
        main.log(entry)
        # Record intraday snapshot and enforce intraday drawdown
        now = datetime.now(tz=main.timezone)
        main.intraday_equity_history.append((now, equity))
        main.check_and_halt_intraday()
    except Exception as e:
        main.log(f"[equity] Error fetching account equity: {e}")
