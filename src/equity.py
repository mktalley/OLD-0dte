from datetime import datetime

def fetch_and_log_equity():
    """
    Fetch account equity, log changes, update intraday history, and enforce intraday drawdown cap.
    """
    import sys
    main = sys.modules.get("__main__") or sys.modules.get("src.main")
    """
    Fetch account equity, log changes, update intraday history, and enforce intraday drawdown cap.
    """
    # Use existing main module instance without re-importing to avoid double logging init
    import sys
    main = sys.modules.get("__main__")
    # Fallback for test/import contexts
    if main is None or not hasattr(main, "trade_client"):
        import src.main as main
    """
    Fetch account equity, log changes, update intraday history, and enforce intraday drawdown cap.
    """
    # Use existing main module instance without re-importing to avoid double logging init
    main = sys.modules.get("src.main") or sys.modules.get("__main__")
    try:
        acct = main.trade_client.get_account()
        equity = float(acct.equity)
        entry = f"💰 Account equity: ${equity:,.2f}"
        change = None
        pct = None
        # Try Alpaca-provided daily change fields
        if hasattr(acct, 'equity_change') and acct.equity_change is not None and hasattr(acct, 'equity_change_percentage') and acct.equity_change_percentage is not None:
            try:
                change = float(acct.equity_change)
                pct = float(acct.equity_change_percentage)
            except Exception:
                pass
        # Fallback manual computation
        if change is None and main.last_equity is not None:
            change = equity - main.last_equity
            pct = change / main.last_equity if main.last_equity else None
        if change is not None and pct is not None:
            entry += f" (Δ ${change:,.2f}, {pct*100:.2f}% today)"
        main.log(entry)
        # Update last_equity
        main.last_equity = equity
        # Record intraday equity history and enforce drawdown
        now = datetime.now(tz=main.timezone)
        main.intraday_equity_history.append((now, equity))
        main.check_and_halt_intraday()
    except Exception as e:
        main.log(f"[equity] Error fetching account equity: {e}")
