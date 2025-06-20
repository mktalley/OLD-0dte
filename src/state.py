import os
import json
import logging
from datetime import date

logger = logging.getLogger("0dte")

# Default logs root
LOG_ROOT = "logs"


def load_state(log_root: str = LOG_ROOT):
    """
    Load or initialize today's trading state from state.json.
    Returns tuple:
      daily_pnl_accumulated (float), halted (bool),
      last_equity (float|None), intraday_equity_history (list)
    """
    # Ensure log directory exists
    today_str = date.today().isoformat()
    log_dir = os.path.join(log_root, today_str)
    os.makedirs(log_dir, exist_ok=True)
    state_file = os.path.join(log_dir, "state.json")

    # Default initial values
    daily_pnl_accumulated = 0.0
    halted = False
    last_equity = None
    intraday_equity_history = []

    try:
        if os.path.exists(state_file):
            with open(state_file) as sf:
                state = json.load(sf)
            daily_pnl_accumulated = state.get("daily_pnl_accumulated", 0.0)
            last_equity = state.get("last_equity", None)
            intraday_equity_history = state.get("intraday_equity_history", [])
            # Clear any stale halt flag
            halted = False
            # Persist refreshed state
            with open(state_file, "w") as sf:
                json.dump({
                    "daily_pnl_accumulated": daily_pnl_accumulated,
                    "halted": halted,
                    "last_equity": last_equity,
                    "intraday_equity_history": intraday_equity_history
                }, sf)
            logger.info(
                f"🗄️ Loaded state: pnl=${daily_pnl_accumulated:.2f}, halted={halted}, "
                f"last_equity={last_equity}, intraday_count={len(intraday_equity_history)}"
            )
        else:
            # First run today: write initial state
            with open(state_file, "w") as sf:
                json.dump({
                    "daily_pnl_accumulated": daily_pnl_accumulated,
                    "halted": halted,
                    "last_equity": last_equity,
                    "intraday_equity_history": intraday_equity_history
                }, sf)
            logger.info(
                f"🗄️ Initialized state: pnl=${daily_pnl_accumulated:.2f}, halted={halted}, "
                f"last_equity={last_equity}, intraday_count=0"
            )
    except Exception as e:
        logger.error(f"[init] Error handling state file: {e}")

    return daily_pnl_accumulated, halted, last_equity, intraday_equity_history
