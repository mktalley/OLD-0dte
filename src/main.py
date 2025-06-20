# 0DTE Bot (Fully Optimized)

import os
import csv
import time as time_module

# Ensure scheduling uses Eastern Time for ET-based triggers
os.environ.setdefault("TZ", "America/New_York")
time_module.tzset()
import pandas as pd
import numpy as np
from datetime import datetime, time as dt_time, date, timedelta
from scipy.stats import norm
from scipy.optimize import brentq
from zoneinfo import ZoneInfo
import smtplib
from email.mime.text import MIMEText
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, OrderClass, TimeInForce, AssetStatus, ContractType, AssetClass, PositionSide
from alpaca.trading.requests import GetOptionContractsRequest, OptionLegRequest, LimitOrderRequest
from alpaca.data.historical.option import OptionHistoricalDataClient
from alpaca.data.historical.stock import StockHistoricalDataClient, StockLatestTradeRequest
from alpaca.data.requests import OptionLatestQuoteRequest
from alpaca.data.requests import StockBarsRequest
from typing import List, Any, Dict
from alpaca.data.timeframe import TimeFrame
import json
import logging

import schedule
import functools
# Monkey-patch Logger._log to filter out reserved keys in extra to avoid KeyError on 'args', 'message', etc.
from logging.handlers import TimedRotatingFileHandler
if not getattr(logging.Logger, '_openhands_log_patched', False):
    _orig_log_method = logging.Logger._log

    def _safe_log(self, level, msg, args, exc_info=None, extra=None, stack_info=False, stacklevel=1):
        # filter out reserved names in LogRecord
        if extra:
            reserved = set(logging.LogRecord(self.name if hasattr(self, 'name') else '', level, '', 0, msg, args, exc_info).__dict__.keys()) | {"message", "asctime"}
            extra = {k: v for k, v in extra.items() if k not in reserved} or None
        return _orig_log_method(self, level, msg, args, exc_info, extra, stack_info, stacklevel)

    # Apply the monkey patch
    logging.Logger._log = _safe_log
    logging.Logger._openhands_log_patched = True

# Track last fetched equity for manual daily-change fallback
last_equity: float | None = None
equity_open: float | None = None


def sum_unrealized_pnl(positions: List[Any]) -> float:
    """
    Helper to sum unrealized PnL across positions safely.
    """
    return sum(
        getattr(pos.usd, 'unrealized_pl', 0.0) or 0.0
        for pos in positions
        if getattr(pos, 'usd', None)
    )

# Columns for daily PnL snapshot CSV
PNL_SNAPSHOT_COLUMNS = ["timestamp", "symbol", "side", "qty", "entry", "mid", "pnl_share", "pnl", "pnl_pct", "cumulative_pnl"]



def write_pnl_snapshot_row(row: list):
    """
    Append a PnL snapshot row (with cumulative PnL) to the daily CSV file,
    writing header if the file does not yet exist.
    """
    date_str = datetime.now().date().isoformat()
    file_path = os.path.join(PNL_LOG_DIR, f"pnl_log_{date_str}.csv")
    write_header = not os.path.exists(file_path)
    with open(file_path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(PNL_SNAPSHOT_COLUMNS)
        writer.writerow(row)


def snapshot_positions(positions: List[Any]) -> List[Dict[str, Any]]:
    """
    For a list of positions, fetch mid prices, compute PnL and PnL percent,
    then append a PnL snapshot row per position to daily CSV.
    Returns a list of dicts with snapshot info for each position.
    """
    # Compute total unrealized PnL
    total_unrealized = sum_unrealized_pnl(positions)
    cumulative_pnl = daily_pnl_accumulated + total_unrealized
    timestamp = datetime.now().isoformat()
    snapshots = []
    for p in positions:
        try:
            quote = guarded_get_option_latest_quote(
                OptionLatestQuoteRequest(symbol_or_symbols=p.symbol)
            ).get(p.symbol)
            mid = (quote.bid_price + quote.ask_price) / 2
        except Exception:
            mid = float(p.avg_entry_price)
        qty = int(p.qty)
        entry = float(p.avg_entry_price)
        contract_size = 100
        pnl_share = (entry - mid) if p.side == PositionSide.SHORT else (mid - entry)
        pnl = pnl_share * qty * contract_size
        pnl_pct = None
        if p.usd and p.usd.unrealized_plpc is not None:
            pnl_pct = p.usd.unrealized_plpc
        side_val = p.side.value if hasattr(p.side, 'value') else p.side
        # Write CSV row
        write_pnl_snapshot_row([
            timestamp,
            p.symbol,
            side_val,
            qty,
            entry,
            mid,
            pnl_share,
            pnl,
            pnl_pct,
            cumulative_pnl,
        ])
        snapshots.append({
            'position': p,
            'symbol': p.symbol,
            'side': side_val,
            'qty': qty,
            'mid': mid,
            'pnl': pnl,
            'pnl_pct': pnl_pct,
        })
    return snapshots

from src.config import settings
from tenacity import retry, wait_exponential, stop_after_attempt
# Map settings to global variables
# Risk and filter settings from config
RISK_PER_TRADE_PERCENTAGE = settings.risk_per_trade_percentage
MAX_CONCURRENT_TRADES = settings.max_concurrent_trades
MAX_TOTAL_DELTA_EXPOSURE = settings.max_total_delta_exposure
SPY_MIN_CREDIT_PERCENTAGE = settings.spy_min_credit_percentage

EMAIL_HOST = settings.email_host
EMAIL_PORT = settings.email_port
EMAIL_USER = settings.email_user
EMAIL_PASS = settings.email_pass
EMAIL_FROM = settings.email_from
EMAIL_TO = settings.email_to

API_KEY = settings.alpaca_api_key
API_SECRET = settings.alpaca_secret_key

STOP_LOSS_PERCENTAGE = settings.stop_loss_percentage
PROFIT_TAKE_PERCENTAGE = settings.profit_take_percentage
# Daily take-profit cap (percentage & absolute)
DAILY_MAX_PROFIT_PERCENTAGE = settings.daily_max_profit_percentage
DAILY_MAX_PROFIT = settings.daily_max_profit
DAILY_MAX_LOSS = settings.daily_max_loss
# Daily equity drawdown cap (percentage of opening equity)
DAILY_MAX_LOSS_PERCENTAGE = settings.daily_max_loss_percentage

# === DRAWOWN CAP CONTROL ===

# === INTRADAY DRAWDOWN CONTROL ===
# Tracks equity snapshots to halt if intraday drawdown exceeds threshold
intraday_equity_history: list[tuple[datetime, float]] = []

def check_and_halt_intraday() -> bool:
    """
    Halt trading if equity has fallen more than intraday_max_loss_percentage
    over the past intraday_window_minutes.
    """
    global halted, intraday_halt_time

    """
    Halt trading if equity has fallen more than intraday_max_loss_percentage
    over the past intraday_window_minutes, with cooldown.
    """
    global halted, intraday_halt_time

    """
    Halt trading if equity has fallen more than intraday_max_loss_percentage
    over the past intraday_window_minutes.
    """
    # No intraday control if already halted
    if halted:
        return True
    now = datetime.now(tz=timezone)
    window_start = now - timedelta(minutes=settings.intraday_window_minutes)
    # Filter history within window
    recent = [(t, eq) for t, eq in intraday_equity_history if t >= window_start]
    if not recent:
        return False
    # Current equity is last logged
    curr_eq = recent[-1][1]
    # Peak equity in window
    peak_eq = max(eq for t, eq in recent)
    drawdown = (peak_eq - curr_eq) / peak_eq
    if drawdown >= settings.intraday_max_loss_percentage:
        log(f"⏱️ Intraday drawdown cap hit ({drawdown*100:.2f}% over last {settings.intraday_window_minutes}m) — closing all positions")
        # Attempt bulk close of all positions
        try:
            trade_client.close_all_positions()
            log("⚠️ Executed close_all_positions() due to intraday drawdown cap")
        except Exception as bulk_err:
            log(f"[intraday] Error calling close_all_positions(): {bulk_err}")
        # Fallback: individually close any remaining positions
        try:
            rem_positions = trade_client.get_all_positions()
            for pos in rem_positions:
                try:
                    trade_client.close_position(pos.symbol)
                    log(f"⚠️ Closed position {pos.symbol} due to intraday drawdown cap")
                except Exception as single_err:
                    log(f"[intraday] Error closing position {pos.symbol}: {single_err}")
        except Exception as fetch_err:
            log(f"[intraday] Error fetching positions after drawdown halt: {fetch_err}")
        send_email(
            "Intraday Drawdown Halt",
            f"Equity dropped by {drawdown*100:.2f}% over the last {settings.intraday_window_minutes} minutes. All positions closed."
        )
        # Record intraday halt time for cooldown
        intraday_halt_time = now
        # Persist state after intraday halt
        try:
            save_state(LOG_ROOT, daily_pnl_accumulated, halted, last_equity, intraday_equity_history, daily_halt_time, intraday_halt_time, equity_open)
        except Exception as persist_err:
            log(f"[intraday] Error persisting state after intraday halt: {persist_err}")
        halted = True
    return halted

# Daily halt cooldown period (minutes) after hitting daily drawdown cap
DAILY_HALT_COOLDOWN_MINUTES = 60
# Timestamp when the daily drawdown cap was hit (for cooldown)
daily_halt_time: datetime | None = None
# Intraday halt cooldown period (minutes) after hitting intraday drawdown cap
INTRADAY_HALT_COOLDOWN_MINUTES: int = DAILY_HALT_COOLDOWN_MINUTES
# Timestamp when the intraday drawdown cap was hit (for cooldown)
intraday_halt_time: datetime | None = None

# Tracks cumulative P&L and halts new trades when the daily max loss threshold is hit
# P&L is accumulated when positions are exited via log_exit()
daily_pnl_accumulated = 0.0
halted = False

def add_to_daily_pnl(amount: float):
    global daily_pnl_accumulated
    daily_pnl_accumulated += amount
    # First enforce drawdown cap; if still active, check profit cap
    if not check_and_halt_on_drawdown():
        check_and_halt_on_profit()

def check_and_halt_on_drawdown() -> bool:
    """Halt trading if equity drops more than daily_max_loss_percentage from opening equity.
    Returns True if halted and records halt time for cooldown."""
    global halted, daily_halt_time, equity_open, intraday_halt_time

    # If already halted, do nothing
    if halted:
        return True

    # Require opening equity
    if equity_open is None:
        return False

    # Fetch current equity
    try:
        curr_eq = float(trade_client.get_account().equity)
    except Exception as e:
        log(f"[drawdown] Error fetching account equity: {e}")
        return False

    # Compute drawdown since open
    drawdown_pct = (equity_open - curr_eq) / equity_open

    # Check threshold
    if drawdown_pct >= DAILY_MAX_LOSS_PERCENTAGE:
        log(f"⛔ Daily drawdown cap hit ({drawdown_pct*100:.2f}% ≥ {DAILY_MAX_LOSS_PERCENTAGE*100:.2f}%) — closing all positions")
        try:
            trade_client.close_all_positions()
            log("⚠️ Executed close_all_positions() due to daily drawdown cap")
        except Exception as e:
            log(f"[drawdown] Error closing all positions: {e}")
        send_email(
            "Daily Drawdown Halt",
            f"Daily equity drawdown of {drawdown_pct*100:.2f}% exceeded. All positions closed."
        )
        halted = True
        daily_halt_time = globals()['datetime'].now(tz=globals()['timezone'])
        # Persist state
        try:
            save_state(LOG_ROOT, daily_pnl_accumulated, halted, last_equity, intraday_equity_history, daily_halt_time, intraday_halt_time, equity_open)
        except Exception as e:
            log(f"[drawdown] Error persisting state after halt: {e}")

    return halted


def check_and_halt_on_profit() -> bool:
    """Halt trading if cumulative P&L reaches daily profit cap."""
    global halted, daily_halt_time, equity_open

    # If already halted, do nothing
    if halted:
        return True
    # Need opening equity to evaluate profit
    if equity_open is None:
        return False

    # Cumulative profit (realized)
    profit = daily_pnl_accumulated
    profit_pct = profit / equity_open

    # Check profit cap thresholds
    if profit_pct >= DAILY_MAX_PROFIT_PERCENTAGE or profit >= DAILY_MAX_PROFIT:
        log(f"🎯 Daily profit cap hit ({profit_pct*100:.2f}% ≥ {DAILY_MAX_PROFIT_PERCENTAGE*100:.2f}% or ${profit:.2f} ≥ ${DAILY_MAX_PROFIT:.2f}) — closing all positions")
        try:
            trade_client.close_all_positions()
            log("⚠️ Executed close_all_positions() due to daily profit cap")
        except Exception as e:
            log(f"[profit] Error closing all positions: {e}")

        send_email(
            "Daily Profit Halt",
            f"Profit of {profit_pct*100:.2f}% (=${profit:.2f}) exceeded daily profit target. All positions closed."
        )

        halted = True
        daily_halt_time = globals()['datetime'].now(tz=globals()['timezone'])
        # Persist state after profit halt
        try:
            save_state(LOG_ROOT, daily_pnl_accumulated, halted, last_equity, [], daily_halt_time, intraday_halt_time, equity_open)
        except Exception as e:
            log(f"[profit] Error persisting state after halt: {e}")
        return True

    return False



def can_resume_after_daily_halt() -> bool:
    """
    Returns True if the cooldown period since daily drawdown halt has passed.
    """
    if not halted:
        return True
    if daily_halt_time is None:
        return False
    now = datetime.now(tz=timezone)
    return (now - daily_halt_time) >= timedelta(minutes=DAILY_HALT_COOLDOWN_MINUTES)


def can_resume_after_intraday_halt() -> bool:
    """
    Returns True if the cooldown period since intraday drawdown halt has passed.
    """
    if not halted:
        return True
    if intraday_halt_time is None:
        return False
    now = datetime.now(tz=timezone)
    return (now - intraday_halt_time) >= timedelta(minutes=INTRADAY_HALT_COOLDOWN_MINUTES)


MIN_CREDIT_PERCENTAGE = settings.min_credit_percentage
OI_THRESHOLD = settings.oi_threshold
STRIKE_RANGE = settings.strike_range
SCAN_INTERVAL = settings.scan_interval
CIRCUIT_BREAKER_THRESHOLD = settings.circuit_breaker_threshold

# Symbol-specific override dictionaries
SYMBOL_FILTER_OVERRIDES: dict[str, dict] = {}
SYMBOL_RISK_CAP: dict[str, float] = {}
# Default max risk per trade (dollars): no hard cap
max_risk_per_trade = float('inf')




def send_email(subject, body):
    if not EMAIL_TO:
        return
    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = EMAIL_FROM
    msg["To"] = EMAIL_TO
    try:
        server = smtplib.SMTP(EMAIL_HOST, EMAIL_PORT)
        try:
            server.starttls()
        except Exception:
            pass
        if EMAIL_USER and EMAIL_PASS:
            server.login(EMAIL_USER, EMAIL_PASS)
        server.sendmail(EMAIL_FROM, EMAIL_TO.split(","), msg.as_string())
        server.quit()
        log(f"Email alert sent: {subject}")
    except Exception as e:
        log(f"Failed to send email: {e}")

# === API ERROR HANDLING & CIRCUIT BREAKER ===
_api_failure_count = 0
_circuit_open = False

def api_guard(func):
    """
    Decorator for retry/backoff and circuit breaker on API calls, with debug instrumentation.
    """
    @retry(wait=wait_exponential(min=1, max=30), stop=stop_after_attempt(3), reraise=True)
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        global _api_failure_count, _circuit_open
        # Debug log: call start
        try:
            args_repr = [repr(a) for a in args]
            kwargs_repr = {k: repr(v) for k, v in kwargs.items()}
        except Exception:
            args_repr = []
            kwargs_repr = {}
        logger.debug(f"API call start: {func.__name__}", extra={
            "event": "api_call_start",
            "func": func.__name__,
            "api_args": args_repr,
            "kwargs": kwargs_repr,
            "failure_count": _api_failure_count,
            "circuit_open": _circuit_open
        })
        if _circuit_open:
            msg = f"Circuit breaker open: blocking call to {func.__name__}"
            logger.debug(msg, extra={"event": "api_circuit_open", "func": func.__name__})
            send_email("Circuit Breaker Open", msg)
            raise Exception(msg)
        try:
            result = func(*args, **kwargs)
            _api_failure_count = 0
            logger.debug(f"API call success: {func.__name__}", extra={
                "event": "api_call_success",
                "func": func.__name__,
                "result": repr(result)
            })
            return result
        except Exception as e:
            _api_failure_count += 1
            logger.debug(f"API call error: {func.__name__}: {e}", extra={
                "event": "api_call_error",
                "func": func.__name__,
                "error": str(e),
                "failure_count": _api_failure_count
            })
            send_email(f"API Error: {func.__name__}", f"Error in {func.__name__}: {e}")
            if _api_failure_count >= CIRCUIT_BREAKER_THRESHOLD:
                _circuit_open = True
                send_email("Circuit Breaker Tripped",
                           f"{_api_failure_count} consecutive failures in {func.__name__}")
                logger.debug(f"Circuit breaker tripped for: {func.__name__}", extra={
                    "event": "api_circuit_tripped",
                    "func": func.__name__,
                    "failure_count": _api_failure_count
                })
            raise
    return wrapper
    """
    Decorator for retry/backoff and circuit breaker on API calls.
    """
    @retry(wait=wait_exponential(min=1, max=30), stop=stop_after_attempt(3), reraise=True)
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        global _api_failure_count, _circuit_open
        if _circuit_open:
            msg = f"Circuit breaker open: blocking call to {func.__name__}"
            log(f"❌ {msg}")
            send_email("Circuit Breaker Open", msg)
            raise Exception(msg)
        try:
            result = func(*args, **kwargs)
            _api_failure_count = 0
            return result
        except Exception as e:
            _api_failure_count += 1
            msg = f"API call error in {func.__name__}: {e}"
            log(f"❌ {msg}")
            send_email(f"API Error: {func.__name__}", msg)
            if _api_failure_count >= CIRCUIT_BREAKER_THRESHOLD:
                _circuit_open = True
                send_email("Circuit Breaker Tripped",
                           f"{_api_failure_count} consecutive failures in {func.__name__}")
            raise
    return wrapper

# Wrapped API calls
@api_guard
def guarded_get_stock_latest_trade(req):
    return stock_data_client.get_stock_latest_trade(req)

@api_guard
def guarded_get_option_contracts(req):
    return trade_client.get_option_contracts(req)

@api_guard
def guarded_get_all_positions():
    return trade_client.get_all_positions()

@api_guard
def guarded_submit_order(order):
    return trade_client.submit_order(order)

@api_guard
def guarded_get_option_latest_quote(req):
    return option_data_client.get_option_latest_quote(req)



# Ensure timezone and logger are available for dynamic config
timezone = ZoneInfo("America/Los_Angeles")

# Unified clock entrypoint for business logic; always call module-level datetime
# now() returns current time with timezone, honoring monkey-patched datetime


def now() -> datetime:
    """Return current time with module timezone."""
    return datetime.now(tz=timezone)




def _halt_trade(intraday: bool = False) -> None:
    """Set halted flag and record halt timestamp (daily or intraday)."""
    global halted, daily_halt_time, intraday_halt_time
    halted = True
    ts = now()
    if intraday:
        intraday_halt_time = ts
    else:
        daily_halt_time = ts

# === STRUCTURED LOGGING ===
class JsonLogFormatter(logging.Formatter):
    def format(self, record):
        record_dict = {
            "level": record.levelname,
            "message": record.getMessage(),
        }
        # include custom attributes for structured logging
        for attr in ["symbol", "event", "candidates"]:
            if hasattr(record, attr):
                record_dict[attr] = getattr(record, attr)
        return json.dumps(record_dict)

# Ensure logs directory exists before initializing handlers
os.makedirs("logs", exist_ok=True)
# Configure root logger
logger = logging.getLogger("0dte")
logger.handlers.clear()
logger.propagate = False
logger.setLevel(logging.DEBUG)  # Set to DEBUG for verbose logging
# Deduplicate consecutive log records
class DedupFilter(logging.Filter):
    def __init__(self):
        super().__init__()
        self.last = None
    def filter(self, record):
        msg = record.getMessage()
        if msg == self.last:
            return False
        self.last = msg
        return True
# Apply dedup filter at logger level
logger.addFilter(DedupFilter())
# Convenience wrapper for simple logging calls
def log(message: str, **kwargs):
    logger.info(message, **kwargs)

# Console handler (plain text with timestamp)
console_handler = logging.StreamHandler()
console_formatter = logging.Formatter("[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
console_handler.setFormatter(console_formatter)
logger.addHandler(console_handler)
# === DAILY ROTATING LOG HANDLERS ===
import shutil

class DailyRotatingFileHandler(TimedRotatingFileHandler):
    """
    Keeps active logs under logs/YYYY-MM-DD/orig_filename; on date change, rolls over to a new day directory.
    """
    def __init__(self, orig_filename, when, interval, backupCount, timezone, log_dir="logs"):
        self.orig_filename = orig_filename
        self.timezone = timezone
        self.log_dir = log_dir
        # Ensure top-level log directory exists
        os.makedirs(self.log_dir, exist_ok=True)
        # Initialize current date and ensure today's directory exists
        self.current_date = datetime.now(tz=self.timezone).strftime("%Y-%m-%d")
        self.log_dir_today = os.path.join(self.log_dir, self.current_date)
        os.makedirs(self.log_dir_today, exist_ok=True)
        # Active log file for today
        file_path = os.path.join(self.log_dir_today, self.orig_filename)
        super().__init__(file_path, when=when, interval=interval, backupCount=backupCount)

    def shouldRollover(self, record):
        # Trigger rollover when date has changed
        new_date = datetime.now(tz=self.timezone).strftime("%Y-%m-%d")
        return new_date != self.current_date

    def doRollover(self):
        # Close current log stream
        if self.stream:
            self.stream.close()
            self.stream = None
        # Update to new date directory
        new_date = datetime.now(tz=self.timezone).strftime("%Y-%m-%d")
        new_dir = os.path.join(self.log_dir, new_date)
        os.makedirs(new_dir, exist_ok=True)
        # Update state
        self.current_date = new_date
        self.log_dir_today = new_dir
        # Update baseFilename to new log file path
        new_file_path = os.path.join(self.log_dir_today, self.orig_filename)
        self.baseFilename = new_file_path
        # Reopen the stream
        # Open a new stream for the new log file
        self.stream = self._open()
        self.stream = self._open()

# JSON daily log handler
file_handler = DailyRotatingFileHandler(
    orig_filename="0dte.log",
    when="midnight",
    interval=1,
    backupCount=30,
    timezone=timezone,
)
file_handler.setFormatter(JsonLogFormatter())
logger.addHandler(file_handler)

# Human-readable daily log handler (Pacific Time)
class PacificFormatter(logging.Formatter):
    def formatTime(self, record, datefmt=None):
        ct = datetime.fromtimestamp(record.created, tz=timezone)
        if datefmt:
            return ct.strftime(datefmt)
        return ct.isoformat()
human_handler = DailyRotatingFileHandler(
    orig_filename="0dte_human.log",
    when="midnight",
    interval=1,
    backupCount=30,
    timezone=timezone,
)
human_formatter = PacificFormatter(
    "[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S %Z",
)
human_handler.setFormatter(human_formatter)
human_handler.setLevel(logging.INFO)
logger.addHandler(human_handler)

logger.info("🚀 Log handlers initialized: JSON and human-readable (PST)")

# === DEBUG LOG HANDLER ===
# Dedicated debug-level daily logs for full traceability
debug_handler = DailyRotatingFileHandler(
    orig_filename="debug.log",
    when="midnight",
    interval=1,
    backupCount=30,
    timezone=timezone,
)
debug_handler.setLevel(logging.DEBUG)
debug_handler.setFormatter(JsonLogFormatter())
logger.addHandler(debug_handler)
logger.info("🔍 Debug log handler initialized: debug.log")

# === AGGRESSIVE FILTERS FOR HIGHER P&L ===
MIN_CREDIT_PERCENTAGE = 0.10          # allow credit ≥10% of width
OI_THRESHOLD = 50                     # include strikes with OI down to 50
SHORT_PUT_DELTA_RANGE = (-0.7, -0.5)  # deeper OTM for short leg
LONG_PUT_DELTA_RANGE = (-0.5, -0.3)   # deeper OTM for long leg
STRIKE_RANGE = settings.strike_range  # ±30% from spot for strike scan

# Scan every 1 minute (60s) instead of default to catch more delta swings
SCAN_INTERVAL = 60
risk_free_rate = 0.01


# Default to paper trading
PAPER = True
# Override base URL for paper trading endpoint
paper_api_base_url = os.getenv("ALPACA_API_BASE_URL", None) or "https://paper-api.alpaca.markets"
# Stub clients for when Alpaca credentials not provided
class _NoOpTradeClient:
    """No-op trade client when Alpaca credentials are missing"""
    def get_all_positions(self):
        return []
    def close_all_positions(self):
        pass
    def close_position(self, symbol):
        # return dummy response
        return type('Response', (), {'status': 'no-op'})

class _NoOpOptionDataClient:
    """No-op option data client when credentials are missing"""
    def get_option_latest_quote(self, req):
        raise Exception("NoOpOptionDataClient: no credentials")

class _NoOpStockDataClient:
    """No-op stock data client when credentials are missing"""
    pass

# === CLIENTS ===
if API_KEY and API_SECRET:
    trade_client = TradingClient(API_KEY, API_SECRET, paper=PAPER, url_override=paper_api_base_url)
    option_data_client = OptionHistoricalDataClient(API_KEY, API_SECRET)
    stock_data_client = StockHistoricalDataClient(API_KEY, API_SECRET)
else:
    trade_client = _NoOpTradeClient()
    option_data_client = _NoOpOptionDataClient()
    stock_data_client = _NoOpStockDataClient()

# === LOG FILES ===
# Ensure root logs directory exists
LOG_ROOT = "logs"
os.makedirs(LOG_ROOT, exist_ok=True)
# Create today's log directory under the root
today_str = date.today().isoformat()
LOG_DIR_TODAY = os.path.join(LOG_ROOT, today_str)
os.makedirs(LOG_DIR_TODAY, exist_ok=True)
STATE_FILE = os.path.join(LOG_DIR_TODAY, "state.json")

# === LOAD OR INITIALIZE DAILY STATE ===
from src.state import load_state, save_state
# Load or initialize today's trading state (includes intraday_halt_time & equity_open)
daily_pnl_accumulated, halted, last_equity, intraday_equity_history, daily_halt_time, intraday_halt_time, equity_open = load_state(LOG_ROOT)


# === DAILY CSV LOG FILES ===
EXIT_LOG = os.path.join(LOG_DIR_TODAY, "exit_log.csv")
if not os.path.exists(EXIT_LOG):
    with open(EXIT_LOG, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "symbol", "side", "qty", "exit_price", "pnl", "ratio", "status"])

OPEN_POS_LOG = os.path.join(LOG_DIR_TODAY, "open_positions.csv")
if not os.path.exists(OPEN_POS_LOG):
    with open(OPEN_POS_LOG, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "symbol", "side", "qty", "avg_entry_price", "cost_basis", "unrealized_pl", "unrealized_plpc"])

TRADE_LOG = os.path.join(LOG_DIR_TODAY, "trade_log.csv")
if not os.path.exists(TRADE_LOG):
    with open(TRADE_LOG, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "symbol", "short_strike", "long_strike", "credit", "width", "status"])

# PnL log rotates daily under the same directory
PNL_LOG_DIR = LOG_DIR_TODAY
daily_pnl_file = os.path.join(PNL_LOG_DIR, f"pnl_log_{today_str}.csv")
if not os.path.exists(daily_pnl_file):
    with open(daily_pnl_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(PNL_SNAPSHOT_COLUMNS)
# === POSITION MANAGEMENT ===

# === POSITION MANAGEMENT ===
def fetch_positions():
    """
    Fetch all open positions, record them to CSV, and return the list.
    """
    try:
        positions = trade_client.get_all_positions()
    except Exception as e:
        send_email("API Error: fetch_positions failed", f"Error fetching positions: {e}")
        return []
    # Record to CSV
    with open(OPEN_POS_LOG, "a", newline="") as f:
        writer = csv.writer(f)
        for p in positions:
            # Basic position fields
            writer.writerow([
                datetime.now().isoformat(),
                p.symbol,
                p.side.value if hasattr(p.side, 'value') else p.side,
                p.qty,
                p.avg_entry_price,
                p.cost_basis,
                (p.usd.unrealized_pl if p.usd and p.usd.unrealized_pl is not None else ""),
                (p.usd.unrealized_plpc if p.usd and p.usd.unrealized_plpc is not None else ""),
            ])
    return positions

def log_exit(symbol, side, qty, exit_price, pnl, ratio, status):
    """
    Log an exit or hedge trade to CSV (writes exit_log.csv).
    """
    """
    Log an exit or hedge trade to CSV.
    """
    """
    Log an exit or hedge trade to CSV and update daily PnL accumulator.
    """
    with open(EXIT_LOG, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.now().isoformat(),
            symbol,
            side,
            qty,
            exit_price,
            pnl,
            ratio,
            status,
        ])
    # Update daily P&L and enforce drawdown cap
    try:
        add_to_daily_pnl(pnl)
    except Exception as e:
        log(f"[drawdown] Error updating daily PnL accumulator: {e}")

TRADE_LOG = "logs/trade_log.csv"

# === UTILITIES ===

def safe_divide(a, b):
    return a / b if b else 0

def calculate_iv(option_price, S, K, T, r, option_type):
    intrinsic = max(0, (S - K) if option_type == 'call' else (K - S))
    if option_price <= intrinsic + 1e-6:
        return 0.0
    def f(sigma):
        d1 = safe_divide((np.log(S / K) + (r + 0.5 * sigma**2) * T), sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        if option_type == 'call':
            return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2) - option_price
        else:
            return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1) - option_price
    try:
        return brentq(f, 1e-6, 5.0)
    except:
        return None

def calculate_delta(option_price, strike, expiry, spot, r, option_type):
    # Handle naive and aware expiry datetimes
    if expiry.tzinfo is None:
        expiry = expiry.replace(tzinfo=timezone)
    else:
        expiry = expiry.astimezone(timezone)
    now = datetime.now(tz=timezone)
    T = max((expiry - now).total_seconds() / (365 * 24 * 3600), 1e-6)
    iv = calculate_iv(option_price, spot, strike, T, r, option_type)
    if not iv:
        return None
    d1 = safe_divide((np.log(spot / strike) + (r + 0.5 * iv**2) * T), iv * np.sqrt(T))
    return norm.cdf(d1) if option_type == 'call' else -norm.cdf(-d1)



def calculate_num_contracts(equity: float, width: float, max_risk_per_trade: float) -> int:
    """
    Calculate number of contracts based on equity, spread width, and max risk per trade.
    """
    risk_amt = equity * RISK_PER_TRADE_PERCENTAGE
    risk_cap_amt = min(max_risk_per_trade, risk_amt)
    return int(risk_cap_amt / (width * 100))


def calculate_pnl(entry_price: float, mid_price: float, qty: int, side, contract_size: int = 100) -> float:
    """
    Calculate PnL given entry price, current mid price, quantity, side, and contract size.
    """
    from alpaca.trading.enums import PositionSide
    # PnL per share
    pnl_share = (entry_price - mid_price) if side == PositionSide.SHORT else (mid_price - entry_price)
    return pnl_share * qty * contract_size


def should_exit(pnl_pct: float) -> str | None:
    """
    Determine if position should exit based on pnl percentage thresholds.
    Returns 'stop_loss', 'profit_take', or None.
    """
    if pnl_pct <= -STOP_LOSS_PERCENTAGE:
        return 'stop_loss'
    if pnl_pct >= PROFIT_TAKE_PERCENTAGE:
        return 'profit_take'
    return None

def is_market_open():
    # Check US equity market hours in Eastern Time
    now_et = datetime.now(tz=ZoneInfo("America/New_York"))
    return now_et.weekday() < 5 and dt_time(9, 30) <= now_et.time() <= dt_time(16, 0)

def get_fallback_tickers():
    # Full default universe
    return ["SPY", "QQQ", "TSLA", "AAPL", "MSFT", "NVDA", "META", "AMZN", "AMD", "GOOG",
            "BA", "XLF", "XLK", "DIA", "IWM", "XLE", "XBI", "TSM", "GDX", "ARKK"]

def load_tickers():
    today_str = date.today().isoformat()
    filename = f"tickers_selected/tickers_selected_{today_str}.txt"
    if os.path.exists(filename):
        with open(filename) as f:
            return [line.strip() for line in f if line.strip()]
    log("⚠️ No ticker file for today. Using fallback list.")
    return get_fallback_tickers()

def get_all_underlying_prices(tickers):
    try:
        req = StockLatestTradeRequest(symbol_or_symbols=tickers)
        res = stock_data_client.get_stock_latest_trade(req)
        return {symbol: res[symbol].price for symbol in tickers if symbol in res}
    except Exception as e:
        log(f"API Error in get_all_underlying_prices: {e}")
        send_email("API Error: get_all_underlying_prices failed", str(e))
        return {}


def get_market_regime():
    """
    Compute recent realized volatility and trend on SPY to determine filter overrides.
    """
    # Fetch last 20 daily bars for SPY
    try:
        resp = stock_data_client.get_bars(
            StockBarsRequest(
                symbol_or_symbols=["SPY"],
                timeframe=TimeFrame.Day,
                start=None,
                end=None,
                limit=20
            )
        ).data.get("SPY", [])
    except Exception:
        return {}
    closes = [bar.close for bar in resp]
    if len(closes) < 2:
        return {}
    returns = np.diff(closes) / closes[:-1]
    realized_vol = np.std(returns) * (252 ** 0.5)
    trend = (closes[-1] / closes[0] - 1)

    # Base aggressive filters
    overrides = {
        "MIN_CREDIT_PERCENTAGE": 0.10,
        "OI_THRESHOLD": 50,
        "SHORT_PUT_DELTA_RANGE": (-0.7, -0.5),
        "LONG_PUT_DELTA_RANGE": (-0.5, -0.3),
        "SCAN_INTERVAL": 120,
    }
    # High vol: require more credit and OI
    if realized_vol > 0.25:
        overrides["MIN_CREDIT_PERCENTAGE"] = 0.12
        overrides["OI_THRESHOLD"] = 100
    # Strong uptrend: shallower strikes
    if trend > 0.02:
        overrides["SHORT_PUT_DELTA_RANGE"] = (-0.6, -0.4)
        overrides["LONG_PUT_DELTA_RANGE"] = (-0.4, -0.2)
    # Downtrend: deeper OTM but more credit
    if trend < -0.02:
        overrides["SHORT_PUT_DELTA_RANGE"] = (-0.8, -0.6)
        overrides["LONG_PUT_DELTA_RANGE"] = (-0.6, -0.4)
        overrides["MIN_CREDIT_PERCENTAGE"] = 0.15
    return overrides



def get_0dte_options(symbol):
    # Debug: start scanning options
    logger.debug(f"Options scan start for {symbol}", extra={"event":"options_scan_start","symbol":symbol})
    spot = get_all_underlying_prices([symbol]).get(symbol)
    # Debug: spot price
    logger.debug(f"Spot price for {symbol}: {spot}", extra={"event":"options_scan_spot","symbol":symbol,"spot":spot})
    if not spot:
        logger.debug(f"No spot price for {symbol}, returning empty list", extra={"event":"options_scan_no_spot","symbol":symbol})
        return []
    min_strike = str(spot * (1 - STRIKE_RANGE))
    max_strike = str(spot * (1 + STRIKE_RANGE))
    # Debug: scan parameters
    logger.debug(f"Option scan parameters for {symbol}: min_strike={min_strike}, max_strike={max_strike}, oi_threshold={OI_THRESHOLD}", extra={"event":"options_scan_params","symbol":symbol,"min_strike":min_strike,"max_strike":max_strike,"oi_threshold":OI_THRESHOLD})
    spot = get_all_underlying_prices([symbol]).get(symbol)
    if not spot: return []
    min_strike = str(spot * (1 - STRIKE_RANGE))
    max_strike = str(spot * (1 + STRIKE_RANGE))
    req = GetOptionContractsRequest(
        underlying_symbols=[symbol],
        strike_price_gte=min_strike,
        strike_price_lte=max_strike,
        expiration_date=date.today(),
        status=AssetStatus.ACTIVE,
        root_symbol=symbol,
        type=ContractType.PUT,
    )
    contracts = trade_client.get_option_contracts(req).option_contracts
    if len(contracts) < 5:
        log(f"⚠️ Low contract count for {symbol}, retrying...")
    # Apply symbol-specific filter overrides
    overrides = SYMBOL_FILTER_OVERRIDES.get(symbol, {})
    min_credit_pct = overrides.get("MIN_CREDIT_PERCENTAGE", MIN_CREDIT_PERCENTAGE)
    short_delta_range = overrides.get("SHORT_PUT_DELTA_RANGE", SHORT_PUT_DELTA_RANGE)
    long_delta_range = overrides.get("LONG_PUT_DELTA_RANGE", LONG_PUT_DELTA_RANGE)

    return contracts

def log_trade(symbol, short_strike, long_strike, credit, spread_width, status):
    with open(TRADE_LOG, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([datetime.now().isoformat(), symbol, short_strike, long_strike, credit, spread_width, status])

def trade(symbol, spot):
    global halted, daily_halt_time, intraday_halt_time, equity_open
    # Record opening equity at first executed trade
    if equity_open is None:
        try:
            acct = trade_client.get_account()
            equity_open = float(acct.equity)
            log(f"🎬 First trade – recorded opening equity ${equity_open:,.2f}")
        except Exception as e:
            log(f"[trade] Error fetching opening equity: {e}")
    # If currently halted, attempt daily or intraday cooldown
    if halted:
        # Daily cooldown
        if daily_halt_time is not None:
            if can_resume_after_daily_halt():
                halted = False
                daily_halt_time = None
                log("🔄 Cooldown complete—resuming trading after daily drawdown cap")
            else:
                log(f"[{symbol}] Skipped trade: halted due to daily drawdown cap (cooldown)")
                return
        # Intraday cooldown
        if halted and intraday_halt_time is not None:
            if can_resume_after_intraday_halt():
                halted = False
                intraday_halt_time = None
                log("🔄 Cooldown complete—resuming trading after intraday drawdown cap")
            else:
                log(f"[{symbol}] Skipped trade: halted due to intraday drawdown cap (cooldown)")
                return
        # Still halted
        if halted:
            log(f"[{symbol}] Skipped trade: halted, awaiting reset")
            return
    # Pre-trade intraday drawdown check
    check_and_halt_intraday()
    # If currently halted, check for daily cooldown or skip
    if halted:
        if daily_halt_time is not None:
            if can_resume_after_daily_halt():
                halted = False
                daily_halt_time = None
                log("🔄 Cooldown complete—resuming trading after daily drawdown cap")
            else:
                log(f"[{symbol}] Skipped trade: halted due to daily drawdown cap (cooldown)")
                return
        else:
            log(f"[{symbol}] Skipped trade: halted, awaiting reset")
            return
    if halted:
        log(f"[{symbol}] Skipped trade: halted due to daily drawdown cap")
        return
    # Pre-trade intraday drawdown check
    check_and_halt_intraday()
    if halted:
        log(f"[{symbol}] Skipped trade: halted due to intraday drawdown cap")
        return
    # === Symbol-specific filter overrides ===
    static_overrides = SYMBOL_FILTER_OVERRIDES.get(symbol, {})
    live_overrides = get_market_regime() if symbol == "SPY" else {}
    overrides = {**static_overrides, **live_overrides}
    threshold = overrides.get("OI_THRESHOLD", OI_THRESHOLD)
    short_range = overrides.get("SHORT_PUT_DELTA_RANGE", SHORT_PUT_DELTA_RANGE)
    long_range = overrides.get("LONG_PUT_DELTA_RANGE", LONG_PUT_DELTA_RANGE)
    min_credit_pct = overrides.get("MIN_CREDIT_PERCENTAGE", MIN_CREDIT_PERCENTAGE)
    # Initialize candidate logging for SPY
    candidates = [] if symbol == "SPY" else None

    options = get_0dte_options(symbol)
    short_put = long_put = None
    for opt in options:
        if not opt.open_interest or int(opt.open_interest) < threshold:
            continue
        try:
            quote_data = guarded_get_option_latest_quote(
                OptionLatestQuoteRequest(symbol_or_symbols=opt.symbol)
            )
            quote = quote_data.get(opt.symbol)
        except Exception as e:
            log(f"[{symbol}] Error fetching option latest quote for {opt.symbol}: {e}")
            continue
        if not quote or not getattr(quote, 'bid_price', None) or not getattr(quote, 'ask_price', None):
            continue
        price = (quote.bid_price + quote.ask_price) / 2
        expiry = datetime.combine(opt.expiration_date, dt_time(16, 0)).replace(tzinfo=timezone)
        delta = calculate_delta(price, float(opt.strike_price), expiry, spot, risk_free_rate, 'put')
        if delta is None:
            continue
        # Candidate logging for SPY
        if symbol == "SPY":
            candidate = {"option_symbol": opt.symbol, "open_interest": int(opt.open_interest), "delta": delta, "mid_price": price}
            
            candidates.append(candidate)
        # Delta filter
        if short_range[0] <= delta <= short_range[1]:
            short_put = (opt, price)
        elif long_range[0] <= delta <= long_range[1]:
            long_put = (opt, price)
        if short_put and long_put:
            break
    # Emit full candidate list for SPY before selection
    if symbol == "SPY":
        logger.debug(f"Candidates for {symbol}: {candidates}", extra={"event":"candidates", "symbol": symbol, "candidates": candidates})
    if not short_put or not long_put:
        log(f"[{symbol}] No valid spread found.")
        return
    credit = short_put[1] - long_put[1]
    width = abs(float(short_put[0].strike_price) - float(long_put[0].strike_price))
    min_credit = min_credit_pct * width
    if credit < min_credit:
        log_trade(symbol, short_put[0].strike_price, long_put[0].strike_price, credit, width, "rejected")
        return
    # Determine risk cap (symbol-specific override or default)
    cap = SYMBOL_RISK_CAP.get(symbol, max_risk_per_trade)
    if width * 100 > cap:
        log(f"[{symbol}] Skipped: spread too large (${width * 100:.2f} > ${cap:.2f})")
        return
    # === RISK SIZING ===
    try:
        account = trade_client.get_account()
        equity = float(account.equity)
    except Exception as e:
        log(f"[{symbol}] Error fetching account: {e}")
        return
    risk_amt = equity * RISK_PER_TRADE_PERCENTAGE
    risk_cap_amt = min(cap, risk_amt)
    num_contracts = int(risk_cap_amt / (width * 100))
    if num_contracts < 1:
        log(f"[{symbol}] Skipped: risk per trade too small for width ${width:.2f}")
        return
    # === CONCURRENCY LIMIT ===
    try:
        positions = trade_client.get_all_positions()
    except Exception as e:
        log(f"[{symbol}] Error fetching positions: {e}")
        return
    option_positions = [p for p in positions if p.asset_class == AssetClass.US_OPTION]
    num_spreads = len(option_positions) // 2
    if num_spreads >= MAX_CONCURRENT_TRADES:
        log(f"[{symbol}] Skipped: max concurrent trades reached ({MAX_CONCURRENT_TRADES})")
        return
    # === DELTA EXPOSURE ===
    short_delta = delta
    long_delta = calculate_delta(long_put[1], float(long_put[0].strike_price), expiry, spot, risk_free_rate, 'put')
    new_delta = (short_delta + long_delta) * 100 * num_contracts
    if abs(new_delta) > MAX_TOTAL_DELTA_EXPOSURE:
        log(f"[{symbol}] Skipped: trade delta exposure {new_delta:.2f} exceeds max {MAX_TOTAL_DELTA_EXPOSURE}")
        return

    try:
        order = LimitOrderRequest(
            qty=num_contracts,
            limit_price=round(credit, 2),
            order_class=OrderClass.MLEG,
            time_in_force=TimeInForce.DAY,
            legs=[
                OptionLegRequest(symbol=short_put[0].symbol, side=OrderSide.SELL, ratio_qty=1),
                OptionLegRequest(symbol=long_put[0].symbol, side=OrderSide.BUY, ratio_qty=1),
            ],
        )
        trade_client.submit_order(order)
        log_trade(symbol, short_put[0].strike_price, long_put[0].strike_price, credit, width, "submitted")
        log(f"✅ {symbol} Spread placed: Credit ${credit:.2f}, Width ${width:.2f}")
    except Exception as e:
        log_trade(symbol, short_put[0].strike_price, long_put[0].strike_price, credit, width, "submission_failed")
        log(f"[{symbol}] Error submitting order: {e}")


# Equity snapshot scheduling
from src.equity import fetch_and_log_equity



# Fetch and log account equity every minute
schedule.every().minute.do(fetch_and_log_equity)



if __name__ == "__main__":
    import argparse
    import sys
    import os
    import runpy

    parser = argparse.ArgumentParser(description="0DTE Bot: backtest and trading for put credit and strangle strategies")
    parser.add_argument('--backtest', action='store_true', help='Run 0DTE backtest for the past year')
    parser.add_argument('--strangle', action='store_true', help='Use strangle strategy (requires --backtest for backtesting)')
    parser.add_argument('--start', help='Start date YYYY-MM-DD (optional for backtests)')
    parser.add_argument('--end', help='End date YYYY-MM-DD (optional for backtests)')
    args = parser.parse_args()

    if args.backtest:
        # Backtest mode
        if args.strangle:
            script = 'run_backtest_strangle_spy_last_year.py'
        else:
            script = 'run_backtest_spy_last_year.py'
        script_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'scripts', script))
        # Reset sys.argv for the backtest script to avoid passing main.py args
        sys.argv = [script_path]
        runpy.run_path(script_path, run_name='__main__')
        sys.exit(0)
    elif args.strangle:
        # Live strangle trading bot
        script_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'scripts', 'trading_bot.py'))
        runpy.run_path(script_path, run_name='__main__')
        sys.exit(0)
    else:
        # Live put credit spread bot
        log("✅ Starting live 0DTE put credit spread bot now; it will sleep every 5 minutes until market open at 09:30 ET if closed.")
        from src.trading_loop import main_loop
        main_loop()
