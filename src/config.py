import sys
from pydantic_settings import BaseSettings
from pydantic import Field, ValidationError, ConfigDict
from typing import Optional

class Settings(BaseSettings):
    # Email settings
    email_host: str = Field("localhost", env="EMAIL_HOST")
    email_port: int = Field(25, env="EMAIL_PORT")
    email_user: Optional[str] = Field(None, env="EMAIL_USER")
    email_pass: Optional[str] = Field(None, env="EMAIL_PASS")
    email_from: str = Field("alerts@example.com", env="EMAIL_FROM")
    email_to: Optional[str] = Field(None, env="EMAIL_TO")

    # Alpaca credentials
    alpaca_api_key: str = Field("", env="ALPACA_API_KEY")
    alpaca_secret_key: str = Field("", env="ALPACA_SECRET_KEY")

    # Risk and filter settings
    stop_loss_percentage: float = Field(0.5, env="STOP_LOSS_PERCENTAGE")
    profit_take_percentage: float = Field(0.5, env="PROFIT_TAKE_PERCENTAGE")
    min_credit_percentage: float = Field(0.15, env="MIN_CREDIT_PERCENTAGE")
    oi_threshold: int = Field(100, env="OI_THRESHOLD")
    strike_range: float = Field(0.1, env="STRIKE_RANGE")
    scan_interval: int = Field(300, env="SCAN_INTERVAL")
    circuit_breaker_threshold: int = Field(5, env="CIRCUIT_BREAKER_THRESHOLD")
    risk_per_trade_percentage: float = Field(0.01, env="RISK_PER_TRADE_PERCENTAGE")
    max_concurrent_trades: int = Field(10, env="MAX_CONCURRENT_TRADES")
    max_total_delta_exposure: float = Field(200, env="MAX_TOTAL_DELTA_EXPOSURE")
    spy_min_credit_percentage: float = Field(0.10, env="SPY_MIN_CREDIT_PERCENTAGE")

    # Drawdown caps
    daily_max_loss: float = Field(500.0, env="DAILY_MAX_LOSS")  # Max loss per day before halting trades
    intraday_max_loss_percentage: float = Field(0.01, env="INTRADAY_MAX_LOSS_PERCENTAGE")  # Max equity drawdown percent in window
    intraday_window_minutes: int = Field(15, env="INTRADAY_WINDOW_MINUTES")  # Lookback window for intraday drawdown (minutes)

    # Other settings
    symbols: str = Field('SPY,SPX,XSP', env="SYMBOLS")  # Allow custom SYMBOLS from .env
    # Daily drawdown percentage cap (halt trading if equity drops ≥ this % from open)
    daily_max_loss_percentage: float = Field(0.10, env="DAILY_MAX_LOSS_PERCENTAGE")
    # Daily take-profit percentage cap (halt trading if equity gains ≥ this % from open)
    daily_max_profit_percentage: float = Field(0.10, env="DAILY_MAX_PROFIT_PERCENTAGE")
    # Daily absolute take-profit cap (halt trading if profit ≥ this $ amount)
    daily_max_profit: float = Field(500.0, env="DAILY_MAX_PROFIT")

    model_config = ConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

# Instantiate settings and validate
try:
    settings = Settings()
except ValidationError as e:
    print("Configuration error:")
    print(e)
    sys.exit(1)
