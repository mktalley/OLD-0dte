import os
import json
import pandas as pd
from collections import Counter
from datetime import date
from src.main import send_email, logger


def analyze_logs_for_date(date_str: str) -> None:
    """
    Read the structured debug log and CSV snapshots for a given date,
    produce a summary of events, errors, and PnL metrics,
    and send recommendations via email.
    """
    log_dir = os.path.join("logs", date_str)
    debug_log_path = os.path.join(log_dir, "debug.log")

    events = Counter()
    error_records = []

    # Parse debug log JSON lines
    if os.path.exists(debug_log_path):
        with open(debug_log_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                lvl = rec.get('level')
                evt = rec.get('event')
                if evt:
                    events[evt] += 1
                if lvl == 'ERROR':
                    error_records.append(rec)

    # Build summary text
    summary_lines = [f"Log Analysis for {date_str}"]
    total_events = sum(events.values())
    summary_lines.append(f"Total structured events: {total_events}")
    summary_lines.append("Event counts:")
    for evt, cnt in events.most_common():
        summary_lines.append(f"  - {evt}: {cnt}")
    summary_lines.append(f"Total ERROR-level entries: {len(error_records)}")

    # PnL snapshot analysis
    pnl_csv = os.path.join(log_dir, 'pnl_snapshot.csv')
    if os.path.exists(pnl_csv):
        try:
            df = pd.read_csv(pnl_csv)
            total_pnl = df['pnl'].sum()
            win_rate = (df['pnl'] > 0).mean() * 100 if not df.empty else 0
            cumulative = df.get('cumulative_pnl')
            drawdown = 0
            if cumulative is not None and not df.empty:
                rolling_max = cumulative.cummax()
                drawdown = (rolling_max - cumulative).max()
            summary_lines.append(f"Total PnL: {total_pnl:.2f}")
            summary_lines.append(f"Win rate: {win_rate:.1f}%")
            summary_lines.append(f"Max drawdown: {drawdown:.2f}")
        except Exception as e:
            summary_lines.append(f"Error reading pnl_snapshot.csv: {e}")
    else:
        summary_lines.append("No pnl_snapshot.csv file found.")

    # Generate recommendations
    recommendations = []
    if error_records:
        recommendations.append("Investigate ERROR-level log entries; consider improving error handling.")
    if events.get('api_call_error', 0) > 5:
        recommendations.append("High API error count; check service health and backoff thresholds.")
    if events.get('api_circuit_tripped', 0) > 0:
        recommendations.append("Circuit breaker triggered; review stability or adjust thresholds.")
    if not recommendations:
        recommendations.append("No significant issues detected in logs.")

    summary_lines.append("\nRecommendations:")
    for rec in recommendations:
        summary_lines.append(f"  - {rec}")

    body = "\n".join(summary_lines)
    subject = f"EOD Log Analysis: {date_str}"
    # Send via email
    try:
        send_email(subject, body)
        logger.info(f"EOD log analysis sent for {date_str}", extra={"event": "log_analysis_sent", "date": date_str})
    except Exception as e:
        logger.error(f"Failed to send EOD log analysis: {e}", extra={"event": "log_analysis_error", "error": str(e)})
