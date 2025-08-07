from __future__ import annotations

import os
import math
import random
from datetime import datetime, timedelta
from typing import Tuple

import numpy as np
import pandas as pd

try:
    import snowflake.connector  # type: ignore
except Exception:  # pragma: no cover
    snowflake = None  # noqa: N816

from .config import get_snowflake_config


def _read_sql_template() -> str:
    sql_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "sql", "query_template.sql")
    with open(sql_path, "r", encoding="utf-8") as f:
        return f.read()


def _render_sql(cfg) -> str:
    template = _read_sql_template()
    if cfg.table_fqn and all(part in cfg.table_fqn for part in ["."]):
        # If fully qualified provided, split into DB.SCHEMA.TABLE
        db, schema, table = cfg.table_fqn.split(".", 2)
    else:
        db = cfg.database or "DB"
        schema = cfg.schema or "SCHEMA"
        table = (cfg.table_fqn or "CHATS").split(".")[-1]
    return (
        template
        .replace("{{DB}}", db)
        .replace("{{SCHEMA}}", schema)
        .replace("{{TABLE}}", table)
    )


def load_dataframe() -> pd.DataFrame:
    """Load dataframe from Snowflake if enabled; otherwise use or generate synthetic sample.

    Returns a dataframe with the following required columns (uppercase):
    - TOTAL_USER_MESSAGES, AVG_USER_WORDS_PER_MSG, MAX_USER_WORDS_IN_MSG, MIN_USER_WORDS_IN_MSG,
      USER_ENGAGEMENT_DURATION, TOTAL_AGENT_MESSAGES, AVG_AGENT_WORDS_PER_MSG, MAX_AGENT_WORDS_IN_MSG, MIN_AGENT_WORDS_IN_MSG,
      MESSAGE_COUNT, HAS_APPT_SCHEDULED, HAS_RFI_SUBMISSION, START_TS
    """
    cfg = get_snowflake_config()
    if cfg.enabled and snowflake is not None:
        sql = _render_sql(cfg)
        conn = snowflake.connector.connect(
            user=cfg.user,
            account=cfg.account,
            warehouse=cfg.warehouse,
            database=cfg.database,
            schema=cfg.schema,
            authenticator=cfg.authenticator,
            role=cfg.role,
        )
        try:
            df = pd.read_sql(sql, conn)
        finally:
            conn.close()
        df.columns = [c.upper() for c in df.columns]
        return df

    # Fallback: synthetic sample
    sample_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "sample", "sample_chats.csv")
    force_regen = os.getenv("SAMPLE_FORCE_REGENERATE", "0") == "1"
    if os.path.exists(sample_path) and not force_regen:
        df = pd.read_csv(sample_path, parse_dates=["START_TS"])  # type: ignore[arg-type]
        df.columns = [c.upper() for c in df.columns]
        return df

    num_rows = int(os.getenv("SAMPLE_ROWS", "800"))
    df = _generate_synthetic_sample(num_rows=num_rows, seed=42)
    os.makedirs(os.path.dirname(sample_path), exist_ok=True)
    df.to_csv(sample_path, index=False)
    return df


def _generate_synthetic_sample(num_rows: int = 800, seed: int | None = None) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    base_ts = datetime(2025, 5, 26, 8, 0, 0)

    # Baseline rates used for non-explicit requests
    appt_rate = 0.12
    rfi_rate = 0.44

    rows = []
    for i in range(num_rows):
        # Simulate core messaging behavior
        total_user_messages = int(rng.normal(loc=9, scale=4))
        total_user_messages = max(1, total_user_messages)
        avg_user_words = max(2, rng.normal(loc=6.0, scale=2.0))
        max_user_words = int(max(avg_user_words + rng.normal(4, 3), avg_user_words))
        min_user_words = int(max(1, avg_user_words - rng.normal(3, 1)))

        # Agent behavior tied loosely to user messages
        total_agent_messages = int(max(1, rng.normal(loc=total_user_messages * 0.9, scale=2)))
        avg_agent_words = max(2, rng.normal(loc=7.0, scale=2.0))
        max_agent_words = int(max(avg_agent_words + rng.normal(4, 3), avg_agent_words))
        min_agent_words = int(max(1, avg_agent_words - rng.normal(3, 1)))

        # Duration grows with messages, plus noise, clipped at 10-minute artifact + noise tail
        duration = int(max(0, rng.normal(loc=total_user_messages * 20, scale=60)))
        if duration > 1200 and rng.random() < 0.05:
            duration = 600 + int(abs(rng.normal(0, 60)))

        message_count = total_user_messages + total_agent_messages

        # Outcomes with signal: higher messages/duration raise odds (baseline)
        appt_logit = -2.0 + 0.06 * total_user_messages + 0.001 * duration + 0.04 * total_agent_messages
        appt_prob = 1 / (1 + math.exp(-appt_logit))
        appt_baseline = 1 if rng.random() < appt_prob else 0

        rfi_logit = -0.2 + 0.09 * (total_user_messages > 8) + 0.06 * (max_user_words > 12) + 0.001 * duration
        rfi_prob = 1 / (1 + math.exp(-rfi_logit))
        rfi = 1 if rng.random() < rfi_prob else 0

        # Explicit appointment request flag (~18-22%)
        explicit_flag = 1 if rng.random() < rng.uniform(0.18, 0.22) else 0

        # Agent behavior counts and pattern assignment when explicit
        beh_booking = 0
        beh_offers = 0
        beh_info = 0
        beh_deflect = 0
        beh_error = 0
        pattern = "NONE"
        appt = appt_baseline

        if explicit_flag:
            # Assign a pattern by probability mass (sums ~1.0)
            # We bias towards patterns volume similar to earlier analysis
            r = rng.random()
            if r < 0.17:
                pattern = "INFO_THEN_DEFLECTION"
                beh_info = 1
                beh_deflect = 1
                appt = 1 if rng.random() < 0.118 else 0
            elif r < 0.33:
                pattern = "INFO_ONLY"
                beh_info = 1
                appt = 1 if rng.random() < 0.067 else 0
            elif r < 0.48:
                pattern = "BOOKING_THEN_DEFLECTION"
                beh_booking = 1
                beh_deflect = 1
                appt = 1 if rng.random() < 0.146 else 0
            elif r < 0.60:
                pattern = "NO_ACTION"
                appt = 1 if rng.random() < 0.044 else 0
            elif r < 0.71:
                pattern = "COMPLEX_PATTERN"
                beh_booking = rng.integers(0, 2)
                beh_offers = rng.integers(0, 2)
                beh_info = rng.integers(0, 2)
                beh_deflect = rng.integers(0, 2)
                appt = 1 if rng.random() < 0.333 else 0
            elif r < 0.81:
                pattern = "PURE_BOOKING"
                beh_booking = 1
                appt = 1 if rng.random() < 0.167 else 0
            elif r < 0.90:
                pattern = "BOOKING_THEN_INFO"
                beh_booking = 1
                beh_info = 1
                appt = 1 if rng.random() < 0.212 else 0
            elif r < 0.96:
                pattern = "PURE_DEFLECTION"
                beh_deflect = 1
                appt = 1 if rng.random() < 0.0 else 0
            else:
                pattern = "INFO_THEN_BOOKING"
                beh_info = 1
                beh_booking = 1
                appt = 1 if rng.random() < 0.136 else 0

        start_ts = base_ts + timedelta(minutes=int(rng.integers(0, 60 * 24)))

        rows.append({
            "ID": i + 1,
            "TOTAL_USER_MESSAGES": total_user_messages,
            "AVG_USER_WORDS_PER_MSG": float(round(avg_user_words, 2)),
            "MAX_USER_WORDS_IN_MSG": int(max_user_words),
            "MIN_USER_WORDS_IN_MSG": int(min_user_words),
            "USER_ENGAGEMENT_DURATION": int(duration),
            "TOTAL_AGENT_MESSAGES": int(total_agent_messages),
            "AVG_AGENT_WORDS_PER_MSG": float(round(avg_agent_words, 2)),
            "MAX_AGENT_WORDS_IN_MSG": int(max_agent_words),
            "MIN_AGENT_WORDS_IN_MSG": int(min_agent_words),
            "MESSAGE_COUNT": int(message_count),
            "HAS_APPT_SCHEDULED": int(appt),
            "HAS_RFI_SUBMISSION": int(rfi),
            "START_TS": start_ts.strftime("%Y-%m-%d %H:%M:%S"),
            "SALES_INTENT": "TRUE",
            "EXPLICIT_APPT_REQUEST": int(explicit_flag),
            "BEH_BOOKING": int(beh_booking),
            "BEH_OFFERS": int(beh_offers),
            "BEH_INFO": int(beh_info),
            "BEH_DEFLECTION": int(beh_deflect),
            "BEH_ERROR": int(beh_error),
            "SEQUENCE_PATTERN": pattern,
        })

    df = pd.DataFrame(rows)
    df.columns = [c.upper() for c in df.columns]
    return df

