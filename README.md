Chatbot Conversion Analysis (Sanitized)
======================================

[![CI](https://github.com/jrodr995/chatbot-conversion-analysis/actions/workflows/ci.yml/badge.svg)](https://github.com/jrodr995/chatbot-conversion-analysis/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Made with](https://img.shields.io/badge/Made%20with-Python-3776AB)

Analyze which chat behaviors (user and agent) correlate with RFIs and appointment scheduling, with reproducible statistics and ML.

This public version is fully sanitized:
- No PII or customer content
- No proprietary table names or credentials
- Environment-based configuration
- Synthetic sample data for full local reproducibility

Originally developed as part of an internship analytics project focused on improving conversion outcomes in chat experiences. This repo showcases the methods and engineering, not any proprietary content.

Key features
------------
- Robust engagement duration based on first ↔ last user message
- Parallel user and agent metrics (message counts, avg/max words)
- Statistical testing: point-biserial correlations with p-values
- Logistic Regression with class imbalance handling
- Professional, emoji-free terminal outputs suitable for screenshots

Project structure
-----------------
- `src/` – analysis scripts and common utilities
- `sql/` – SQL template with placeholders (`{{DB}}.{{SCHEMA}}.{{TABLE}}`)
- `data/sample/` – synthetic dataset (generated on first run if missing)
- `figures/` – optional plots/screenshots you save
- `presentation/` – optional presentation script/materials (sanitized)

Setup
-----
1. Python 3.10+ recommended
2. Create a virtual environment and install deps:
   ```bash
   python -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt
   ```
3. Copy `.env.example` to `.env` and edit if you will connect to Snowflake. If you do not set Snowflake env vars, scripts will use the synthetic sample automatically.

Environment variables (Snowflake)
---------------------------------
```
SNOWFLAKE_ENABLED=0            # set to 1 to enable Snowflake path
SNOWFLAKE_USER=your_user
SNOWFLAKE_ACCOUNT=your_account
SNOWFLAKE_WAREHOUSE=WAREHOUSE
SNOWFLAKE_DATABASE=DB
SNOWFLAKE_SCHEMA=SCHEMA
SNOWFLAKE_ROLE=ROLE
SNOWFLAKE_AUTH=externalbrowser
SNOWFLAKE_TABLE=CHATS_TABLE   # e.g., ANALYTICS.PUBLIC.CHATS
```

How to run
----------
- End-to-end analysis (auto sample or Snowflake if enabled):
  ```bash
  python src/chatbot_parsing.py
  ```

- Save ROC and Confusion Matrix images:
  ```bash
  python src/chatbot_parsing.py --save-figures
  # images in ./figures
  ```

- Agent metric correlation tables (screenshot-ready):
  ```bash
  python src/agent_metrics_correlations.py
  ```

- Professional summary outputs (emoji-free):
  ```bash
  python src/sierra_professional_analysis.py
  python src/sequential_professional_summary.py
  python src/correlation_professional_display.py
  ```

- Pattern significance and agent response prediction:
  ```bash
  python src/sequential_pattern_significance.py
  python src/predict_agent_pattern.py
  ```

Notes on data and privacy
-------------------------
- No PII or customer content is included.
- SQL uses placeholders and requires env vars to target your own tables.
- Synthetic sample data mirrors the schema and general distributions for demonstration.

Highlights for reviewers
------------------------
- Clear problem framing around business outcomes (RFIs, appointments)
- Sound statistics (ρ, p-values) and interpretable models
- Class imbalance handled with `class_weight='balanced'`
- Clean CLI outputs for stakeholders
- Chi-square significance of behavior patterns vs appointments
- Multiclass logistic regression predicting agent response patterns

License
-------
MIT. See `LICENSE`.

