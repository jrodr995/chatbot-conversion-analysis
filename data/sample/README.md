Synthetic Sample Data
=====================

This dataset is generated on first run if Snowflake is disabled. It mirrors the structure used by the analysis scripts, without containing any real messages or PII.

Fields (selected)
-----------------
- TOTAL_USER_MESSAGES, TOTAL_AGENT_MESSAGES: message counts
- AVG_USER_WORDS_PER_MSG, MAX_USER_WORDS_IN_MSG: user message word stats
- AVG_AGENT_WORDS_PER_MSG, MAX_AGENT_WORDS_IN_MSG: agent message word stats
- USER_ENGAGEMENT_DURATION: seconds between first and last user messages
- MESSAGE_COUNT: total messages
- HAS_APPT_SCHEDULED, HAS_RFI_SUBMISSION: binary outcomes
- EXPLICIT_APPT_REQUEST: binary flag indicating explicit appointment intent
- SEQUENCE_PATTERN: one of several agent behavior patterns

Regeneration
------------
- Set `SAMPLE_FORCE_REGENERATE=1` and optionally `SAMPLE_ROWS=<N>` to regenerate
  ```bash
  SAMPLE_FORCE_REGENERATE=1 SAMPLE_ROWS=800 python src/sierra_professional_analysis.py
  ```

