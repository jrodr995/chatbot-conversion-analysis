-- Sanitized SQL template. Replace placeholders with environment variables in code.
-- {{DB}}.{{SCHEMA}}.{{TABLE}} should refer to your chats table.

WITH message_data AS (
  SELECT
    c.id,
    c.has_appt_scheduled,
    c.sales_intent,
    c.has_rfi_submission,
    c.start_ts,
    c.message_count,
    m.value AS message_json,
    m.value['author'] AS message_author,
    m.value['text'] AS message_text,
    TO_TIMESTAMP(m.value['timestamp']) AS message_timestamp
  FROM {{DB}}.{{SCHEMA}}.{{TABLE}} c,
  LATERAL FLATTEN(input => messages) m
  WHERE c.sales_intent = 'TRUE'
),
user_messages AS (
  SELECT
    id,
    message_text,
    message_timestamp,
    ARRAY_SIZE(SPLIT(message_text::string, ' ')) AS word_count
  FROM message_data
  WHERE LOWER(message_author::string) = 'user'
    AND message_text IS NOT NULL
    AND TRIM(message_text::string) != ''
),
agent_messages AS (
  SELECT
    id,
    message_text,
    message_timestamp,
    ARRAY_SIZE(SPLIT(message_text::string, ' ')) AS word_count
  FROM message_data
  WHERE LOWER(message_author::string) = 'agent'
    AND message_text IS NOT NULL
    AND TRIM(message_text::string) != ''
),
user_input_features AS (
  SELECT
    id,
    COUNT(*) AS total_user_messages,
    AVG(word_count) AS avg_user_words_per_msg,
    MAX(word_count) AS max_user_words_in_msg,
    MIN(word_count) AS min_user_words_in_msg,
    MIN(message_timestamp) AS first_user_message_ts,
    MAX(message_timestamp) AS last_user_message_ts,
    CASE WHEN COUNT(*) > 1 THEN DATEDIFF('second', MIN(message_timestamp), MAX(message_timestamp)) ELSE 0 END AS user_engagement_duration
  FROM user_messages
  GROUP BY id
),
agent_input_features AS (
  SELECT
    id,
    COUNT(*) AS total_agent_messages,
    AVG(word_count) AS avg_agent_words_per_msg,
    MAX(word_count) AS max_agent_words_in_msg,
    MIN(word_count) AS min_agent_words_in_msg
  FROM agent_messages
  GROUP BY id
),
chat_metadata AS (
  SELECT DISTINCT
    id,
    has_appt_scheduled,
    sales_intent,
    has_rfi_submission,
    start_ts,
    message_count
  FROM message_data
)

SELECT
  u.id,
  u.total_user_messages,
  u.avg_user_words_per_msg,
  u.max_user_words_in_msg,
  u.min_user_words_in_msg,
  u.user_engagement_duration,
  u.first_user_message_ts,
  u.last_user_message_ts,
  COALESCE(a.total_agent_messages, 0) AS total_agent_messages,
  COALESCE(a.avg_agent_words_per_msg, 0) AS avg_agent_words_per_msg,
  COALESCE(a.max_agent_words_in_msg, 0) AS max_agent_words_in_msg,
  COALESCE(a.min_agent_words_in_msg, 0) AS min_agent_words_in_msg,
  t.message_count,
  t.has_appt_scheduled,
  t.sales_intent,
  t.has_rfi_submission,
  t.start_ts
FROM user_input_features u
JOIN chat_metadata t ON u.id = t.id
LEFT JOIN agent_input_features a ON u.id = a.id
ORDER BY u.user_engagement_duration DESC;

