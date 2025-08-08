import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass
class SnowflakeConfig:
    enabled: bool = os.getenv("SNOWFLAKE_ENABLED", "0") == "1"
    user: str | None = os.getenv("SNOWFLAKE_USER")
    account: str | None = os.getenv("SNOWFLAKE_ACCOUNT")
    warehouse: str | None = os.getenv("SNOWFLAKE_WAREHOUSE")
    database: str | None = os.getenv("SNOWFLAKE_DATABASE")
    schema: str | None = os.getenv("SNOWFLAKE_SCHEMA")
    role: str | None = os.getenv("SNOWFLAKE_ROLE")
    authenticator: str | None = os.getenv("SNOWFLAKE_AUTH", "externalbrowser")
    table_fqn: str | None = os.getenv("SNOWFLAKE_TABLE")  # Optional fully qualified name


def get_snowflake_config() -> SnowflakeConfig:
    return SnowflakeConfig()
