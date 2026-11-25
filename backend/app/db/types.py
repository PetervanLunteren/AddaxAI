"""Column types shared by the models."""

from datetime import UTC, datetime

from sqlalchemy import DateTime
from sqlalchemy.engine import Dialect
from sqlalchemy.types import TypeDecorator


class UtcDateTime(TypeDecorator[datetime]):
    """The type of every ``*_utc`` column. SQLite stores no timezone and
    hands the value back naive; these are all written as UTC
    (``datetime.now(UTC)``), so say so on the way out. Without it the API
    sends no offset and the browser reads the time as local."""

    impl = DateTime(timezone=True)
    cache_ok = True

    def process_result_value(
        self, value: datetime | None, dialect: Dialect
    ) -> datetime | None:
        return value.replace(tzinfo=UTC) if value is not None else None
