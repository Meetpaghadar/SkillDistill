
import sqlite3
from pathlib import Path

DB_ROOT = Path("data/spider_data/database")


def get_schema(db_id):
    path = DB_ROOT / db_id / f"{db_id}.sqlite"

    if not path.is_file():
        raise FileNotFoundError(path)

    lines = []

    with sqlite3.connect(
        f"file:{path.resolve()}?mode=ro",
        uri=True,
    ) as conn:
        tables = [
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type='table' "
                "AND name NOT LIKE 'sqlite_%' "
                "ORDER BY name"
            )
        ]

        for table in tables:
            quoted = '"' + table.replace('"', '""') + '"'

            columns = conn.execute(
                f"PRAGMA table_info({quoted})"
            ).fetchall()

            column_text = ", ".join(
                f"{col[1]} {col[2]}" for col in columns
            )

            lines.append(f"{table}({column_text})")

            foreign_keys = conn.execute(
                f"PRAGMA foreign_key_list({quoted})"
            ).fetchall()

            for fk in foreign_keys:
                lines.append(
                    f"FK: {table}.{fk[3]} -> "
                    f"{fk[2]}.{fk[4]}"
                )

    return "\n".join(lines)
