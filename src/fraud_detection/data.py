"""Data access layer.

All raw files are exposed as DuckDB views so the 1.2 GB / 13.3M-row transaction
table can be queried out-of-core with SQL. Downstream modules only ever speak to
these views, never to CSV paths directly.

Raw files and quirks they contain (all handled here, once):

* ``amount`` is a currency string (``"$-77.00"``); negatives are refunds/credits.
* ``date`` is a timestamp string.
* ``errors`` is a sparse categorical flag (not junk).
* ``users.per_capita_income`` / ``yearly_income`` / ``total_debt`` / card
  ``credit_limit`` are currency strings too.
* fraud labels ship as ``{"target": {"<txn_id>": "Yes"|"No"}}``.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import duckdb
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .config import Config

# Explicit schemas avoid DuckDB guessing wrong on currency strings / sparse cols.
_TRANSACTION_COLUMNS = {
    "id": "BIGINT",
    "date": "TIMESTAMP",
    "client_id": "BIGINT",
    "card_id": "BIGINT",
    "amount": "VARCHAR",
    "use_chip": "VARCHAR",
    "merchant_id": "BIGINT",
    "merchant_city": "VARCHAR",
    "merchant_state": "VARCHAR",
    "zip": "DOUBLE",
    "mcc": "BIGINT",
    "errors": "VARCHAR",
}

_USER_COLUMNS = {
    "id": "BIGINT",
    "current_age": "INTEGER",
    "retirement_age": "INTEGER",
    "birth_year": "INTEGER",
    "birth_month": "INTEGER",
    "gender": "VARCHAR",
    "address": "VARCHAR",
    "latitude": "DOUBLE",
    "longitude": "DOUBLE",
    "per_capita_income": "VARCHAR",
    "yearly_income": "VARCHAR",
    "total_debt": "VARCHAR",
    "credit_score": "INTEGER",
    "num_credit_cards": "INTEGER",
}

_CARD_COLUMNS = {
    "id": "BIGINT",
    "client_id": "BIGINT",
    "card_brand": "VARCHAR",
    "card_type": "VARCHAR",
    "card_number": "VARCHAR",
    "expires": "VARCHAR",
    "cvv": "VARCHAR",
    "has_chip": "VARCHAR",
    "num_cards_issued": "INTEGER",
    "credit_limit": "VARCHAR",
    "acct_open_date": "VARCHAR",
    "year_pin_last_changed": "INTEGER",
    "card_on_dark_web": "VARCHAR",
}


def money(expr: str) -> str:
    """SQL snippet that turns a currency string column into a DOUBLE."""
    return f"CAST(replace(replace({expr}, '$', ''), ',', '') AS DOUBLE)"


def connect(cfg: Config, memory_limit: str = "4GB") -> duckdb.DuckDBPyConnection:
    """Open an in-memory DuckDB connection tuned for this dataset."""
    con = duckdb.connect()
    con.execute(f"PRAGMA memory_limit='{memory_limit}'")
    con.execute("PRAGMA threads=4")
    tmp = cfg.artifacts.dir / "duckdb_tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    con.execute(f"PRAGMA temp_directory='{tmp.as_posix()}'")
    return con


def _register_csv(
    con: duckdb.DuckDBPyConnection, view: str, path: Path, columns: dict[str, str]
) -> None:
    cols = ", ".join(f"'{k}': '{v}'" for k, v in columns.items())
    con.execute(
        f"CREATE OR REPLACE VIEW {view} AS "
        f"SELECT * FROM read_csv('{path.as_posix()}', "
        f"columns={{{cols}}}, header=true, sample_size=-1)"
    )


def _iter_label_pairs(path: Path) -> Iterator[tuple[int, int]]:
    """Stream ``{"target": {"<txn_id>": "Yes"|"No"}}`` without building the dict.

    A plain ``json.load`` would materialise a 13.3M-entry dict (multiple GB).
    DuckDB's JSON reader builds a single giant object and stalls, so we decode
    key/value pairs incrementally with ``JSONDecoder.raw_decode`` and yield
    ``(id, 0|1)`` pairs instead.
    """
    import json

    text = path.read_text(encoding="utf-8")
    decoder = json.JSONDecoder()
    n = len(text)

    def skip(i: int) -> int:
        while i < n and text[i] in " \t\r\n":
            i += 1
        return i

    i = skip(text.index("{", text.index('"target"')) + 1)
    while i < n and text[i] != "}":
        key, i = decoder.raw_decode(text, skip(i))
        i = skip(i) + 1  # consume ':'
        value, i = decoder.raw_decode(text, skip(i))
        yield int(key), 1 if value == "Yes" else 0
        i = skip(i)
        if i < n and text[i] == ",":
            i = skip(i + 1)


def load_labels(path: Path, cache_path: Path, force: bool = False) -> pa.Table:
    """Return an ``(id, is_fraud)`` Arrow table, caching the parse to Parquet."""
    if cache_path.exists() and not force:
        return pq.read_table(cache_path)

    ids: list[int] = []
    flags: list[int] = []
    for txn_id, flag in _iter_label_pairs(path):
        ids.append(txn_id)
        flags.append(flag)

    table = pa.table({"id": pa.array(ids, type=pa.int64()), "is_fraud": pa.array(flags, type=pa.int8())})
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, cache_path)
    return table


def _register_labels(con: duckdb.DuckDBPyConnection, cfg: Config) -> None:
    """Expose fraud labels as the ``labels(id, is_fraud)`` view.

    ``is_fraud`` is an integer (1 = fraud) for cheap joins; ``create_enriched_view``
    keeps the same integer convention.
    """
    cache_path = cfg.artifacts.dir / "labels.parquet"
    table = load_labels(cfg.data.labels, cache_path)
    con.register("_labels_tbl", table)
    con.execute("CREATE OR REPLACE VIEW labels AS SELECT id, is_fraud FROM _labels_tbl")


def _register_mcc(con: duckdb.DuckDBPyConnection, path: Path) -> None:
    """Merchant category codes are a tiny flat object; load via pandas."""
    import json

    with open(path, encoding="utf-8") as fh:
        codes = json.load(fh)
    mcc_df = pd.DataFrame({"mcc": list(codes.keys()), "mcc_desc": list(codes.values())})
    mcc_df["mcc"] = mcc_df["mcc"].astype("int64")
    con.register("_mcc_df", mcc_df)
    con.execute("CREATE OR REPLACE VIEW mcc AS SELECT * FROM _mcc_df")


def register_views(con: duckdb.DuckDBPyConnection, cfg: Config) -> None:
    """Register ``transactions``, ``users``, ``cards``, ``mcc`` and ``labels``."""
    _register_csv(con, "transactions", cfg.data.transactions, _TRANSACTION_COLUMNS)
    _register_csv(con, "users", cfg.data.users, _USER_COLUMNS)
    _register_csv(con, "cards", cfg.data.cards, _CARD_COLUMNS)
    _register_mcc(con, cfg.data.mcc)
    _register_labels(con, cfg)


def create_enriched_view(con: duckdb.DuckDBPyConnection) -> None:
    """Denormalised transaction fact table: tx + card + user + category + label.

    This is the single source for EDA, features and the dashboard so every
    downstream number is definitionally consistent.
    """
    con.execute(
        f"""
        CREATE OR REPLACE VIEW tx_enriched AS
        SELECT
            t.id                        AS transaction_id,
            t.date                      AS ts,
            {money('t.amount')}         AS amount,
            t.use_chip,
            t.merchant_id,
            t.merchant_city,
            t.merchant_state,
            t.zip,
            t.mcc,
            t.errors,
            m.mcc_desc,
            t.client_id,
            t.card_id,
            u.current_age,
            u.gender,
            u.latitude,
            u.longitude,
            {money('u.per_capita_income')} AS per_capita_income,
            {money('u.yearly_income')}     AS yearly_income,
            {money('u.total_debt')}        AS total_debt,
            u.credit_score,
            u.num_credit_cards,
            c.card_brand,
            c.card_type,
            c.has_chip,
            c.num_cards_issued,
            {money('c.credit_limit')}      AS credit_limit,
            TRY_CAST(c.acct_open_date AS DATE) AS acct_open_date,
            c.year_pin_last_changed,
            c.card_on_dark_web,
            l.is_fraud::INTEGER AS is_fraud
        FROM transactions t
        LEFT JOIN cards  c ON t.card_id  = c.id
        LEFT JOIN users  u ON t.client_id = u.id
        LEFT JOIN mcc    m ON t.mcc       = m.mcc
        LEFT JOIN labels l ON t.id        = l.id
        """
    )


def prepare(cfg: Config, memory_limit: str = "4GB") -> duckdb.DuckDBPyConnection:
    """Convenience: connect, register raw views and build the enriched view."""
    con = connect(cfg, memory_limit=memory_limit)
    register_views(con, cfg)
    create_enriched_view(con)
    return con


def _scalar(con: duckdb.DuckDBPyConnection, sql: str) -> int:
    row = con.execute(sql).fetchone()
    return int(row[0]) if row else 0


def summary(con: duckdb.DuckDBPyConnection) -> dict[str, object]:
    """High-level row counts and fraud base rate, for the README / logs."""
    tx = _scalar(con, "SELECT count(*) FROM transactions")
    frauds = _scalar(con, "SELECT count(*) FROM tx_enriched WHERE is_fraud = 1")
    users = _scalar(con, "SELECT count(*) FROM users")
    cards = _scalar(con, "SELECT count(*) FROM cards")
    return {
        "transactions": tx,
        "users": users,
        "cards": cards,
        "frauds": frauds,
        "fraud_rate": (frauds / tx) if tx else 0.0,
    }
