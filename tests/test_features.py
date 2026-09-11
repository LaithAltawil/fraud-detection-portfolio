import duckdb

from fraud_detection.data import money
from fraud_detection.features import CATEGORICAL_FEATURES, NUMERIC_FEATURES, feature_sql


def test_money_parses_currency_string() -> None:
    con = duckdb.connect()
    value = con.execute("SELECT " + money("'$-77.00'")).fetchone()[0]
    assert value == -77.0


def test_feature_sql_is_causal() -> None:
    sql = feature_sql()
    # Every behavioural feature must be bounded to strictly-prior rows.
    assert sql.count("ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING") == 3
    assert "PARTITION BY card_id" in sql


def test_feature_columns_disjoint() -> None:
    assert not set(NUMERIC_FEATURES) & set(CATEGORICAL_FEATURES)
