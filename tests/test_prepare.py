import pandas as pd
import pytest

from selena_ml.data.prepare import prepare


def _df(**over):
    rows = [
        dict(date="2026-08-01", amount=-25000.0, record_type="expense",
             category="Food & Drinks", labels="", is_transfer=False),
        dict(date="2026-08-01", amount=-500000.0, record_type="expense",
             category="Transfer", labels="", is_transfer=True),
        dict(date="2026-08-02", amount=-50000.0, record_type="expense",
             category="Lending, renting", labels="", is_transfer=False),
        dict(date="2026-08-02", amount=-70000.0, record_type="expense",
             category="Loan, interests", labels="", is_transfer=False),
        dict(date="2026-08-02", amount=-30000.0, record_type="expense",
             category="Groceries", labels="Titip", is_transfer=False),
        dict(date="2026-08-02", amount=0.0, record_type="expense",
             category="Restaurants & fast food", labels="", is_transfer=False),
        dict(date="2026-08-03", amount=2000000.0, record_type="income",
             category="Salary", labels="", is_transfer=False),
    ]
    df = pd.DataFrame(rows)
    for k, v in over.items():
        df[k] = v
    return df.assign(date=pd.to_datetime(df["date"]))


def test_only_plain_expenses_survive():
    spending, excluded = prepare(_df())
    assert len(spending) == 1
    assert spending.loc[0, "category"] == "Food & Drinks"
    assert spending.loc[0, "amount"] == 25000.0


def test_reasons_are_recorded():
    _, excluded = prepare(_df())
    assert set(excluded["reason"]) == {
        "transfer",
        "pass_through_category",
        "pass_through_label",
        "not_expense",
        "zero_amount",
    }


def test_amounts_are_positive():
    spending, _ = prepare(_df())
    assert (spending["amount"] > 0).all()


def test_empty_spending_raises():
    df = _df().assign(is_transfer=True)
    with pytest.raises(ValueError, match="no spending records"):
        prepare(df)
