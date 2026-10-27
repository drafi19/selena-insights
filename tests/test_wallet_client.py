import pandas as pd
import pytest

from selena_ml.data.wallet_client import load_token, to_frame


def _rec(**kw):
    base = {
        "id": "r1",
        "recordDate": "2026-08-21T03:00:00Z",
        "createdAt": "2026-08-21T04:00:00Z",
        "amount": {"value": -25000.0, "currencyCode": "IDR"},
        "recordType": "expense",
        "accountName": "Cash",
        "category": {"id": "c1", "name": "Food & Drinks", "group": "food", "color": "#fff"},
        "counterParty": "Warteg",
        "note": "lunch",
        "labels": [{"id": "l1", "name": "Kuliah"}],
        "transfer": None,
    }
    base.update(kw)
    return base


def test_amount_is_read_from_object_and_keeps_sign():
    df = to_frame([_rec()])
    assert df.loc[0, "amount"] == -25000.0


def test_transfer_flag_and_labels_joined():
    df = to_frame([_rec(transfer={"accountId": "acc2"}), _rec(id="r2")])
    assert bool(df["is_transfer"].tolist()[0]) is True
    assert df.loc[0, "labels"] == "Kuliah"


def test_date_is_jakarta_local_naive():
    df = to_frame([_rec()])
    assert df.loc[0, "date"] == pd.Timestamp("2026-08-21 10:00:00")


def test_load_token_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_token(tmp_path / "nope.txt")
