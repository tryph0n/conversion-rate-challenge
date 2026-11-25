import joblib
import pandas as pd

from src.config import METADATA_FILE, MODEL_FILE
from src.features import add_features

ENGINEERED = [
    "age_group",
    "pages_group",
    "new_user_x_pages",
    "age_x_pages",
    "is_germany",
    "is_china",
    "is_seo",
]


def _raw_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "country": "Germany",
                "age": 30,
                "new_user": 1,
                "source": "Seo",
                "total_pages_visited": 5,
            },
            {
                "country": "China",
                "age": 50,
                "new_user": 0,
                "source": "Ads",
                "total_pages_visited": 12,
            },
        ]
    )


def test_add_features_adds_engineered_columns():
    raw = _raw_frame()
    out = add_features(raw)
    assert len(out) == len(raw)
    for column in ENGINEERED:
        assert column in out.columns


def test_committed_model_predicts():
    model = joblib.load(MODEL_FILE)
    proba = model.predict_proba(add_features(_raw_frame()))
    assert proba.shape == (2, 2)
    assert ((proba >= 0) & (proba <= 1)).all()
    assert round(joblib.load(METADATA_FILE)["best_threshold"], 4) == 0.4459
