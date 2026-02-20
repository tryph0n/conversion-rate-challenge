"""Feature engineering for conversion rate prediction."""

import pandas as pd


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add engineered features to the dataframe.

    Args:
        df: Raw dataframe with columns:
            country, age, new_user, source, total_pages_visited.

    Returns:
        DataFrame with original + engineered features.
    """
    df = df.copy()

    df["age_group"] = pd.cut(
        df["age"],
        bins=[0, 25, 35, 45, 100],
        labels=["young", "adult", "middle", "senior"],
        include_lowest=True,
    )

    df["pages_group"] = pd.cut(
        df["total_pages_visited"],
        bins=[0, 2, 5, 10, float("inf")],
        labels=["low", "medium", "high", "very_high"],
        include_lowest=True,
    )

    df["new_user_x_pages"] = df["new_user"] * df["total_pages_visited"]
    df["age_x_pages"] = df["age"] * df["total_pages_visited"]

    df["is_germany"] = (df["country"] == "Germany").astype(int)
    df["is_china"] = (df["country"] == "China").astype(int)
    df["is_seo"] = (df["source"] == "Seo").astype(int)

    return df


CATEGORICAL_FEATURES = [
    "country",
    "source",
    "age_group",
    "pages_group",
]
NUMERICAL_FEATURES = [
    "age",
    "total_pages_visited",
    "new_user_x_pages",
    "age_x_pages",
]
BINARY_FEATURES = [
    "new_user",
    "is_germany",
    "is_china",
    "is_seo",
]
