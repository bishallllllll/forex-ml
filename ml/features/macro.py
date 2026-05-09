import pandas as pd


def compute_surprise(actual: pd.Series, forecast: pd.Series) -> pd.Series:
    return actual - forecast


def engineer_macro_features(calendar_df: pd.DataFrame) -> pd.DataFrame:
    df = calendar_df.copy()
    if "Surprise" not in df.columns and "Actual" in df.columns and "Forecast" in df.columns:
        df["Surprise"] = compute_surprise(df["Actual"], df["Forecast"])
    df["Surprise_Abs"] = df["Surprise"].abs()
    df["Surprise_Direction"] = df["Surprise"].apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
    return df
