import pandas as pd

def create_labels(acct_features: pd.DataFrame, alert_df: pd.DataFrame) -> pd.DataFrame:
    """合併警示帳戶標籤"""
    alert_df["label"] = 1
    acct_features = acct_features.merge(alert_df[["acct", "label"]], on="acct", how="left")
    acct_features["label"] = acct_features["label"].fillna(0).astype(int)
    return acct_features