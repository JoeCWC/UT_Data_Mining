import pandas as pd

def load_transaction_data(path: str) -> pd.DataFrame:
    """讀取交易資料"""
    return pd.read_csv(path)

def load_alert_data(path: str) -> pd.DataFrame:
    """讀取警示帳戶資料"""
    return pd.read_csv(path)