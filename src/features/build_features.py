import pandas as pd

# =========================
# 「帳戶層級特徵表」
# - 時間特徵（交易頻率、夜間比例）
# - 金額特徵（平均、最大、標準差）
# - 網路特徵（出度、入度、對手數量）

# 透過 時間線拆解 + 金額分析 + 交易對手網路，去找出「未來可能被判定為警示」的帳戶。這個需求其實就是 反洗錢 (AML) / 可疑交易偵測 的典型任務。

# 時間線拆解 (Temporal Features)
# - 交易頻率：計算帳戶在不同時間窗內的交易次數（近 1 天、7 天、30 天）。
# - 交易間隔：平均交易間隔時間、最短/最長間隔。
# - 時間分布：夜間交易比例（22:00–06:00）、尖峰時段交易比例。
# - 異常模式：短時間內多筆大額交易、連續轉帳到不同帳戶。
# 👉 這些特徵能捕捉「洗錢常見的時間異常行為」

# 金額分析 (Amount Features)
# - 統計特徵：平均金額、最大金額、最小金額、標準差。
# - 大額比例：大於某門檻（例如 50 萬）的交易比例。
# - 金額分布：小額高頻 vs. 大額低頻。
# - 幣別異常：是否頻繁使用外幣（USD、JPY…）轉帳。
# 👉 這些特徵能捕捉「小額切割 (structuring)」或「大額異常」的模式

# 交易對手網路 (Graph Features)
# - 出度 (out-degree)：該帳戶轉出給多少不同帳戶。
# - 入度 (in-degree)：該帳戶收過多少不同帳戶。
# - 雙向交易比例：是否存在「互相轉帳」的關係。
# - 與警示帳戶的關聯：是否直接或間接與已知警示帳戶有交易。
# - 社群特徵：透過圖演算法（PageRank、Connected Components）找出可疑集團。
# 👉 這些特徵能捕捉「人頭帳戶集團」或「資金洗白網路」

# 不合理交易模式的判斷邏輯
# 除了模型預測，你也可以設計 規則檢測 (rule-based features)，例如：
# - 單日交易次數 > 50 且金額總和 > 100 萬。
# - 夜間交易比例 > 80%。
# - 出度 > 20 且交易對手多為新帳戶。
# - 與已知警示帳戶有直接交易。
# 這些規則可以和 XGBoost 模型結合，形成 Hybrid System，提升可解釋性。
# =========================

def build_account_features(txn_df: pd.DataFrame) -> pd.DataFrame:
    """建立帳戶層級特徵 (金額、時間、網路)"""
    print("[INFO] 建立帳戶層級特徵...")

    # (a) 金額特徵
    amt_stats = txn_df.groupby("from_acct")["txn_amt"].agg(
        txn_amt_mean="mean",
        txn_amt_max="max",
        txn_amt_std="std",
        txn_count="count"
    ).reset_index().rename(columns={"from_acct": "acct"})

    # (b) 時間特徵
    txn_df["txn_hour"] = pd.to_datetime(txn_df["txn_time"], format="%H:%M:%S", errors="coerce").dt.hour
    txn_df["is_night"] = txn_df["txn_hour"].apply(lambda h: 1 if pd.notnull(h) and (h < 6 or h >= 22) else 0)

    time_stats = txn_df.groupby("from_acct").agg(
        night_ratio=("is_night", "mean"),
        txn_per_day=("txn_date", lambda x: len(x) / (x.max() - x.min() + 1))
    ).reset_index().rename(columns={"from_acct": "acct"})

    # (c) 網路特徵
    out_degree = txn_df.groupby("from_acct")["to_acct"].nunique().reset_index()
    out_degree = out_degree.rename(columns={"from_acct": "acct", "to_acct": "out_degree"})

    in_degree = txn_df.groupby("to_acct")["from_acct"].nunique().reset_index()
    in_degree = in_degree.rename(columns={"to_acct": "acct", "from_acct": "in_degree"})

    # (d) 合併所有特徵
    acct_features = amt_stats.merge(time_stats, on="acct", how="left")
    acct_features = acct_features.merge(out_degree, on="acct", how="left")
    acct_features = acct_features.merge(in_degree, on="acct", how="left")

    # 缺失值補 0
    acct_features = acct_features.fillna(0)
    return acct_features