import seaborn as sns
import matplotlib.pyplot as plt
import os 
import pandas as pd

def monitor_account_features(acct_features: pd.DataFrame, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    # 1️⃣ 金額特徵 pairplot
    amt_cols = ["txn_amt_mean", "txn_amt_max", "txn_amt_std"]
    sns.pairplot(acct_features[amt_cols])
    plt.suptitle("Pairplot of Amount Features", fontsize=14)
    plt.savefig(os.path.join(output_dir, "pairplot_amount_features.png"))
    plt.close()

    # 2️⃣ 時間特徵散佈圖
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=acct_features, x="txn_per_day", y="night_ratio")
    plt.title("Scatterplot of Time Features")
    plt.xlabel("Transactions per Day")
    plt.ylabel("Night Transaction Ratio")
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "scatter_time_features.png"))
    plt.close()

    # 3️⃣ 網路特徵 jointplot
    sns.jointplot(data=acct_features, x="out_degree", y="in_degree", kind="scatter", height=6)
    plt.suptitle("Jointplot of Network Features", fontsize=14)
    plt.savefig(os.path.join(output_dir, "jointplot_network_features.png"))
    plt.close()