import os
import seaborn as sns
import matplotlib.pyplot as plt
import shap
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

'''
分布檢查（偏態係數）

- 偏態係數是一個統計量，用來衡量分布的「對稱性」。
- skewness ≈ 0 → 分布大致對稱（像常態分布）。
- skewness > 0 → 右偏（右邊尾巴長，常見於金額、收入這類特徵）。
- skewness < 0 → 左偏（左邊尾巴長）。
- 偏態係數絕對值 > 2 通常表示分布高度偏斜，建議進行轉換（如 log transform）或標準化。

為什麼要檢查偏態
- 避免模型受極端值影響
- 偏態太大代表有很多極端值（outliers），可能讓模型過度關注少數異常樣本。
- 提升模型收斂速度
- 高度偏態的特徵，會讓梯度下降或樹模型的切分不穩定，導致訓練效率下降。
- 改善特徵可解釋性
- 經過轉換（例如 log transform）後，分布更接近常態，特徵與標籤的關係會更清晰。

- 如果 |skewness| > 2 → 建議做轉換（log、Box-Cox、標準化），讓分布更平滑。
- 如果 |skewness| 在 1 到 2 之間 → 中度偏態，可視情況考慮轉換。
- 如果 |skewness| < 1 → 偏態不明顯，通常不需要特別處理。

- 交易金額 (txn_amt)：通常右偏很嚴重（大部分人小額交易，少數人超大額），skewness 可能 > 5。 → 適合做 log(1+x) 轉換。
'''

'''
標籤分組視覺化（中位數差異）

1. 檢查特徵與標籤的關聯性
- 如果一個特徵在正樣本（label=1）和負樣本（label=0）之間的分布幾乎一樣，代表它對分類幫助不大。
- 反之，如果兩組的中位數差異明顯，這個特徵可能對區分類別有潛力。

2. 避免被極端值誤導
- 平均數容易被 outlier 拉高或拉低。
- 中位數更穩健，可以更真實地反映「大部分樣本」的差異。

3. 快速視覺化可解釋性
- 用 boxplot 或 violin plot 可以直觀看到「不同標籤下的分布差異」。
- 這有助於判斷特徵是否值得保留，或是否需要轉換。
'''

'''
SHAP 分析（排名）

1. 避免只看統計差異的偏誤
- 前面我們做了「偏態檢查」和「中位數差異」，這些是統計層面的檢查。
- 但有些特徵雖然在統計上差異不大，卻可能在模型裡被高度利用（例如交互作用特徵）。
- SHAP 可以直接告訴你模型實際上有沒有「依賴」這個特徵。

2. 輔助特徵篩選與模型精簡
- 如果某個特徵 SHAP 排名很低，且統計檢查也顯示沒什麼差異，就可以考慮移除，讓模型更簡潔。
- 如果某個特徵 SHAP 排名很高，即使它的分布差異不大，也值得保留。
'''

def diagnose_features(X_train, X_test, y_train, y_test, model, output_dir: str, summary_csv: str):
    """
    對所有特徵進行診斷：
    1. 輸出每個特徵的圖表與文字報告
    2. 彙整成一份 CSV 總表 (偏態係數、中位數差異、SHAP 排名、KS 檢定、建議)
    """

    os.makedirs(output_dir, exist_ok=True)
    results = []

    # === SHAP 分析 (一次算完所有特徵) ===
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train)
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    shap_ranking = mean_abs_shap.argsort()[::-1]
    feature_to_rank = {X_train.columns[i]: rank+1 for rank, i in enumerate(shap_ranking)}

    for feature_name in X_train.columns:
        # 1️⃣ 分布檢查（偏態係數）
        skewness = X_train[feature_name].skew()
        plt.figure(figsize=(8, 6))
        sns.histplot(X_train[feature_name], bins=50, kde=True, color="steelblue")
        plt.title(f"Distribution of {feature_name} (Train)")
        plt.xlabel(feature_name)
        plt.ylabel("Count")
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, f"{feature_name}_distribution.png"))
        plt.close()

        # 2️⃣ 標籤分組視覺化（中位數差異）
        median_pos = X_train.loc[y_train==1, feature_name].median()
        median_neg = X_train.loc[y_train==0, feature_name].median()
        median_diff_ratio = abs(median_pos - median_neg) / (abs(median_neg) + 1e-6)

        plt.figure(figsize=(8, 6))
        sns.boxplot(x=y_train, y=X_train[feature_name])
        plt.title(f"{feature_name} vs Label (Train)")
        plt.xlabel("Label")
        plt.ylabel(feature_name)
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, f"{feature_name}_boxplot.png"))
        plt.close()

        # 3️⃣ SHAP 排名
        feature_rank = feature_to_rank[feature_name]

        # 4️⃣ Train vs Test 分布比較（KS 檢定）
        ks_stat, ks_pvalue = ks_2samp(X_train[feature_name], X_test[feature_name])
        plt.figure(figsize=(8, 6))
        sns.kdeplot(X_train[feature_name], label="Train", fill=True)
        sns.kdeplot(X_test[feature_name], label="Test", fill=True)
        plt.title(f"{feature_name} Distribution: Train vs Test")
        plt.xlabel(feature_name)
        plt.ylabel("Density")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, f"{feature_name}_overfit_check.png"))
        plt.close()

        # 5️⃣ 自動報告判斷邏輯
        if abs(skewness) > 2:
            suggestion = "需轉換（建議 log transform 或標準化）; \n原因: skewness > 2"
        elif median_diff_ratio < 0.05 and feature_rank > 10:
            suggestion = "建議移除（分布差異小且 SHAP 排名低）; \n原因: median_diff_ratio < 0.05 and feature_rank > 10"
        elif feature_rank <= 5 and ks_pvalue >= 0.01:
            suggestion = "建議保留（高 SHAP 排名且分布穩定）; \n原因: feature_rank <= 5 and ks_pvalue >= 0.01"
        elif ks_pvalue < 0.01:
            suggestion = "⚠️ 可能過擬合（Train/Test 分布差異大）; \n原因: ks_pvalue < 0.01"
        else:
            suggestion = "中性（可保留，但需觀察）"

        # 6️⃣ 個別文字報告
        report_lines = [
            "="*60,
            f"🔎 特徵診斷報告：{feature_name}",
            f"- 偏態係數 (skewness): {skewness:.2f}",
            f"- 標籤中位數差異比例: {median_diff_ratio:.2%}",
            f"- SHAP 排名: {feature_rank}",
            f"- KS 檢定 p-value (Train vs Test): {ks_pvalue:.4f}",
            f"👉 建議：{suggestion}",
            "="*60
        ]
        for line in report_lines:
            print(line)

        report_path = os.path.join(output_dir, f"{feature_name}_diagnosis.txt")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(report_lines))

        # 7️⃣ 加入總表
        results.append({
            "feature": feature_name,
            "skewness": round(skewness, 2),
            "median_diff_ratio": round(median_diff_ratio, 4),
            "shap_rank": feature_rank,
            "ks_pvalue": round(ks_pvalue, 4),
            "suggestion": suggestion
        })

    # === 輸出 CSV 總表 ===
    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values(by="shap_rank")
    os.makedirs(os.path.dirname(summary_csv), exist_ok=True)
    df_results.to_csv(summary_csv, index=False, encoding="utf-8-sig")

    print(f"[INFO] 特徵診斷總表已輸出到 {summary_csv}")
    return df_results