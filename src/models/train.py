import matplotlib.pyplot as plt
from xgboost import XGBClassifier, plot_importance, plot_tree
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import os
from datetime import datetime


def train_xgb(X_train, y_train, X_val, y_val, params, output_dir):
    """訓練 XGBoost 模型並輸出監控圖表 (自動加上時間戳記)"""
    os.makedirs(output_dir, exist_ok=True)
    # 建立時間戳記 (例如 20250929_0005)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    """訓練 XGBoost 模型"""
    model = XGBClassifier(**params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=True)
    # 取得學習曲線 - 顯示 train-auc 與 validation-auc，方便判斷是否過擬合。
    results = model.evals_result()
    # 自動偵測有哪些 metric
    metrics = list(results['validation_0'].keys())


    # 為每個 metric 畫圖
    for metric in metrics:
        plt.figure(figsize=(8, 6))

        # 遍歷所有 eval_set (validation_0, validation_1, ...)
        for eval_name, eval_result in results.items():
            epochs = len(eval_result[metric])
            x_axis = range(0, epochs)
            plt.plot(x_axis, eval_result[metric], label=f"{eval_name} {metric}")

        plt.xlabel('Boosting Round')
        plt.ylabel(metric.upper())
        plt.title(f"XGBoost Learning Curve ({metric})")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, f"learning_curve_{metric}_{timestamp}.png"))
        plt.close()

    # 特徵重要性 - 依據 gain 排序，顯示哪些特徵對模型最重要。
    plt.figure(figsize=(8, 6))
    plot_importance(model, importance_type="gain")
    plt.title("Feature Importance (Gain)")
    plt.savefig(os.path.join(output_dir, f"feature_importance_{timestamp}.png"))
    plt.close()


    # 視覺化決策樹 (第一棵樹)
    plt.figure(figsize=(20, 10))
    plot_tree(model, num_trees=0, rankdir="LR")
    plt.savefig(os.path.join(output_dir, f"tree_structure_{timestamp}.png"))
    plt.close()

    return model