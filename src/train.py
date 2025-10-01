import argparse, os, yaml
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

# === 匯入自訂模組 ===
from data.load_data import load_transaction_data, load_alert_data
from data.labeling import create_labels
from features.build_features import build_account_features
from features.monitor_features import monitor_account_features
from preprocessing.pipeline import build_preprocessing_pipeline
from utils.file_utils import check_input_files, get_output_dir
from utils.class_weights import compute_scale_pos_weight
from utils.io_utils import save_model, load_model
from optimization.pso import PSO
from optimization.fitness import fitness_function
from models.train import train_xgb
from evaluation.metrics import evaluate_model, find_best_threshold
from evaluation.diagnose_feature import diagnose_features
from predict.predict import run_prediction

# =========================
# 1. 載入 config
# =========================
parser = argparse.ArgumentParser()
parser.add_argument("--config", default="configs/config.yaml", help="Path to config file")
args = parser.parse_args()
project_root = os.path.dirname(os.path.dirname(__file__))
config_path = os.path.join(project_root, args.config)

# 讀取 config.yaml
with open(config_path, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

# Input檔案路徑
acct_transaction_csv = config["input"]["acct_transaction_csv"]
acct_alert_csv = config["input"]["acct_alert_csv"]
acct_predict_csv = config["input"]["acct_predict_csv"]

# 檢查Input檔案
check_input_files([acct_transaction_csv, acct_alert_csv, acct_predict_csv])

# 設定Output檔案路徑
base_output_dir = get_output_dir("outputs")

# =========================
# 2. 載入資料
# =========================
print("[INFO] 載入資料...")
txn_df = load_transaction_data(acct_transaction_csv)
alert_df = load_alert_data(acct_alert_csv)

# =========================
# 3. 特徵工程 + 標籤
# =========================
print("[INFO] 建立帳戶層級特徵...")
acct_features = build_account_features(txn_df)
print("[INFO] 建立標籤...")
acct_features = create_labels(acct_features, alert_df)

X = acct_features.drop(columns=["acct", "label"])
y = acct_features["label"]

feature_plots_dir = os.path.join(base_output_dir, "feature_plots")
monitor_account_features(acct_features, output_dir=feature_plots_dir)

# =========================
# 4. 切分資料
# =========================
print("[INFO] 切分資料...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# =========================
# 5. 前處理 pipeline
# =========================
print("[INFO] 建立前處理 pipeline...")
pipeline = build_preprocessing_pipeline(X.columns.tolist())
X_train_processed = pipeline.fit_transform(X_train)
X_test_processed = pipeline.transform(X_test)

# =========================
# 6. 樣本不平衡處理
#   設為 (負樣本數 / 正樣本數)
# =========================
scale_pos_weight = compute_scale_pos_weight(y_train)
print(f"[INFO] scale_pos_weight = {scale_pos_weight:.2f}")

# =========================
# 7. 粒子群最佳化 (PSO)
# =========================
# 定義搜尋範圍
bounds = np.array([
    [3, 10],       # max_depth
    [0.01, 0.1],   # learning_rate
    [0.6, 1.0],    # subsample
    [0.6, 1.0],    # colsample_bytree
    [0.5, 3.0],    # reg_lambda
    [0.0, 1.0]     # reg_alpha
])

print("[INFO] 開始粒子群最佳化 (PSO)...")
pso_config = config["pso"]
bounds = np.array(pso_config["bounds"])
fitness = lambda params: fitness_function(params, X_train_processed, y_train, scale_pos_weight)
pso = PSO(
    fitness_func=fitness,
    dim=pso_config["dim"],
    bounds=bounds,
    num_particles=pso_config["num_particles"],
    max_iter=pso_config["max_iter"]
)
best_params, best_score = pso.optimize()
print(f"[INFO] PSO 最佳參數: {best_params}")
print(f"[INFO] PSO 最佳 AUC: {best_score:.4f}")

# =========================
# 8. 訓練最終模型
# =========================
params = {
    "scale_pos_weight": scale_pos_weight,
    "max_depth": int(best_params[0]),
    "learning_rate": best_params[1],
    "subsample": best_params[2],
    "colsample_bytree": best_params[3],
    "reg_lambda": best_params[4],
    "reg_alpha": best_params[5],
    "n_estimators": config["model"]["n_estimators"],
    "early_stopping_rounds": config["model"]["early_stopping_rounds"],
    "eval_metric": config["model"]["eval_metric"],
    "objective": config["model"]["objective"],
    "device": config["model"]["device"],
    "random_state": config["model"]["random_state"],
    "n_jobs": config["model"]["n_jobs"],
}
print(f"[INFO] 最終模型參數: {params}")
print("[INFO] 訓練最終模型...")
model_plots_dir = os.path.join(base_output_dir, "model_plots")
model = train_xgb(X_train_processed, y_train, X_test_processed, y_test, params, output_dir=model_plots_dir)

# =========================
# 9. 評估模型
# =========================
print("[INFO] 進行特徵診斷...")
summary_csv = os.path.join(base_output_dir, "feature_diagnosis_summary.csv")
diagnose_features(X_train, X_test, y_train, y_test, model, output_dir=feature_plots_dir, summary_csv=summary_csv)

report, auc, y_pred, y_proba = evaluate_model(model, X_test_processed, y_test)
print(report)
print(f"AUC: {auc:.4f}")

best_threshold, best_f1 = find_best_threshold(y_test, y_proba)
print(f"[INFO] 最佳 Threshold: {best_threshold:.2f}, F1={best_f1:.4f}")

# =========================
# 10. 儲存模型
# =========================
saved_model_path = os.path.join(base_output_dir, "xgb_acctlevel_model.joblib")
save_model({"pipeline": pipeline, "model": model, "best_threshold": best_threshold}, saved_model_path)
print(f"[INFO] 模型已儲存到 {saved_model_path}")

# =========================
# 11. 載入模型並預測
# =========================
saved = load_model(saved_model_path)
pipeline, model, best_threshold = saved["pipeline"], saved["model"], saved["best_threshold"]

acct_predict_result_csv = os.path.join(base_output_dir, "acct_predict_result.csv")
result_df = run_prediction(model, pipeline, acct_features, acct_predict_csv, acct_predict_result_csv)
