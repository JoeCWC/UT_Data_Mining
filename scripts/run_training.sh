#!/bin/bash
# ============================================
# run_training.sh
# 一鍵執行 ML pipeline：資料處理 → 特徵工程 → 調參 → 訓練 → 評估 → 儲存 → 預測
# ============================================

# 1. 啟動虛擬環境 (依照你的環境修改)
# 如果用 conda
#source ~/miniconda3/etc/profile.d/conda.sh
#conda activate ai_cup_env

# 如果用 venv
# source venv/bin/activate

# 2. 切換到專案根目錄
cd "$(dirname "$0")/.." || exit 1
#pwd

# 3. 建立 logs 目錄
mkdir -p logs

# 4. 執行訓練，並將輸出寫入 log 檔
timestamp=$(date +"%Y%m%d_%H%M%S")
log_file="logs/train_$timestamp.log"

echo "[INFO] 開始訓練流程，log 輸出到 $log_file"
python -u src/train.py --config configs/config.yaml 2>&1 | tee "$log_file"

# 5. 執行結果提示
if [ $? -eq 0 ]; then
    echo "[INFO] Pipeline 執行完成 ✅"
else
    echo "[ERROR] Pipeline 執行失敗 ❌，請檢查 $log_file"
fi