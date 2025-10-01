import os, sys
from datetime import datetime

def check_input_files(paths):
    print("[INFO] 檢查輸入檔案是否存在...")
    for path in paths:
        if not os.path.exists(path):
            print(f"[ERROR] 找不到必要的輸入檔案: {path}")
            sys.exit(1)

def ensure_output_dirs(paths):
    print("[INFO] 檢查/建立輸出目錄...")
    for path in paths:
        out_dir = os.path.dirname(path)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)
            print(f"[INFO] 建立目錄: {out_dir}")

def get_output_dir(base_dir="outputs"):
    print("[INFO] 檢查/建立輸出目錄...")
    # 取得今天日期
    today = datetime.now().strftime("%Y-%m-%d")
    version = 1

    while True:
        candidate = os.path.join(base_dir, f"{today}_{version}")
        if not os.path.exists(candidate):
            os.makedirs(candidate, exist_ok=True)
            return candidate
        version += 1
