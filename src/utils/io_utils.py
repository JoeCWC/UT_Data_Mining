import joblib

def save_model(obj, path: str):
    """儲存模型與 pipeline"""
    joblib.dump(obj, path)

def load_model(path: str):
    """載入模型與 pipeline"""
    return joblib.load(path)