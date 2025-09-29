from sklearn.metrics import classification_report, roc_auc_score, f1_score, precision_score, recall_score
import numpy as np

def evaluate_model(model, X_test, y_test):
    """輸出 classification report 與 AUC"""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    report = classification_report(y_test, y_pred, digits=4)
    auc = roc_auc_score(y_test, y_proba)
    return report, auc, y_pred, y_proba

def find_best_threshold(y_test, y_proba):
    """搜尋最佳 threshold (以 F1 為準)"""
    thresholds = np.linspace(0.1, 0.9, 17)
    best_threshold, best_f1 = 0.5, 0
    for t in thresholds:
        y_pred_thresh = (y_proba >= t).astype(int)
        f1 = f1_score(y_test, y_pred_thresh)
        if f1 > best_f1:
            best_f1, best_threshold = f1, t
    return best_threshold, best_f1