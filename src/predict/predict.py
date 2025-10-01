import pandas as pd
import numpy as np

# def run_prediction(model, pipeline, acct_features, predict_csv, output_csv):
#     """執行預測流程"""
#     predict_df = pd.read_csv(predict_csv)
#     predict_df = predict_df.merge(acct_features, on="acct", how="left").fillna(0)
#     X_pred = predict_df.drop(columns=["acct", "label"], errors="ignore")
#     X_pred_processed = pipeline.transform(X_pred)
#     y_pred = model.predict(X_pred_processed)
#     result_df = pd.DataFrame({"acct": predict_df["acct"], "label": y_pred})
#     result_df.to_csv(output_csv, index=False)
#     return result_df

def run_prediction(model, pipeline, acct_features, predict_csv, output_csv):
    """執行預測流程（含防呆）"""
    predict_df = pd.read_csv(predict_csv)
    # print("[DEBUG] 預測資料筆數：", len(predict_df))
    # print("[DEBUG] 欄位：", predict_df.columns.tolist())

    # 確保 acct_features 沒有重複帳戶
    assert acct_features["acct"].is_unique, "[ERROR] acct_features 中 acct 欄位有重複值"

    predict_df = predict_df.merge(acct_features, on="acct", how="left").fillna(0)
    X_pred = predict_df.drop(columns=["acct", "label"], errors="ignore")

    # print("[DEBUG] 預測欄位：", X_pred.columns.tolist())
    assert X_pred.shape[0] > 0, "[ERROR] 預測資料為空"

    try:
        X_pred_processed = pipeline.transform(X_pred)
    except Exception as e:
        print("[ERROR] pipeline.transform() 失敗：", e)
        return

    assert not np.isnan(X_pred_processed).any(), "[ERROR] 預測資料包含 NaN"

    try:
        y_pred = model.predict(X_pred_processed)
    except Exception as e:
        print("[ERROR] model.predict() 失敗：", e)
        return

    result_df = pd.DataFrame({"acct": predict_df["acct"], "label": y_pred})
    result_df.to_csv(output_csv, index=False)
    print(f"[INFO] 預測完成，結果輸出到 {output_csv}")
    return result_df