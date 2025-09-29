import pandas as pd

def run_prediction(model, pipeline, acct_features, predict_csv, output_csv):
    """執行預測流程"""
    predict_df = pd.read_csv(predict_csv)
    predict_df = predict_df.merge(acct_features, on="acct", how="left").fillna(0)
    X_pred = predict_df.drop(columns=["acct", "label"], errors="ignore")
    X_pred_processed = pipeline.transform(X_pred)
    y_pred = model.predict(X_pred_processed)
    result_df = pd.DataFrame({"acct": predict_df["acct"], "label": y_pred})
    result_df.to_csv(output_csv, index=False)
    return result_df