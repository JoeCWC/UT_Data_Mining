from sklearn.model_selection import StratifiedKFold, cross_val_score
from xgboost import XGBClassifier

def fitness_function(params, X_train_processed, y_train, scale_pos_weight):
    """以 AUC 作為適應度函數"""
    max_depth = int(params[0])
    learning_rate = params[1]
    subsample = params[2]
    colsample_bytree = params[3]
    reg_lambda = params[4]
    reg_alpha = params[5]

    model = XGBClassifier(
        scale_pos_weight=scale_pos_weight,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        reg_lambda=reg_lambda,
        reg_alpha=reg_alpha,
        n_estimators=2000,
        early_stopping_rounds=None,
        eval_metric="auc",
        objective="binary:logistic",
        device="cuda",
        random_state=42,
        n_jobs=-1,
        verbosity=0
    )

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    scores = cross_val_score(model, X_train_processed, y_train, cv=cv, scoring="roc_auc")
    return scores.mean()
    #return auc_score