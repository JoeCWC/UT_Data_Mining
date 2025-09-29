from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

def build_preprocessing_pipeline(numeric_features):
    """建立前處理 pipeline"""
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ]), numeric_features)
        ]
    )
    return Pipeline(steps=[("preprocessor", preprocessor)])