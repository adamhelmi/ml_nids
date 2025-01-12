# train.py (no "../..")

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from src.preprocessing.data_processor import NetworkDataProcessor

def main():
    print("Test complete.")

    # 1. Initialize data processor
    processor = NetworkDataProcessor()

    # 2. Load training data from ml_nids/data/KDDTrain+.txt
    df = processor.load_data("data/KDDTrain+.txt")  
    df_processed, info = processor.preprocess_data(df)

    # 3. Save preprocessing info (label encoders, scaler, etc.)
    joblib.dump(info, "models/preprocessing_info.joblib")

    # 4. Separate features/labels
    y = df_processed["label"]
    X = df_processed.drop("label", axis=1)

    # 5. Train a RandomForest model
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X, y)

    # 6. Save the model and feature list to ml_nids/models/
    joblib.dump(clf, "models/nids_model.joblib")
    joblib.dump({"features": X.columns.tolist()}, "models/features.joblib")

    print("Model trained and saved to 'models/' folder.")

if __name__ == "__main__":
    main()
