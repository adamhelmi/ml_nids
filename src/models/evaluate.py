import os
import sys
import joblib
from sklearn.metrics import accuracy_score

# Add the project root directory to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)

from src.preprocessing.data_processor import NetworkDataProcessor

# Initialize the data processor
processor = NetworkDataProcessor()

# Load and preprocess the test data
test_df = processor.load_data("data/KDDTest+.txt")
test_processed, _ = processor.preprocess_data(test_df)

# Separate features and labels
X_test = test_processed.drop("label", axis=1)
y_test = test_processed["label"]

# Load the trained model
clf = joblib.load("models/nids_model.joblib")

# Predict on the test data
y_pred = clf.predict(X_test)

# Calculate and display the accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy on Test Data: {accuracy * 100:.2f}%")