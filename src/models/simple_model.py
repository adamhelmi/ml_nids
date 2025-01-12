import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import joblib

# Read data with explicit column names
print("Reading data...")
columns = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes',
    'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'hot',
    'num_failed_logins', 'logged_in', 'num_compromised',
    'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
    'num_shells', 'num_access_files', 'num_outbound_cmds',
    'is_host_login', 'is_guest_login', 'count', 'srv_count',
    'serror_rate', 'srv_serror_rate', 'rerror_rate',
    'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
    'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
    'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
    'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate',
    'dst_host_serror_rate', 'dst_host_srv_serror_rate',
    'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'label'
]

data = pd.read_csv("C:/Users/talab/Documents/ml_nids/data/KDDTrain+.txt", names=columns)

# Handle string columns
print("Converting strings...")
categorical_mask = data.dtypes == object
string_columns = data.columns[categorical_mask]

for col in string_columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col].astype(str))

# Split data
print("Splitting data...")
X = data.drop('label', axis=1)
y = data['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
print("Training...")
rf = RandomForestClassifier()
rf.fit(X_train, y_train)

# Test
print("Testing...")
score = rf.score(X_test, y_test)
print(f"Accuracy: {score:.4f}")

# Save both model and column information
print("\nSaving model and preprocessing info...")
joblib.dump(rf, 'C:/Users/talab/Documents/ml_nids/models/nids_model.joblib')
joblib.dump({'features': X.columns.tolist()}, 'C:/Users/talab/Documents/ml_nids/models/features.joblib')