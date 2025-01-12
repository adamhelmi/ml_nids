import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def detect_intrusion(network_data):
    """
    Use the trained model to detect intrusions in new network traffic data
    """
    try:
        # Load model and feature info
        model = joblib.load('C:/Users/talab/Documents/ml_nids/models/nids_model.joblib')
        features = joblib.load('C:/Users/talab/Documents/ml_nids/models/features.joblib')['features']

        # Process categorical features before reordering columns
        categorical_features = ['protocol_type', 'service', 'flag']
        for feature in categorical_features:
            le = LabelEncoder()
            network_data[feature] = le.fit_transform(network_data[feature])

        # Convert all remaining columns to float
        for col in network_data.columns:
            if col not in categorical_features:
                network_data[col] = pd.to_numeric(network_data[col], errors='coerce')

        # Ensure same column order as training
        network_data = network_data[features]
        
        # Make prediction
        predictions = model.predict(network_data)
        probabilities = model.predict_proba(network_data)
        
        return predictions, probabilities
        
    except Exception as e:
        print(f"Error during detection: {str(e)}")
        return None, None

if __name__ == "__main__":
    # Define columns
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
    
    print("Loading test data...")
    test_data = pd.read_csv("C:/Users/talab/Documents/ml_nids/data/KDDTrain+.txt", names=columns).head(5)
    X_test = test_data.drop('label', axis=1)
    y_test = test_data['label']
    
    predictions, probabilities = detect_intrusion(X_test)
    
    if predictions is not None:
        print("\nPredictions for first 5 network connections:")
        for i, (pred, prob, true) in enumerate(zip(predictions, probabilities, y_test)):
            print(f"Connection {i+1}:")
            print(f"- Predicted class: {pred}")
            print(f"- Actual class: {true}")
            print(f"- Confidence: {prob.max():.2%}")