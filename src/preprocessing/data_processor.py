import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Tuple, Dict, Any

class NetworkDataProcessor:
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
    
    def load_data(self, filepath: str) -> pd.DataFrame:
        """
        Load the NSL-KDD dataset.
        """
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
        
        df = pd.read_csv(filepath, names=columns)
        print(f"Loaded dataset with shape: {df.shape}")
        return df
    
    def preprocess_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Preprocess the data:
        1. Handle categorical features
        2. Handle missing values
        3. Scale numerical features
        4. Encode labels
        """
        df_processed = df.copy()
        
        categorical_columns = df_processed.select_dtypes(include=['object']).columns
        numerical_columns = df_processed.select_dtypes(include=['int64', 'float64']).columns
        numerical_columns = numerical_columns.drop('label' if 'label' in numerical_columns else [])
        
        # Handle categorical features
        for column in categorical_columns:
            if column != 'label':
                if column not in self.label_encoders:
                    self.label_encoders[column] = LabelEncoder()
                df_processed[column] = self.label_encoders[column].fit_transform(df_processed[column])
        
        # Handle missing values
        df_processed = df_processed.fillna(df_processed.mean())
        
        # Scale numerical features
        df_processed[numerical_columns] = self.scaler.fit_transform(df_processed[numerical_columns])
        
        # Encode labels
        if 'label' in df_processed.columns:
            if 'label' not in self.label_encoders:
                self.label_encoders['label'] = LabelEncoder()
            df_processed['label'] = self.label_encoders['label'].fit_transform(df_processed['label'])
        
        preprocessing_info = {
            'categorical_columns': list(categorical_columns),
            'numerical_columns': list(numerical_columns),
            'label_encoders': self.label_encoders,
            'scaler': self.scaler
        }
        
        return df_processed, preprocessing_info

    def analyze_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Perform basic data analysis.
        """
        analysis = {
            'shape': df.shape,
            'columns': list(df.columns),
            'dtypes': df.dtypes.to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'label_distribution': df['label'].value_counts().to_dict() if 'label' in df.columns else None,
        }
        return analysis
    
if __name__ == "__main__":
    print("NetworkDataProcessor is ready to use!")
    processor = NetworkDataProcessor()
    print("Initialized NetworkDataProcessor")
    
    # Optional: Add some basic tests
    try:
        # Attempt to load a sample dataset (you'll need to update this path)
        sample_data = processor.load_data('C:/Users/talab/Documents/ml_nids/data/KDDTrain+.txt')
        print(f"Successfully loaded sample data with shape: {sample_data.shape}")
        
        # Attempt to preprocess the data
        processed_data, info = processor.preprocess_data(sample_data)
        print(f"Successfully preprocessed data. New shape: {processed_data.shape}")
        
        # Analyze the processed data
        analysis = processor.analyze_data(processed_data)
        print("Data analysis completed. Summary:")
        print(f"- Number of features: {len(analysis['columns']) - 1}")  # Subtract 1 for the label column
        print(f"- Number of samples: {analysis['shape'][0]}")
        print(f"- Label distribution: {analysis['label_distribution']}")
        
    except Exception as e:
        print(f"An error occurred during testing: {str(e)}")
        print("Note: This error might be due to missing sample data. Please update the file path.")
    
print("Test complete.")
 