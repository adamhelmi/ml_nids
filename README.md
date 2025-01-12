ML NIDS (Network Intrusion Detection System)
This project implements a Machine Learning–based Intrusion Detection System using the NSL-KDD dataset. It demonstrates the full ML pipeline: data ingestion, preprocessing, training, detection (inference), and visualization of results.

TABLE OF CONTENTS

Folder Structure
Features & Tools Used
Installation
Usage
Project Workflow
Results & Visualizations
Next Steps / Ideas for Improvement

____________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________


FOLDER STRUCTURE

ml_nids/
├── data/
│   ├── KDDTrain+.txt       # Training data (NSL-KDD)
│   └── KDDTest+.txt        # (Optional) Test data (NSL-KDD)
├── models/
│   ├── nids_model.joblib   # Trained model (saved after running train.py)
│   ├── features.joblib     # Features list from training
│   └── preprocessing_info.joblib  # Label encoders, scaler, etc. (optional)
├── src/
│   ├── preprocessing/
│   │   └── data_processor.py
│   ├── visualization/
│   │   └── data_visualizer.py
│   ├── models/
│   │   ├── train.py
│   │   ├── detect.py
│   │   └── simple_model.py
├── visualizations/
│   ├── attack_distribution.png
│   ├── feature_correlations.png
│   └── feature_distributions.png
├── requirements.txt
└── README.md  # <--- You place this file here (the top-level folder).

data/ holds the NSL-KDD dataset files.
models/ stores the serialized .joblib model files after training.
src/preprocessing/ contains data_processor.py for data loading & preprocessing.
src/models/ contains train.py (training script), detect.py / simple_model.py (inference scripts).
src/visualization/ holds data_visualizer.py for generating plots.
visualizations/ is where the .png output of your plots is saved

____________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________


FEATURES & TOOLS USED

Python 3.x
pandas for data manipulation
scikit-learn (RandomForestClassifier, LabelEncoder, StandardScaler)
Matplotlib / Seaborn for data visualization
joblib for saving/loading trained models
NSL-KDD dataset as the basis for network intrusion detection research

____________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________


INSTALLATION

Clone or download this repository.
Install the required libraries:
	pip install -r requirements.txt
(Optional) Verify you have the correct data files (KDDTrain+.txt, KDDTest+.txt) in the data folder. If you don’t, you can download them from the NSL-KDD Dataset site.

____________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________


USAGE

1. Train the Model
Run the training script from the project’s root directory (ml_nids/):
	cd ml_nids
	python -m src.models.train
This will:
	Load data/KDDTrain+.txt.
	Preprocess and train a RandomForest model.
	Save the model to models/nids_model.joblib (and optionally features.joblib & preprocessing_info.joblib).
2. Detect Intrusions
There are two scripts you can use:
	simple_model.py
		python src/models/simple_model.py
		This will load a small sample of training data (first 5 rows) and print out predicted classes.
	detect.py
		python src/models/detect.py
		Similarly, loads a sample of data and shows predictions and confidence scores.
Note: Adjust paths in these scripts if needed (e.g., if your data/model files are in different locations).
3. Visualize Data
python src/visualization/data_visualizer.py
Generates:
	attack_distribution.png (class distribution)
	feature_correlations.png (heatmap of top correlated features)
	feature_distributions.png (histograms of selected features)
Saved in the visualizations/ folder.

____________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________


PROJECT WORKFLOW

1. Data Processing (data_processor.py):
	Loads NSL-KDD data, handles categorical encoding with LabelEncoder, scales numeric features with StandardScaler.
2. Model Training (train.py):
	Uses data_processor.py to preprocess the data.
	Trains a RandomForestClassifier.
	Saves the trained model and feature metadata with joblib.
3. Detection (simple_model.py / detect.py):
	Loads the saved model and makes predictions on new data.
	Optionally processes categorical fields on the fly (though in a production scenario, you’d reuse the same fitted encoders as training).
4. Visualization (data_visualizer.py):
	Provides plots for label distribution, feature correlations, and feature distributions.

____________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________


RESULTS & VISUALIZATIONS

Attack Distribution
	Shows counts of different attack types (or normal traffic) in the dataset.
Feature Correlation Heatmap
	Highlights which features have the strongest relationships.
Feature Distributions
	Helps you see how certain numerical features (e.g., duration, src_bytes, etc.) are distributed across the dataset.
Model Performance (Optional)
	You can run detect.py on KDDTest+.txt (if you have it) and measure accuracy or view a confusion matrix (scikit-learn’s classification_report, confusion_matrix) to gauge how well the model detects intrusions vs. normal traffic.

____________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________


NEXT STEPS/IDEAS FOR IMPROVEMENT

1. Real-Time Packet Capture: Integrate something like Scapy to parse live network packets and feed them to the model.
2. Handling Unknown Categories: If new protocols or services appear that weren’t in the training data, build a fallback/unknown handling strategy.
3. Hyperparameter Tuning: Use GridSearchCV or RandomizedSearchCV to find the best RandomForest parameters.
4. Add More Attack Classes: Possibly break down multi-class classification and see how well it distinguishes specific attacks.
5. Evaluation Metrics: Add a script that prints precision, recall, and F1-scores for each attack type.