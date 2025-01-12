#!/usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib
matplotlib.use('Agg')  # Set this first
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

class NetworkDataVisualizer:
    def __init__(self, data_path: str):
        """
        Initialize the visualizer with the path to the processed data
        """
        self.data = pd.read_csv(data_path, names=[
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
        ])

    def plot_attack_distribution(self):
        """
        Plot the distribution of different attack types and save to file
        """
        plt.figure(figsize=(15, 8))
        attack_counts = self.data['label'].value_counts()
        sns.barplot(x=attack_counts.index, y=attack_counts.values)
        plt.title('Distribution of Network Traffic Types', fontsize=14, pad=20)
        plt.xlabel('Attack Type', fontsize=12)
        plt.ylabel('Count', fontsize=12)
        plt.xticks(rotation=45, ha='right', fontsize=10)
        plt.yticks(fontsize=10)
        plt.tight_layout()
        plt.savefig('C:/Users/talab/Documents/ml_nids/visualizations/attack_distribution.png', 
                    dpi=300, bbox_inches='tight')
        plt.close()

    def plot_feature_correlations(self, n_features=15):
        """
        Plot correlation matrix of numerical features and save to file
        Focus on the most important correlations for better readability
        """
        numerical_data = self.data.select_dtypes(include=['float64', 'int64'])
        corr_matrix = numerical_data.corr()
        
        # Get the most important correlations
        correlations = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i):
                if i != j:
                    feature1 = corr_matrix.columns[i]
                    feature2 = corr_matrix.columns[j]
                    correlation = abs(corr_matrix.iloc[i, j])
                    correlations.append((feature1, feature2, correlation))
        
        # Sort by correlation value and get top N
        correlations.sort(key=lambda x: x[2], reverse=True)
        top_correlations = correlations[:n_features]
        
        # Create a subset of the correlation matrix with only the most important features
        important_features = set()
        for feat1, feat2, _ in top_correlations:
            important_features.add(feat1)
            important_features.add(feat2)
        
        important_features = list(important_features)
        subset_corr = corr_matrix.loc[important_features, important_features]
        
        # Plot with larger figure size and adjusted font sizes
        plt.figure(figsize=(15, 12))
        sns.heatmap(subset_corr, 
                    annot=True, 
                    cmap='coolwarm', 
                    center=0,
                    fmt='.2f',
                    square=True,
                    annot_kws={'size': 8},
                    cbar_kws={'label': 'Correlation Coefficient'})
        
        plt.title('Feature Correlation Matrix (Most Important Features)', pad=20, fontsize=14)
        plt.xticks(rotation=45, ha='right', fontsize=8)
        plt.yticks(rotation=0, fontsize=8)
        plt.tight_layout()
        plt.savefig('C:/Users/talab/Documents/ml_nids/visualizations/feature_correlations.png', 
                    dpi=300, bbox_inches='tight')
        plt.close()

    def plot_feature_distributions(self, features=None):
        """
        Plot distributions of specified numerical features and save to file
        """
        if features is None:
            features = ['duration', 'src_bytes', 'dst_bytes', 'count']
        
        plt.figure(figsize=(15, 10))
        for i, feature in enumerate(features, 1):
            plt.subplot(2, 2, i)
            sns.histplot(self.data[feature], bins=50)
            plt.title(f'Distribution of {feature}', fontsize=12)
            plt.xlabel(feature, fontsize=10)
            plt.ylabel('Count', fontsize=10)
            plt.xticks(fontsize=8)
            plt.yticks(fontsize=8)
        plt.tight_layout()
        plt.savefig('C:/Users/talab/Documents/ml_nids/visualizations/feature_distributions.png', 
                    dpi=300, bbox_inches='tight')
        plt.close()

if __name__ == "__main__":
    visualizer = NetworkDataVisualizer("C:\\Users\\talab\\Documents\\ml_nids\\data\\KDDTrain+.txt")
    print("Creating visualizations...")
    
    visualizer.plot_attack_distribution()
    print("Created attack distribution plot")
    
    visualizer.plot_feature_correlations()
    print("Created feature correlations plot")
    
    visualizer.plot_feature_distributions()
    print("Created feature distributions plot")
    
    print("All visualizations have been saved in the 'visualizations' folder with improved readability.")