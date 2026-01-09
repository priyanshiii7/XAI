# Explainable AI for Medical Diagnosis: IEEE CBMS 2026

A comprehensive implementation of Explainable Artificial Intelligence (XAI) techniques for medical diagnosis using the Breast Cancer Wisconsin dataset. This project demonstrates how to build interpretable machine learning models that balance high performance with transparency, addressing the critical need for explainability in clinical decision support systems.

Overview
This notebook implements and compares multiple machine learning models with state-of-the-art explainability methods, providing insights into model decision-making processes that are crucial for clinical adoption of AI systems.
Key Features

Multiple Model Architectures: Implementation of Random Forest, XGBoost, and Neural Network classifiers
Comprehensive Explainability: Integration of SHAP, LIME, and Feature Importance methods
Performance Evaluation: Detailed metrics including accuracy, AUC-ROC, precision, recall, and F1-score
Publication-Ready Visualizations: High-quality figures suitable for academic papers
Real Experimental Results: Actual performance metrics and computation times

Dataset
Breast Cancer Wisconsin Diagnostic Dataset

569 samples
30 numerical features
Binary classification: Malignant (0) vs Benign (1)
Publicly available through scikit-learn

Models Implemented
1. Random Forest Classifier

100 decision trees
Maximum depth: 10
Optimized for interpretability and performance balance

2. XGBoost Classifier

Gradient boosting implementation
Learning rate: 0.1
Maximum depth: 6
Cross-entropy loss optimization

3. Neural Network

Architecture: 30 → 64 → 32 → 2
ReLU activation functions
Dropout regularization (0.3)
Adam optimizer with learning rate 0.001

Explainability Methods
SHAP (SHapley Additive exPlanations)

TreeExplainer for tree-based models
DeepExplainer for neural networks
Provides both global and local interpretations
Feature contribution analysis

LIME (Local Interpretable Model-agnostic Explanations)

Local linear approximations
Individual prediction explanations
Superpixel-based interpretability

Feature Importance

Built-in Random Forest importance
Permutation importance for model-agnostic analysis
Comparative analysis across methods

Installation and Setup
Prerequisites
bashPython 3.8+
Google Colab (recommended) or Jupyter Notebook
Required Libraries
bashpip install shap lime xgboost scikit-learn pandas numpy matplotlib seaborn torch torchvision
```

### Google Colab Setup
1. Open the notebook in Google Colab
2. Runtime → Change runtime type → Select "T4 GPU"
3. Run all cells sequentially

## Project Structure
```
├── Data Loading and Preprocessing
│   ├── Dataset loading
│   ├── Train-test split (80-20)
│   └── Feature standardization
│
├── Model Training
│   ├── Random Forest implementation
│   ├── XGBoost implementation
│   └── Neural Network (PyTorch)
│
├── Performance Evaluation
│   ├── Accuracy, AUC-ROC, Precision, Recall
│   ├── Confusion matrices
│   └── ROC curves comparison
│
├── Explainability Analysis
│   ├── SHAP value calculation
│   ├── LIME explanations
│   ├── Feature importance extraction
│   └── Visualization generation
│
└── Results Export
    ├── CSV files (metrics tables)
    └── PNG files (visualizations)
Usage
Step 1: Environment Setup
python# Install required packages
!pip install shap lime xgboost scikit-learn pandas numpy matplotlib seaborn -q

# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
Step 2: Load and Preprocess Data
pythonfrom sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load dataset
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name='target')

# Split and normalize
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
Step 3: Train Models
pythonfrom sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

# Train Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)

# Train XGBoost
xgb_model = xgb.XGBClassifier(n_estimators=100, random_state=42)
xgb_model.fit(X_train_scaled, y_train)
Step 4: Apply Explainability Methods
pythonimport shap
import lime.lime_tabular

# SHAP Analysis
explainer_rf = shap.TreeExplainer(rf_model)
shap_values = explainer_rf.shap_values(X_test_scaled)

# LIME Analysis
explainer_lime = lime.lime_tabular.LimeTabularExplainer(
    X_train_scaled,
    feature_names=data.feature_names,
    class_names=['Malignant', 'Benign'],
    mode='classification'
)
Output Files
CSV Files

model_performance_results.csv: Comprehensive model metrics
xai_comparison.csv: Explainability methods comparison

Visualizations

roc_curves_comparison.png: ROC curves for all models
confusion_matrices.png: Confusion matrices visualization
shap_summary_bar.png: Feature importance ranking
shap_beeswarm.png: Detailed SHAP value distribution
shap_waterfall.png: Single prediction explanation
lime_explanation.png: LIME local explanation
feature_importance_rf.png: Random Forest feature importance

Expected Results
Model Performance

Random Forest: Accuracy ~95%, AUC-ROC ~0.98
XGBoost: Accuracy ~94%, AUC-ROC ~0.97
Neural Network: Accuracy ~93%, AUC-ROC ~0.97

Explainability Metrics

SHAP computation time: <1 second (TreeExplainer)
LIME computation time: ~2-3 seconds per sample
Feature importance: Near-instantaneous

Key Findings

Minimal Performance Trade-off: Interpretable models achieve comparable accuracy to black-box models
Feature Insights: Worst perimeter, worst radius, and mean concave points are consistently most important
Computation Efficiency: TreeExplainer provides fast, accurate explanations for tree-based models
Clinical Applicability: Explanations align with medical knowledge and expert reasoning

Applications

Clinical decision support systems
Medical diagnosis assistance
Regulatory compliance for medical AI
Healthcare professional training
Patient communication and transparency

Citation
If you use this work in your research, please cite:
bibtex@inproceedings{xai_medical_2026,
  title={Explainable Deep Learning Framework for Clinical Decision Support},
  author={Your Name},
  booktitle={IEEE International Symposium on Computer-Based Medical Systems (CBMS)},
  year={2026}
}
References
Core XAI Methods

Lundberg, S.M., & Lee, S.I. (2017). A unified approach to interpreting model predictions. NIPS.
Ribeiro, M.T., Singh, S., & Guestrin, C. (2016). "Why Should I Trust You?": Explaining the Predictions of Any Classifier. KDD.

Medical AI

Topol, E.J. (2019). High-performance medicine: the convergence of human and artificial intelligence. Nature Medicine.
Rajkomar, A., Dean, J., & Kohane, I. (2019). Machine Learning in Medicine. NEJM.

Datasets

Breast Cancer Wisconsin Dataset: UCI Machine Learning Repository

License
This project is licensed under the MIT License.
Acknowledgments

Breast Cancer Wisconsin dataset from UCI Machine Learning Repository
SHAP library developers
LIME library developers
scikit-learn and PyTorch communities

Contact
For questions or collaborations, please open an issue in this repository.
Technical Requirements

RAM: Minimum 8GB (16GB recommended)
GPU: Optional but recommended for neural network training
Disk Space: ~500MB for libraries and outputs
Runtime: Complete execution takes approximately 10-15 minutes

Troubleshooting
Common Issues
Issue: CUDA not available

Solution: Change runtime to GPU in Google Colab (Runtime → Change runtime type → GPU)

Issue: Shape mismatch in SHAP

Solution: Ensure using correct array slicing for multi-class outputs: shap_values[:, :, 1]

Issue: Memory errors

Solution: Reduce batch size or use smaller background dataset for SHAP DeepExplainer

Future Enhancements

Multi-modal data integration (imaging + EHR)
Counterfactual explanations
Interactive visualization dashboard
Uncertainty quantification
Federated learning implementation

Contributing
Contributions are welcome! Please feel free to submit a Pull Request.
