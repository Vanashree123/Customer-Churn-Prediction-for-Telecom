🚀 Customer Churn Prediction for Telecom
Predict which telecom customers are likely to cancel service using machine learning and data-driven modeling.

Overview
Customer churn—when users discontinue service—is a critical business challenge for telecom operators. This project leverages classic machine learning techniques to identify high-risk customers and support proactive retention strategies.

🧠 Project Flow
Data Loading
Uses the Kaggle “Telecom Churn” dataset (~7,000 customers, ~21 features) including account, usage, and service-level variables. 
arxiv.org
+10
github.com
+10
arxiv.org
+10

Exploratory Data Analysis (EDA)
Visualizes and analyzes feature distributions and relationships to churn—e.g. international plan, customer service calls, contract type, payment method, etc.

Data Preprocessing

Handles missing or duplicate entries

Clusters categorical variables

Applies PCA to reduce dimensionality

Encodes categorical variables and scales numeric ones

Modeling

Tested multiple algorithms: Logistic Regression, K-Nearest Neighbors, Random Forest

Used ensemble methods: Bagging, Grid Search for hyperparameter tuning

Observed 97% test accuracy when combining clustering + PCA + Random Forest 
github.com
github.com
+2
github.com
+2
github.com
+1

Ensemble and Tuning

Bagging and GridSearch improved performance by ~2% over baseline Random Forest

Selected model with optimal balance of accuracy and generalization

📋 Repository Structure
bash
Copy
Edit
├── data/
│   └── telecom_churn.csv        # Raw dataset
├── notebooks/
│   └── churn_analysis.ipynb     # EDA and preprocessing
├── models/
│   └── trained_model.pkl        # Serialized best model
├── src/
│   └── preprocessing.py         # Feature engineering
│   └── train_model.py           # Model training pipeline
│   └── evaluate.py              # Evaluation scripts
├── requirements.txt
└── README.md
🛠 Getting Started & Requirements
Language: Python 3.7+

Libraries: pandas, numpy, scikit-learn, matplotlib, seaborn, joblib

Installation
bash
Copy
Edit
git clone https://github.com/Vanashree123/Customer-Churn-Prediction-for-Telecom.git
cd "Customer Churn Prediction for Telecom"
python3 -m venv venv
source venv/bin/activate      # Linux/macOS
# or
venv\Scripts\activate         # Windows

pip install -r requirements.txt
⚙️ Usage
EDA & preprocessing
Run the Jupyter notebook in notebooks/ to inspect data and prepare training features.

Training the model

bash
Copy
Edit
python src/train_model.py
Evaluating performance

bash
Copy
Edit
python src/evaluate.py
Outputs metrics such as confusion matrix, accuracy, precision, recall, and AUC.

✅ Results & Evaluation
Model	Accuracy (test)
Random Forest (baseline)	~95%
RF + PCA + Clustering	~97%
After hyperparameter tuning	Marginal gains

Final model achieves ~97% accuracy on held-out data and is suitable for identifying churn‑risk customers.

🔍 Insights & Feature Importance
Top predictors of churn typically include:

International plan usage

Customer service call count

Contract type (month-to-month vs. annual)

Monthly charges and tenure

These patterns align with broader churn‑prediction findings in telecom literature. 
github.com
+2
github.com
+2
arxiv.org
github.com

📈 Business Value
Early warning system for high-risk customers

Targeted retention campaigns reduce churn and acquisition costs

Data-driven strategy helps prioritize offers and support interventions

🧩 Extending the Project
Consider these potential improvements:

Add more models like XGBoost, LightGBM

Explore stacking or other ensemble strategies

Use explainability tools (e.g. SHAP) for feature-level insights

Deploy via web app (Flask, Streamlit) or batch scoring pipeline


📄 License
This project is licensed under the MIT License.
