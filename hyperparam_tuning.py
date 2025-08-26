# Code for hyperparameter tuning
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, precision_recall_curve, auc, roc_curve, roc_auc_score, make_scorer

# Load dataset
df = pd.read_csv('creditcard.csv')

# Remove duplicate rows
df = df.drop_duplicates()

# Split into features and target
X = df.drop('Class', axis=1)
y = df['Class']

# Train isolation forest
contamination = np.round(y.sum()/len(y), 3) # percentage of fraudulent transactions
iso_forest = IsolationForest(n_estimators=100, max_samples=256, contamination=contamination)
iso_forest.fit(X)

# Calculate anomaly scores and add these to features
anomaly_scores = iso_forest.decision_function(X)
X['anomaly_scores'] = anomaly_scores

# Split the data. Will use 20% for testing, 80% for training + validation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)

# Define custom scoring function to calculate area under precision-recall curve (AUPRC)
def auprc(y_true, y_scores):
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    return auc(recall, precision)

scorer = make_scorer(auprc, response_method="predict_proba")

# Create a pipeline which scales data, then applies model
# Use random forest model
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('model', RandomForestClassifier(n_estimators=100, random_state=1))
])

# Use GridSearchCV to tune the model's hyperparameters
param_grid = {
    'model__n_estimators': [50, 100, 150, 200],
    # 'model__max_depth': [None, 10, 30, 50],
    'model__min_samples_leaf': [2, 4, 6]
}

grid_search = GridSearchCV(estimator=pipe, param_grid=param_grid, cv=5, scoring=scorer, n_jobs=-1, verbose=3)
grid_search.fit(X_train, y_train)

# Get best results
print(f"Best parameters found: {grid_search.best_params_}")
print(f"Best cross-validation score: {grid_search.best_score_}")


# Export results to csv for analysis
cv_results = pd.DataFrame(grid_search.cv_results_)
cv_results.to_csv('randomForest1.csv', index=False)