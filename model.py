# Do some preprocessing then train simple model
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, precision_recall_curve, auc, roc_curve, roc_auc_score
from xgboost import XGBClassifier

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

# Create a pipeline which scales data, then applies model
# Use xgboost
xgb_pipe = Pipeline([
    ('scaler', StandardScaler()),
    # hyperparameters are tuned
    ('model', XGBClassifier(n_estimators=300, learning_rate=0.05, min_child_weight=2, subsample=0.9, colsample_bytree=0.5))
])

# Use knn
knn_pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('model', KNeighborsClassifier(n_neighbors=3))
])

xgb_pipe.fit(X_train, y_train)
knn_pipe.fit(X_train, y_train)

# Probability of positive class
y_score_xgb = xgb_pipe.predict_proba(X_test)[:,1]
y_score_knn = knn_pipe.predict_proba(X_test)[:,1]

# Use simple averaging for final prediction
# (xgboost tends to be too cautious and knn tends to be under-cautious when predicting positive classes;
# using a simple average consistently increases f1-score across test sets)
y_score = 0.5 * (y_score_xgb + y_score_knn)
y_pred = np.round(y_score)

# Evaluate predictions
confusion_mat = confusion_matrix(y_test, y_pred)
print('Confusion matrix:\n',confusion_mat)

precision, recall, thresholds = precision_recall_curve(y_test, y_score)
plt.plot(recall, precision)
plt.title('Precision recall curve')
plt.show()

auc_pr = auc(recall, precision)
print('Area under precision-recall curve:', auc_pr)

fpr, tpr, thresholds_roc = roc_curve(y_test, y_score)
plt.plot(fpr, tpr)
plt.title('ROC curve')
plt.show()

auc_roc = roc_auc_score(y_test, y_score)
print('Area under ROC curve:', auc_roc)