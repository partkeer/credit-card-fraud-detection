# Do some preprocessing then train simple model
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, precision_recall_curve, auc, roc_curve, roc_auc_score

# Load dataset
df = pd.read_csv('creditcard.csv')

# Remove duplicate rows
df = df.drop_duplicates()

# Split into features and target
X = df.drop('Class', axis=1)
y = df['Class']

# Split the data. Will use 20% for testing, 80% for training + validation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)

# Train random forest model
model = RandomForestClassifier(n_estimators=100, random_state=1)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)
# Probability of positive class
y_score = model.predict_proba(X_test)[:,1]

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