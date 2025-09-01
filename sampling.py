# Train logistic regression/SVC using oversampled/undersampled data, and compare results.
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, f1_score
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss

# Load dataset
df = pd.read_csv('creditcard.csv')

# Remove duplicate rows
df = df.drop_duplicates()

# Split into features and target
X = df.drop('Class', axis=1)
y = df['Class']

# Split the data. Will use 20% for testing, 80% for training + validation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)

print('----- Before oversampling -----')
print(f'Shape: {X_train.shape}')
print(f'Number of positive samples: {sum(y_train == 1)}')
print(f'Number of negative samples: {sum(y_train == 0)}')

# train a logistic regression model (scaling data beforehand)
model1 = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression())
])
model1.fit(X_train, y_train)
y_pred1 = model1.predict(X_test)
print(f'Confusion matrix: \n{confusion_matrix(y_test, y_pred1)}')
print('F1 score: ', f1_score(y_test, y_pred1))



# create oversampled dataset
smote = SMOTE(random_state=1, k_neighbors=5)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
print('\n----- After oversampling -----')
print(f'Shape: {X_train_resampled.shape}')
print(f'Number of positive samples: {sum(y_train_resampled == 1)}')
print(f'Number of negative samples: {sum(y_train_resampled == 0)}')

# train a logistic regression model with oversampled dataset
model2 = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression())
])
model2.fit(X_train_resampled, y_train_resampled)
y_pred2 = model2.predict(X_test)    # model predicts more positives as expected (leading to worse performance)
print(f'Confusion matrix: \n{confusion_matrix(y_test, y_pred2)}')
print('F1 score: ', f1_score(y_test, y_pred2))



# create undersampled dataset
nm = NearMiss(sampling_strategy=0.3, version=1, n_neighbors=3)
X_train_undersampled, y_train_undersampled = nm.fit_resample(X_train, y_train)

print('\n----- After undersampling (SVM) -----')
print(f'Shape: {X_train_undersampled.shape}')
print(f'Number of positive samples: {sum(y_train_undersampled== 1)}')
print(f'Number of negative samples: {sum(y_train_undersampled == 0)}')

model3 = Pipeline([
    ('scaler', StandardScaler()),
    ('model', SVC())
])
model3.fit(X_train_undersampled, y_train_undersampled)
y_pred3 = model3.predict(X_test)
print(f'Confusion matrix: \n{confusion_matrix(y_test, y_pred3)}')
print('F1 score: ', f1_score(y_test, y_pred3))