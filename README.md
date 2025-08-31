# credit-card-fraud-detection
Predict whether credit card transactions are fraudulent or genuine.

Dataset is obtained from Kaggle: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

Final model is a simple average between XGBoost and KNN. When predicting positive classes, XGBoost tends to be too cautious and KNN tends to be incautious - simple averaging seems to consistently improve both precision and recall.
