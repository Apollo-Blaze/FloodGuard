import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns

print(sns.__version__)

# Load the data
file_path = 'C:/Users/shrey/OneDrive/Desktop/floodguard/final.csv'  
final = pd.read_csv(file_path)

# Check missing values
NAs = pd.concat([final.isnull().sum()], axis=1, keys=["Final"])
print("Missing values:")
print(NAs[NAs.sum(axis=1) > 0])
final.head(5)

# Check missing values
NAs = pd.concat([final.isnull().sum()], axis=1, keys=["Final"])
print("Missing values:")
print(NAs[NAs.sum(axis=1) > 0])
final.head(5)


# Fill missing values for numerical columns
numerical_cols = ['tavg', 'prcp', 'snow', 'wdir', 'wspd', 'pres']
for col in numerical_cols:
    final[col].fillna(final[col].mean(), inplace=True)

# Create dummy variables
final = pd.get_dummies(final)

# Create severity bins
bins = [final['severity'].min(), 2, 4, final['severity'].max()]
bins.sort()
labels = ['Low', 'Medium', 'High']

# Convert severity to categories
final['severity'] = pd.cut(final['severity'], bins=bins, labels=labels, right=False)


# Split features and target
X = final.drop(['severity'], axis=1)
y = final['severity']


# Split train test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Convert categorical labels to numerical
y_train_num = pd.Categorical(y_train).codes
y_test_num = pd.Categorical(y_test).codes


# Initialize and train XGBoost
xgb_model = xgb.XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=6,
    random_state=42,
    objective='multi:softprob',
    num_class=len(labels),
      eval_metric='mlogloss' 
)


xgb_model.fit(
    X_train, 
    y_train_num,
    eval_set=[(X_test, y_test_num)],
    
    verbose=True
)


# Make predictions
y_pred_proba = xgb_model.predict_proba(X_test)
y_pred_num = np.argmax(y_pred_proba, axis=1)
y_pred = pd.Categorical.from_codes(y_pred_num, categories=labels)




# Print metrics
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))



# Feature importance
feature_importance = pd.DataFrame({
    'feature': X_train.columns,
    'importance': xgb_model.feature_importances_
}).sort_values('importance', ascending=False)



print(feature_importance)

xgb_model.save_model("xgb_model.json")  # Save in JSON format
xgb_model.save_model("xgb_model.bin")  # Save in binary format
