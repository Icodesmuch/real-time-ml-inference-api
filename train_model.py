"""
Training script for churn prediction model.
Uses sklearn's make_classification to generate synthetic data.
"""
import pandas as pd
import joblib
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Task 1.2.1 - Generate synthetic churn prediction data
print("Generating synthetic churn prediction dataset...")
X, y = make_classification(
    n_samples=1000,  # 1000 samples
    n_features=8,    # 8 features (we'll use 5-8 of them)
    n_informative=6, # 6 informative features
    n_redundant=2,   # 2 redundant features
    n_classes=2,     # Binary classification (churn: 0 or 1)
    random_state=42
)

# Create feature names for better understanding
feature_names = [
    'account_age_months',
    'monthly_charges',
    'total_charges',
    'support_calls',
    'contract_length',
    'payment_method_score',
    'usage_frequency',
    'feature_8'
]

# Convert to DataFrame for easier handling
df = pd.DataFrame(X, columns=feature_names)
df['churn'] = y

# Select 5-8 features (we'll use all 8)
selected_features = feature_names
X_selected = df[selected_features]
y = df['churn']

print(f"Dataset shape: {X_selected.shape}")
print(f"Features: {selected_features}")
print(f"Churn rate: {y.mean():.2%}")

# Task 1.2.2 - Split data and train model
print("\nSplitting data into train/test sets...")
X_train, X_test, y_train, y_test = train_test_split(
    X_selected, y, test_size=0.2, random_state=42
)

print(f"Training set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

# Train RandomForestClassifier
print("\nTraining RandomForestClassifier...")
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    max_depth=10
)
model.fit(X_train, y_train)

# Check accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nTest Accuracy: {accuracy:.2%}")

if accuracy < 0.75:
    print("⚠️  Warning: Accuracy is below 75%. Consider tuning hyperparameters.")
else:
    print("✅ Accuracy meets the 75% target!")

# Task 1.2.3 - Save the model
print("\nSaving model to 'model_v1.pkl'...")
joblib.dump(model, 'model_v1.pkl')
print("✅ Model saved successfully!")

# Verify the model can be loaded and makes predictions
print("\nVerifying model can be loaded...")
loaded_model = joblib.load('model_v1.pkl')
test_prediction = loaded_model.predict(X_test[:1])
print(f"✅ Model loaded successfully! Test prediction: {test_prediction[0]}")

# Print feature importance for reference
print("\nFeature Importances:")
for feature, importance in zip(selected_features, model.feature_importances_):
    print(f"  {feature}: {importance:.4f}")