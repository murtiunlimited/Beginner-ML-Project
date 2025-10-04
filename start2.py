import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
df = pd.read_csv("clean_churn_data.csv")
X = df[['plan_type', 'age', 'gender', 'avg_minutes', 'num_logins', 'support_tickets', 'amount_paid', 'auto_renew']]
y = df['churned']
X = pd.get_dummies(X, drop_first=True)
columns_used_for_model = X.columns
result = train_test_split(X, y, test_size=0.25, random_state=42)
X_train = result[0]
X_test = result[1]
y_train = result[2]
y_test = result[3]
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
print("Model training complete!")
new_customers = pd.read_csv("new_customers.csv")
if "churned" in new_customers.columns:
    new_customers = new_customers.drop(columns=["churned"])
new_customers_encoded = pd.get_dummies(new_customers)
new_customers_encoded = new_customers_encoded.reindex(
    columns=columns_used_for_model, fill_value=0
)
churn_predictions = model.predict(new_customers_encoded)
new_customers["predicted_churn"] = churn_predictions
new_customers.to_csv("churn_predictions.csv", index=False)
print("Predictions saved to churn_predictions.csv")
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
