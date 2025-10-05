import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder 
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

df = pd.read_csv('/Users/murtazahussain/Desktop/ChurnProject_Neww/Churn_Modelling.csv')
# print("---------------------------------------------------------------Typical Checks--------------------------------------------------------------------------------------")
# print(df.head())
# print(df.info())
# print(df.isnull().sum())
# print(df[df.duplicated()])
label_encoder = LabelEncoder()

df['Gender'] = label_encoder.fit_transform(df['Gender'])
df = pd.get_dummies(df, columns=['Geography'], drop_first=True)
#print(df.head())

features = ['CreditScore','Gender','Age','Tenure','Balance','NumOfProducts',
'HasCrCard','IsActiveMember','EstimatedSalary','Geography_Germany','Geography_Spain']

X = df[features]
Y = df['Exited']

#Select which data we need to train. if test size is 0.2, 20% data will be tested, 80% will be what the model will train based of
result = train_test_split(X, Y, test_size=0.2, random_state=42)
X_train = result[0]
X_test = result[1]
Y_train = result[2]
Y_test = result[3]




#Feature Scaling: It resizes values so all features contribute equally.
# This is applied before training model :)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
#print(X_train[:5])

#Implementing our model :D with 100 Decision Trees
model = RandomForestClassifier(n_estimators = 100, random_state = 42)
model.fit(X_train,Y_train)

Y_pred = model.predict(X_test)

matrix = confusion_matrix(Y_test, Y_pred)
print("Our Confusion Matrix\n\n",  matrix)
accuracy = accuracy_score(Y_test,Y_pred)
print("\nOur Accuracy Score, ", accuracy)

report = classification_report(Y_test,Y_pred)

print("---------------------------------------------------------------PRECISION REPORT--------------------------------------------------------------------------------------")
print("\n\n\n",report)

importance = model.feature_importances_
print(importance)
indices = np.argsort(importance)[::-1]
print(indices)
names = [features[i] for i in indices]

print(names)

plt.figure(figsize=(10, 6))
plt.title("Feature Importance")
plt.barh(range(X.shape[1]), importance[indices]) 
plt.yticks(range(X.shape[1]), names)
#plt.show()

# Code                    | What it becomes (numbers)                                                            | Meaning
# ---------------------------------------------------------------------------------------------------------------------------------------------------
# range(X.shape[1])   → [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]                                                       → y-axis positions for 11 bars

# importance[indices] → [0.24, 0.18, 0.15, 0.13, 0.09, 0.07, 0.05, 0.04, 0.03, 0.02, 0.00]                       → x-axis lengths (importance values)

# names               → ['Age', 'Balance', 'CreditScore', 'EstimatedSalary', 'Geography_Germany', 
#                       'NumOfProducts', 'IsActiveMember', 'HasCrCard', 'Gender', 'Tenure', 'Geography_Spain']   → y-axis labels

print("="*120)
print("Time to test out different models :D")
print("="*120, "\n\n")


# ================================ LOGISTIC REGRESSION ================================
print("="*50, "LOGISTIC REGRESSION", "="*50)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
log_reg = LogisticRegression(random_state=42)
log_reg.fit(X_train, Y_train)
Y_pred_log_reg = log_reg.predict(X_test)

# Evaluate the model
conf_matrix_log_reg = confusion_matrix(Y_test, Y_pred_log_reg)
class_report_log_reg = classification_report(Y_test, Y_pred_log_reg)
accuracy_log_reg = accuracy_score(Y_test, Y_pred_log_reg)
print("\nConfusion Matrix:\n", conf_matrix_log_reg)
print("\nClassification Report:\n", class_report_log_reg)
print(f"\nAccuracy: {accuracy_log_reg:.4f}\n")





# ================================ SUPPORT VECTOR MACHINES ================================
print("="*50, "SUPPORT VECTOR MACHINES", "="*50)
from sklearn.svm import SVC
svm_model = SVC(kernel='linear', random_state=42)
svm_model.fit(X_train, Y_train)
y_pred_svm = svm_model.predict(X_test)

# Evaluate the model
conf_matrix_svm = confusion_matrix(Y_test, y_pred_svm)
class_report_svm = classification_report(Y_test, y_pred_svm)
accuracy_svm = accuracy_score(Y_test, y_pred_svm)
print("\nConfusion Matrix:\n", conf_matrix_svm)
print("\nClassification Report:\n", class_report_svm)
print(f"\nAccuracy: {accuracy_svm:.4f}\n")





# ================================ KNN MODELS ================================
print("="*50, "KNN MODELS", "="*50)
from sklearn.neighbors import KNeighborsClassifier
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, Y_train)
y_pred_knn = knn_model.predict(X_test)

# Evaluate the model
conf_matrix_knn = confusion_matrix(Y_test, y_pred_knn)
class_report_knn = classification_report(Y_test, y_pred_knn)
accuracy_knn = accuracy_score(Y_test, y_pred_knn)
print("\nConfusion Matrix:\n", conf_matrix_knn)
print("\nClassification Report:\n", class_report_knn)
print(f"\nAccuracy: {accuracy_knn:.4f}\n")





# ================================ GRADIENT BOOSTER ================================
print("="*50, "GRADIENT BOOSTER", "="*50)
from sklearn.ensemble import GradientBoostingClassifier
gbm_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
gbm_model.fit(X_train, Y_train)
y_pred_gbm = gbm_model.predict(X_test)

# Evaluate the model
conf_matrix_gbm = confusion_matrix(Y_test, y_pred_gbm)
class_report_gbm = classification_report(Y_test, y_pred_gbm, zero_division=0)
accuracy_gbm = accuracy_score(Y_test, y_pred_gbm)

print("\nConfusion Matrix:\n", conf_matrix_gbm)
print("\nClassification Report:\n", class_report_gbm)
print(f"\nAccuracy: {accuracy_gbm:.4f}\n")





print("="*120)
print("All models evaluated!")
print("="*120)

