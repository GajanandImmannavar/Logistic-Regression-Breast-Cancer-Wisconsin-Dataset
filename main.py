from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()
X = data.data
y = data.target

print("\n\n")
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



from sklearn.preprocessing import StandardScaler
scale = StandardScaler()
X_train_scaled = scale.fit_transform(X_train)


from sklearn.linear_model import LogisticRegression
model = LogisticRegression(max_iter=1000, solver='lbfgs')
model.fit(X_train_scaled, y_train)
X_test_scaled = scale.transform(X_test)
y_pred = model.predict(X_test_scaled)

train_acc = model.score(X_train_scaled, y_train)
test_acc = model.score(X_test_scaled, y_test)

print("Training Accuracy:", round(train_acc, 3))
print("Testing Accuracy:", round(test_acc, 3))

print("\n\n")
if train_acc - test_acc < 0.05:
    print("Good Model ✅ No Major Overfitting\n\n")
else:
    print("Warning ⚠️ Possible Overfitting\n\n")


from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print("\n\n")

from sklearn.model_selection import cross_val_score
cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
print("Cross Validation Scores:", cv_scores)
print("\n\n")
print("Average CV Score:", round(cv_scores.mean(), 3))
print("\n\n")

from sklearn.metrics import roc_auc_score
y_prob = model.predict_proba(X_test_scaled)[:, 1]
auc_score = roc_auc_score(y_test, y_prob)
print("ROC-AUC Score:", round(auc_score, 3))
print("\n\n")
if auc_score > 0.9:
    print("Excellent Model ⭐")
    print("\n\n")
elif auc_score > 0.8:
    print("Good Model 👍")
    print("\n\n")
else:
    print("Needs Improvement ⚠️")
    print("\n\n")

    
print("--------------------------------------------------")
print("TOP FEATURE IMPORTANCE")
print("--------------------------------------------------")

import pandas as pd
feature_importance = pd.Series(
    model.coef_[0],
    index=data.feature_names
).sort_values(ascending=False)

print(feature_importance.head(10))
print("\n\n")

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)
print("\n\n")


from sklearn.metrics import classification_report
report = classification_report(y_test, y_pred)
print("Classification Report:")
print(report)




from sklearn.metrics import RocCurveDisplay
import matplotlib.pyplot as plt

RocCurveDisplay.from_estimator(model, X_test_scaled, y_test)
plt.title("ROC Curve - Breast Cancer Prediction")
plt.grid()
plt.show()
