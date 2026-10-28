import pandas as pd, numpy as np, matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, roc_curve, auc

df = pd.read_csv("bank-additional-full.csv", sep=";")
if "duration" in df.columns: df = df.drop(columns=["duration"])
y = df["y"].apply(lambda x: 1 if x == "yes" else 0)
X = df.drop(columns=["y"])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

num, cat = X_train.select_dtypes(include=["int64", "float64"]).columns, X_train.select_dtypes(include=["object"]).columns
prep = ColumnTransformer([("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), num), ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat)])

model = Pipeline([("p", prep), ("m", LogisticRegression(max_iter=1000, random_state=42))])
model.fit(X_train, y_train)

# Сохранение результатов в текстовый файл
report = []
report.append("=== BLOCK A ===")
report.append(str(confusion_matrix(y_test, model.predict(X_test))))

rf = Pipeline([("p", prep), ("m", RandomForestClassifier(random_state=42, n_jobs=-1))])
auc_lr = cross_val_score(model, X_train, y_train, cv=3, scoring="roc_auc").mean()
report.append("=== BLOCK B ===")
report.append(f"AUC LR: {auc_lr}")
rf.fit(X_train, y_train)
p_rf = rf.predict_proba(X_test)[:, 1]
report.append(str(confusion_matrix(y_test, (p_rf >= np.percentile(p_rf, 90)).astype(int))))

report.append("=== BLOCK C ===")
bal = Pipeline([("p", prep), ("m", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42))])
bal.fit(X_train, y_train)
p_bal = bal.predict_proba(X_test)[:, 1]
report.append(str(confusion_matrix(y_test, (p_bal >= np.percentile(p_bal, 90)).astype(int))))

with open("report.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(report))

# Построение графика ROC-кривой
plt.figure(figsize=(8, 6))
for name, clf in [("Logistic Regression", model), ("Random Forest", rf)]:
    probs = clf.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, probs)
    plt.plot(fpr, tpr, label=f"{name} (AUC = {auc(fpr, tpr):.3f})")

plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves")
plt.legend()
plt.savefig("roc_curve.png")
print("Готово! Отчет сохранен в report.txt, а график — в roc_curve.png")

