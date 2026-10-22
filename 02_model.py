import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedGroupKFold, GridSearchCV, cross_validate
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, ConfusionMatrixDisplay, RocCurveDisplay)

CLEAN_PATH = "data/clean.csv"
LABEL = "low_income"
KEYS = ["year", "cd"]      # identifier columns

df = pd.read_csv(CLEAN_PATH)
if "district" not in df.columns:
    df["district"] = df["State"].astype(str) + "_" + df["cd"].astype(str)

X = df.drop(columns=[c for c in KEYS + [LABEL, "district"] if c in df.columns])
y = df[LABEL]
groups = df["district"]
print("Features:", X.columns.tolist())
print(y.value_counts(normalize=True))

# 80/20 split
cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
train_idx, test_idx = next(cv.split(X, y, groups))
X_train, y_train, g_train = X.iloc[train_idx], y.iloc[train_idx], groups.iloc[train_idx]
X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]

# Baseline CV (training data only)
base = RandomForestClassifier(class_weight="balanced", random_state=42, n_jobs=-1)
res = cross_validate(base, X_train, y_train, groups=g_train, cv=cv,
                     scoring=["accuracy", "precision", "recall", "f1", "roc_auc"])
print("Baseline CV:", {k[5:]: round(v.mean(), 3) for k, v in res.items() if k.startswith("test_")})

# Tune
grid = {"n_estimators": [200, 500], "max_depth": [None, 10, 20], "max_features": ["sqrt", 0.3]}
search = GridSearchCV(
    RandomForestClassifier(class_weight="balanced", random_state=42, n_jobs=-1),
    grid, cv=cv, scoring={"f1": "f1", "roc_auc": "roc_auc"}, refit="f1")
search.fit(X_train, y_train, groups=g_train)
print("Best params:", search.best_params_, "| CV F1:", round(search.best_score_, 3))

# Final test evaluation
model = search.best_estimator_
pred = model.predict(X_test)
proba = model.predict_proba(X_test)[:, 1]
print("Test accuracy :", round(accuracy_score(y_test, pred), 3))
print("Test precision:", round(precision_score(y_test, pred), 3))
print("Test recall   :", round(recall_score(y_test, pred), 3))
print("Test F1       :", round(f1_score(y_test, pred), 3))
print("Test ROC-AUC  :", round(roc_auc_score(y_test, proba), 3))

ConfusionMatrixDisplay.from_predictions(y_test, pred)
plt.savefig("confusion.png", dpi=200); plt.close()
RocCurveDisplay.from_predictions(y_test, proba)
plt.savefig("roc.png", dpi=200); plt.close()

# Feature importance
imp = pd.Series(model.feature_importances_, index=X.columns).sort_values().tail(15)
imp.plot(kind="barh", figsize=(8, 6), title="Top 15 feature importances")
plt.tight_layout(); plt.savefig("importance.png", dpi=200); plt.close()
print(imp.sort_values(ascending=False))