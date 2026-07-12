import pandas as pd
import numpy as np

df = pd.read_csv("cardio_train.csv", sep=';')

# age in years
df['age_years'] = df['age'] / 365.25

# male
df['male'] = (df['gender'] == 2).astype(int)

# systolic BP
df['sysBP'] = df['ap_hi']

# smoker
df['currentSmoker'] = df['smoke']

# diabetes
df['diabetes'] = (df['gluc'] > 1).astype(int)

# BMI
df['BMI'] = df['weight'] / ((df['height']/100)**2)

# target
df['target'] = df['cardio']

print(df[['male',
          'age_years',
          'sysBP',
          'currentSmoker',
          'diabetes',
          'BMI',
          'target']].head())


from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

# Features and target
X = df[['male',
        'age_years',
        'sysBP',
        'currentSmoker',
        'diabetes',
        'BMI']]

y = df['target']

# Same model as Framingham
model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("lr", LogisticRegression(
        max_iter=1000,
        solver="lbfgs"
    ))
])

# 5-fold cross validation
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

auc_scores = []

for train_idx, test_idx in cv.split(X, y):

    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]

    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]

    model.fit(X_train, y_train)

    y_prob = model.predict_proba(X_test)[:,1]

    auc = roc_auc_score(y_test, y_prob)

    auc_scores.append(auc)

print("\nExternal Validation Results")
print("Fold AUCs:", auc_scores)
print("Mean ROC-AUC:", round(np.mean(auc_scores),3))
print("Std:", round(np.std(auc_scores),3))

mean_auc = np.mean(auc_scores)
std_auc = np.std(auc_scores)

margin = 1.96 * (std_auc/np.sqrt(5))

print()
print(
    f"95% CI: ({float(mean_auc-margin):.3f}, "
    f"{float(mean_auc+margin):.3f})"
)

from sklearn.metrics import brier_score_loss

model.fit(X, y)

y_prob = model.predict_proba(X)[:,1]

brier = brier_score_loss(y, y_prob)

print("Brier Score:",
      round(brier,3))

from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt

prob_true, prob_pred = calibration_curve(
    y,
    y_prob,
    n_bins=10
)

plt.figure(figsize=(6,6))

plt.plot(
    prob_pred,
    prob_true,
    marker='o',
    label='External Validation'
)

plt.plot(
    [0,1],
    [0,1],
    '--',
    label='Perfect Calibration'
)

plt.xlabel('Predicted Probability')
plt.ylabel('Observed Probability')
plt.title('External Validation Calibration')

plt.legend()

plt.show()




