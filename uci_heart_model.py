# ------------------ Part -1 -----------------------------

import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier



# Columns names from UCI Document

columns = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak",
    "slope", "ca", "thal", "target"
    ]

# Load Data

df = pd.read_csv(
    "uci_heart.csv",
    header = None,
    names = columns)

# Replace '?' with 'NaN'

df.replace('?', np.nan, inplace=True)

# Convert all columns to numeric

df = df.apply(pd.to_numeric)

# Binary target: 0-> no heart disease, 1-> heart disease present

df["target"] = (df["target"]>0).astype(int)

# Splitting features and target

X = df.drop(columns=["target"])
y = df["target"]

# Identifying categorical_columns and numerical columns

categorical_cols = [
    "sex", "cp", "fbs", "restecg","exang", "slope", "thal"
    ]
numeric_cols = [
    col for col in X.columns if col not in categorical_cols
    ]

# Defining preprocessing

numeric_transformer = Pipeline(
    steps = [
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
        ])

categorical_transformer = Pipeline(
    steps = [
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown = "ignore"))
        ])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_cols),
        ("cat", categorical_transformer, categorical_cols)
        ])
         

# Cross Validation strategy

cv = StratifiedKFold(
    n_splits = 5,
    shuffle = True,
    random_state =42
    )

# Logistic Regression model

pipe_lr = Pipeline(
    steps = [
        ("preprocess", preprocessor),
        ("model", LogisticRegression(max_iter=1000))
        ])

lr_auc = cross_val_score(
    pipe_lr,
    X,
    y,
    cv=cv,
    scoring = "roc_auc"
    )

from scipy.stats import t

n = len(lr_auc)
mean_auc = lr_auc.mean()
std_auc = lr_auc.std(ddof=1)

t_value = t.ppf(0.975, df=n-1)

margin = t_value * std_auc / np.sqrt(n)

lower_ci = mean_auc - margin
upper_ci = mean_auc + margin

print("\nLogistic Regression")
print(f"Mean ROC-AUC: {mean_auc:.3f}")
print(f"95% CI: ({lower_ci:.3f}, {upper_ci:.3f})")


# Random Forest model(non-linear)

pipe_rf = Pipeline(
    steps=[
        ("preprocess", preprocessor),
        ("model", RandomForestClassifier(
            n_estimators=300,
            random_state = 42,
            n_jobs = -1
        ))
        ])

rf_auc = cross_val_score(
    pipe_rf,
    X,
    y,
    cv=cv,
    scoring="roc_auc"
    )

n = len(rf_auc)
mean_auc = rf_auc.mean()
std_auc = rf_auc.std(ddof=1)

t_value = t.ppf(0.975, df=n-1)

margin = t_value * std_auc / np.sqrt(n)

lower_ci = mean_auc - margin
upper_ci = mean_auc + margin

print("\nRandom Forest")
print(f"Mean ROC-AUC: {mean_auc:.3f}")
print(f"95% CI: ({lower_ci:.3f}, {upper_ci:.3f})")

# Train final Logistic Regression model on full data

pipe_lr.fit(X,y)

# Extract the trained Logistic Regression model
lr_model = pipe_lr.named_steps["model"]

# Extract preprocessing step
preprocess = pipe_lr.named_steps["preprocess"]

# Get feature names after preprocessing
feature_names = preprocess.get_feature_names_out()

# Build a coefficient table

coef_df = pd.DataFrame({
    "feature": feature_names,
    "coefficient": lr_model.coef_[0]
    })

#Add odds ratios
coef_df["odds_ratio"] = np.exp(coef_df["coefficient"])

#Sort by importance
coef_df = coef_df.sort_values(
    by="odds_ratio",
    ascending=False
    )
print(coef_df.head(15))

# Saving the trained UCI model

joblib.dump(pipe_lr, "uci_heart_lr_model.pkl")

#-------------------- Part-2 -----------------------

# Load trained UCI model

model = joblib.load("uci_heart_lr_model.pkl")
print("UCI Heart Disease model loaded successfully.")


#Asking user(patient) details

print("\nEnter patient details:")

age = int(input("Age (years): "))
sex = int(input("Sex (1 = Male, 0 = Female): "))
cp = int(input("Chest pain type (0–3): "))
trestbps = float(input("Resting blood pressure (mmHg): "))
chol = float(input("Serum cholesterol (mg/dl): "))
fbs = int(input("Fasting blood sugar >120 mg/dl? (1 = Yes, 0 = No): "))
restecg = int(input("Resting ECG result (0–2): "))
thalach = float(input("Maximum heart rate achieved: "))
exang = int(input("Exercise induced angina? (1 = Yes, 0 = No): "))
oldpeak = float(input("ST depression (oldpeak): "))
slope = int(input("Slope of ST segment (0–2): "))
ca = int(input("Number of major vessels (0–3): "))
thal = int(input("Thalassemia (1 = normal, 2 = fixed defect, 3 = reversible defect): "))


# Convert inputs into DataFrame

uci_features = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak", "slope",
    "ca", "thal"
    ]

patient_df = pd.DataFrame([[
    age, sex, cp, trestbps, chol, fbs,
    restecg, thalach, exang, oldpeak,
    slope, ca, thal
    ]], columns = uci_features)


print("\n Patient data captured successfully.")
print(patient_df)

#--------------------- Part-3-----------------------------
#Predict Probability

risk_prob = model.predict_proba(patient_df)[0][1]
risk_percent = risk_prob * 100

print("\n Estimated probability of Heart Disease:")
print(f"{risk_percent:.1f}%")

# Probability categorization
if risk_percent < 10:
    risk_level = "LOW"

elif risk_percent < 20:
    risk_level = "MODERATE"
else:
    risk_level = "HIGH"


print(f"Risk Category: {risk_level}")

if risk_level == "MODERATE":
    print("This suggests some risk factors are present, but no strong high-risk signals.")

# Risk indicating factors

print("Factors increasing risk:")

risk_found = False

if age >= 45:
    print("• Age is a non-modifiable factor that gradually increases cardiac risk.")
    risk_found = True

if cp != 0:
    print("• Chest pain type is associated with higher cardiac risk.")
    risk_found = True

if trestbps >= 130:
    print("• Resting blood pressure is above the ideal range.")
    risk_found = True

if chol >= 200:
    print("• Elevated cholesterol increases heart disease risk.")
    risk_found = True

if exang == 1:
    print("• Exercise-induced angina suggests possible cardiac stress.")
    risk_found = True

if oldpeak >= 1.0:
    print("• ST-segment depression indicates cardiac workload stress.")
    risk_found = True

if ca > 0:
    print("• Presence of blocked major vessels increases risk.")
    risk_found = True

if not risk_found:
    print("• No strong high-risk indicators detected.")


# Protective factors

print("\nProtective factors:")

protective_found = False

if trestbps < 130:
    print("• Blood pressure is within a healthy range.")
    protective_found = True

if chol < 200:
    print("• Cholesterol level is within the recommended range.")
    protective_found = True

if exang == 0:
    print("• No exercise-induced angina reported.")
    protective_found = True

if thalach >= 150:
    print("• Good exercise capacity (higher maximum heart rate).")
    protective_found = True

if ca == 0:
    print("• No major blood vessel blockage detected.")
    protective_found = True

if not protective_found:
    print("• No strong protective factors identified.")


# Actionable recommendations

print("\nWhat you can do to reduce risk:")

print("• Maintain regular physical activity.")
print("• Follow a heart-healthy diet (low salt, low saturated fat).")

if trestbps >= 130:
    print("• Aim to reduce blood pressure below 120–130 mmHg.")

if chol >= 200:
    print("• Consider dietary changes to lower cholesterol.")

print("• Regular medical check-ups are recommended.")


# Clinical disclaimer

print("\nNote:")
print("This tool provides a model-based estimate only.")
print("It does NOT replace clinical evaluation or medical advice.")
print("The estimate is generated using a logistic regression model trained on clinical data.")









































































