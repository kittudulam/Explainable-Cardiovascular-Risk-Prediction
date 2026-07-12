# ------------------- Part-1-----------------------

import statsmodels.api as sm
import pandas as pd
import numpy as np
import joblib 

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

df = pd.read_csv("framingham.csv")

#print(df.head())
#$print(df.columns)

#Selecting only basic attributes

basic_features = [
    "male", "age", "sysBP", "currentSmoker","diabetes","BMI"]

# Seperate input(X) and output(y)

X = df[basic_features]
y = df["TenYearCHD"]

# Pipeline
model = Pipeline([
    ("imputer", SimpleImputer(strategy = "median")),
    ("scaler", StandardScaler()),
    ("lr", LogisticRegression(
        max_iter = 1000,
        solver = "lbfgs"
        ))
    ])



# Cross-validation 

cv = StratifiedKFold(
    n_splits = 5,
    shuffle = True,
    random_state = 42
    )

# Evaluating model using ROC_AUC

auc_scores = cross_val_score(
    model,
    X,
    y,
    cv=cv,
    scoring = "roc_auc"
    )
from scipy.stats import t

n = len(auc_scores)
mean_auc = auc_scores.mean()
std_auc = auc_scores.std(ddof=1)

t_value = t.ppf(0.975, df=n-1)

margin = t_value * std_auc / np.sqrt(n)

lower_ci = mean_auc - margin
upper_ci = mean_auc + margin

print(f"Mean ROC-AUC: {mean_auc:.3f}")
print(f"95% CI: ({lower_ci:.3f}, {upper_ci:.3f})")
#Train model on full data

model.fit(X, y)

# ---------- SHAP Explainability ----------

import shap

# Process the features using the fitted pipeline
X_imp = model.named_steps["imputer"].transform(X)
X_scaled = model.named_steps["scaler"].transform(X_imp)

# Create SHAP explainer for logistic regression
explainer = shap.LinearExplainer(
    model.named_steps["lr"],
    X_scaled
)

# Calculate SHAP values
shap_values = explainer(X_scaled)

# Create SHAP summary plot
shap.summary_plot(
    shap_values,
    X_scaled,
    feature_names=basic_features,
    show=False
)

plt.savefig(
    "SHAP_Framingham.png",
    dpi=300,
    bbox_inches='tight'
)

plt.close()
# Saving the trained model
joblib.dump(model, "framingham_basic_model.pkl")

# ---------- Calibration Analysis ----------

# Predict probabilities on full dataset
y_prob = model.predict_proba(X)[:,1]

# Brier score
brier = brier_score_loss(y, y_prob)

print("\nCalibration Analysis")
print(f"Brier Score: {brier:.3f}")

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
    label='Framingham Model'
)

plt.plot(
    [0,1],
    [0,1],
    '--',
    label='Perfect Calibration'
)

plt.xlabel('Predicted Probability')
plt.ylabel('Observed Probability')
plt.title('Calibration Curve')
plt.legend()

plt.show()

# ---------- Fairness Analysis ----------

# Predict probabilities
y_prob = model.predict_proba(X)[:,1]

# Male subgroup
male_mask = X["male"] == 1

male_auc = roc_auc_score(
    y[male_mask],
    y_prob[male_mask]
)

# Female subgroup
female_mask = X["male"] == 0

female_auc = roc_auc_score(
    y[female_mask],
    y_prob[female_mask]
)

print("\nFairness Analysis")
print(f"Male ROC-AUC   : {male_auc:.3f}")
print(f"Female ROC-AUC : {female_auc:.3f}")
print(f"Difference     : {abs(male_auc-female_auc):.3f}")

plt.figure(figsize=(6,4))

groups = ["Male", "Female"]
scores = [male_auc, female_auc]

plt.bar(groups, scores)

plt.ylim(0.5,1.0)

plt.ylabel("ROC-AUC")
plt.title("Fairness Analysis Across Sex")

for i,v in enumerate(scores):
    plt.text(i, v+0.01, f"{v:.3f}")

plt.show()
# ---------- Odds Ratio with 95% Confidence Intervals ----------

# Impute missing values
imputer = SimpleImputer(strategy="median")
X_imp = imputer.fit_transform(X)

# Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imp)

# Add intercept
X_sm = sm.add_constant(X_scaled)

# Fit logistic regression using statsmodels
sm_model = sm.Logit(y, X_sm)
result = sm_model.fit()

# Regression coefficients
coef = result.params

# Confidence intervals
conf = result.conf_int()
conf.columns = ["Lower", "Upper"]

# Odds ratios
or_values = np.exp(coef)
lower_or = np.exp(conf["Lower"])
upper_or = np.exp(conf["Upper"])
# p-values
p_values = result.pvalues

# Create table
coef_df = pd.DataFrame({
    "Feature": ["Intercept"] + basic_features,
    "Odds Ratio": or_values,
    "CI Lower": lower_or,
    "CI Upper": upper_or,
    "P Value": p_values
})
coef_df["Odds Ratio"] = coef_df["Odds Ratio"].round(3)
coef_df["CI Lower"] = coef_df["CI Lower"].round(3)
coef_df["CI Upper"] = coef_df["CI Upper"].round(3)
coef_df["P Value"] = coef_df["P Value"].round(4)

coef_df.index = range(len(coef_df))
coef_df.index = range(len(coef_df))
print("\nOdds Ratios with 95% Confidence Intervals")
print(coef_df)

#-------------------- Part-2 -----------------------------

# Load the trained model
model = joblib.load("framingham_basic_model.pkl")

# Ask the user for inputs

print("\nEnter patient details:")

male = int(input("Male ( 1=Male, 0=Female): "))
age= int(input("Age(years): "))
sysBP = float (input("Systolic BP (mmHg): "))
currentSmoker = int(input("Current Smoker? (1=Yes, 0=No): "))
diabetes = int(input("Diabetes?(1=Yes, 0=No): "))
BMI= float(input("BMI: "))

# Converting variables into ML model dataframe

basic_features = [
    "male",
    "age",
    "sysBP",
    "currentSmoker",
    "diabetes",
    "BMI"
    ]

person_df = pd.DataFrame([[
    male,
    age,
    sysBP,
    currentSmoker,
    diabetes,
    BMI
    ]], columns = basic_features)

# Predict Risk

risk_prob = model.predict_proba(person_df)[0][1]
risk_percent = risk_prob * 100

# Risk categorization

if risk_percent < 10:
    risk_level = "LOW"
elif risk_percent < 20:
    risk_level = "MEDIUM"

else:
    risk_level = "HIGH"


# output

print("\n----- CVD RISK RESULT ----------")
print(f"Estimated 10-year CVD Risk: {risk_percent:.1f}%")
print(f"Risk category: {risk_level}")  


#----------------- PART-3 --------------------

print("\n---------CVD RISK EXPLANATION -------- \n")

#Summary Sentence
print(f"Your estimated 10-year cardiovascular risk is {risk_percent: .1f}%.")
print(f"This places you in the {risk_level} category. \n")


#Explain why (based on the inputs)

print("Main contributors to your risk:")

if sysBP >= 130:
    print("• Blood Pressure is above the ideal range.")
if BMI >= 25:
    print("• Body weight is above the healthy range.")
if age >= 45:
    print("• Age related cardiovascular risk increase.")
if currentSmoker ==1:
    print("• Current smoking increases cardiovascular risk.")
if diabetes == 1:
    print ("• Diabetes increases cardiovascular risk.")

# Protective factors

print("\n Protective factors:")

if currentSmoker == 0:
    print ("• You are not a smoker.")
if diabetes == 0:
    print("• You do not have diabetes.")
if sysBP<130:
    print("• Blood pressure is within the recommended range.")
if BMI < 25:
    print("• Body weight is within the healthy range.")


#Actionable recommendations

print("\n What can you do to reduce your risk:")

if sysBP>= 130:
    print("• Aim to reduce systolic BP below 120 mmHg")

if BMI >= 25:
    print ("• Aim to reduce body weight to achieve a BMI below 25.")

if currentSmoker == 1:
    print("• Quitting smoking can significantly reduce risk.")

if diabetes == 1:
    print("• Tight blood sugar control is important.")

print("• Maintain a healthy diet and regular physical activity.")

#Close with reassurance

print("\nImproving these factors may help you move into a lower risk category.")
print("This tool is for awareness only and does not replace medical advice.")








































































    
























    

































    







