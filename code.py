import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# ---- Simulated basetable (replace with your CSV) ----
np.random.seed(42)
basetable = pd.DataFrame({
    "age": np.random.randint(20, 70, 100),
    "gender_F": np.random.randint(0, 2, 100),
    "time_since_last_gift": np.random.randint(1, 10, 100),
    "max_gift": np.random.randint(50, 200, 100),
    "mean_gift": np.random.randint(20, 100, 100),
    "min_gift": np.random.randint(5, 50, 100),
    "income_high": np.random.randint(0, 2, 100),
    "income_low": np.random.randint(0, 2, 100),
    "number_gift": np.random.randint(1, 10, 100),
    "time_since_first_gift": np.random.randint(1, 20, 100),
    "country_UK": np.random.randint(0, 2, 100),
    "country_India": np.random.randint(0, 2, 100),
    "target": np.random.randint(0, 2, 100)
})

# ----------- AUC Function -----------
def auc(variables, target, basetable):
    """Calculate AUC for given predictors"""
    X = basetable[variables]
    y = basetable[target].values.ravel()
    
    logreg = linear_model.LogisticRegression()
    logreg.fit(X, y)
    
    predictions = logreg.predict_proba(X)[:, 1]
    return roc_auc_score(y, predictions)

# ----------- Forward Stepwise Selection -----------
def next_best(current_variables, candidate_variables, target, basetable):
    """Finds the next best variable to add"""
    best_auc = -1
    best_variable = None
    
    for v in candidate_variables:
        auc_v = auc(current_variables + [v], target, basetable)
        if auc_v > best_auc:
            best_auc = auc_v
            best_variable = v
    return best_variable

# Example: AUC with some predictors
print("AUC with [max_gift, mean_gift, min_gift]:", round(
    auc(["max_gift", "mean_gift", "min_gift"], "target", basetable), 4))

# Forward selection example
candidate_variables = list(basetable.columns)
candidate_variables.remove("target")
current_variables = []

for i in range(5):
    next_var = next_best(current_variables, candidate_variables, "target", basetable)
    current_variables.append(next_var)
    candidate_variables.remove(next_var)
    print(f"Step {i+1}: added {next_var}")

print("Final selected variables:", current_variables)

# ----------- Train-Test Split Evaluation -----------
X = basetable.drop("target", axis=1)
y = basetable["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)

train = pd.concat([X_train, y_train], axis=1)
test = pd.concat([X_test, y_test], axis=1)

variables = ['max_gift','time_since_last_gift','number_gift',
             'mean_gift','income_high','age','gender_F',
             'time_since_first_gift','income_low','country_UK']

auc_values_train = []
auc_values_test = []
variables_evaluate = []

for v in variables:
    variables_evaluate.append(v)
    auc_values_train.append(auc(variables_evaluate, "target", train))
    auc_values_test.append(auc(variables_evaluate, "target", test))

# ----------- Plotting AUC ----------- 
x = np.arange(len(auc_values_train))
plt.xticks(x, variables, rotation=90)
plt.plot(x, auc_values_train, label="Train AUC")
plt.plot(x, auc_values_test, label="Test AUC")
plt.ylim((0.4, 1.0))
plt.legend()
plt.show()

& "C:\Users\Mallesh\OneDrive\Desktop\Files\Online LMS Project\venv\Scripts\Activate.ps1"
-m pip install pandas numpy scikit-learn matplotlib
