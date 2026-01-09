import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.naive_bayes import GaussianNB
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# =========================
# 🎯 Streamlit Title
# =========================
st.title("Bayesian & Gaussian Models on UNSW-NB15")

# =========================
# 1️⃣ Load dataset
# =========================
df = pd.read_csv(r"C:\Users\bouchra\Desktop\ML_TP\UNSW_NB15_testing-set.csv")

st.subheader("Dataset Preview")
st.dataframe(df.head())

# =========================
# 2️⃣ Bayesian Network (Naive Bayes)
# =========================
st.header("1️⃣ Bayesian Network – Naive Bayes")

numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

X = df[numeric_cols[:5]]   # first 5 numeric features
y = df["label"]           # binary target (normal / attack)

X = X.fillna(X.mean())

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

nb = GaussianNB()
nb.fit(X_train, y_train)

y_pred = nb.predict(X_test)
acc = accuracy_score(y_test, y_pred)

st.success(f"Naive Bayes Accuracy: {acc:.4f}")

# =========================
# 3️⃣ Gaussian Process Regression
# =========================
st.header("2️⃣ Gaussian Process Regression (Sampled)")

st.info("⚠️ Gaussian Process is computationally expensive → using 300 samples")

sample_size = st.slider("Sample size for GP", 100, 500, 300)

sample_df = df.sample(n=sample_size, random_state=42)

X_gp = sample_df[[numeric_cols[0]]].values
y_gp = sample_df[numeric_cols[1]].values

kernel = ConstantKernel(1.0) * RBF(1.0)
gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True)

gp.fit(X_gp, y_gp)

X_test_gp = np.linspace(X_gp.min(), X_gp.max(), 100).reshape(-1, 1)
y_pred_gp, y_std = gp.predict(X_test_gp, return_std=True)

# =========================
# 4️⃣ Plot (Streamlit)
# =========================
fig, ax = plt.subplots(figsize=(8, 5))

ax.plot(X_gp, y_gp, "r.", label="Sampled data")
ax.plot(X_test_gp, y_pred_gp, "b-", label="GP prediction")
ax.fill_between(
    X_test_gp.ravel(),
    y_pred_gp - 1.96 * y_std,
    y_pred_gp + 1.96 * y_std,
    alpha=0.2
)

ax.set_xlabel(numeric_cols[0])
ax.set_ylabel(numeric_cols[1])
ax.set_title("Gaussian Process Regression")
ax.legend()

st.pyplot(fig)