# ml_streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import Lasso
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

st.title("IDS Machine Learning Dashboard")

uploaded_file = st.file_uploader("Upload CSV Dataset", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("✅ Dataset Loaded")
    st.write(df.head())

    X = df.drop("label", axis=1)
    y = df["label"]

    # Feature Selection
    selector = VarianceThreshold(threshold=0.01)
    X_var = selector.fit_transform(X)
    cols_after_var = X.columns[selector.get_support()]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_var)

    imputer = SimpleImputer(strategy="mean")
    X_scaled_imputed = imputer.fit_transform(X_scaled)

    lasso = Lasso(alpha=0.01)
    lasso.fit(X_scaled_imputed, y)
    selected_mask = lasso.coef_ != 0
    selected_features = cols_after_var[selected_mask]
    X_selected = X_scaled_imputed[:, selected_mask]

    st.subheader("Selected Features")
    st.write(list(selected_features))

    # Feature Extraction Visualizations
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_selected)

    fa = FactorAnalysis(n_components=2)
    X_factors = fa.fit_transform(X_selected)

    st.subheader("PCA Visualization")
    fig, ax = plt.subplots()
    scatter = ax.scatter(X_pca[:,0], X_pca[:,1], c=y, cmap='coolwarm', alpha=0.6)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("PCA")
    fig.colorbar(scatter, ax=ax, label='Label')
    st.pyplot(fig)

    st.subheader("Factor Analysis Visualization")
    fig, ax = plt.subplots()
    scatter = ax.scatter(X_factors[:,0], X_factors[:,1], c=y, cmap='coolwarm', alpha=0.6)
    ax.set_xlabel("Factor1")
    ax.set_ylabel("Factor2")
    ax.set_title("Factor Analysis")
    fig.colorbar(scatter, ax=ax, label='Label')
    st.pyplot(fig)

    # Random Forest Training
    X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.3, random_state=42)
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)

    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

    st.subheader("Classification Report")
    st.text(classification_report(y_test, y_pred))

    # Feature Importances
    importances = rf.feature_importances_
    fi_df = pd.DataFrame({
        'Feature': selected_features,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)
    st.subheader("Feature Importances")
    st.table(fi_df)