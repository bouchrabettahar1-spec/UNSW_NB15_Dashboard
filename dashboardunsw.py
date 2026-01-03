import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, silhouette_score

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA


st.set_page_config(page_title="UNSW-NB15 Dashboard", layout="centered")
st.title(" UNSW-NB15 ML Dashboard")



@st.cache_data
def load_data(sample_size=5000):
    df = pd.read_csv(r"C:\Users\bouchra\Desktop\ML_TP\UNSW_NB15_testing-set.csv")
    df.dropna(inplace=True)

    if len(df) > sample_size:
        df = df.sample(sample_size, random_state=42)

    categorical_cols = ["proto", "service", "state", "attack_cat"]
    for col in categorical_cols:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))

    X = df.select_dtypes(include=["int64","float64"]).drop(columns=["label"])
    y = df["label"]

    X = StandardScaler().fit_transform(X)
    return train_test_split(X, y, test_size=0.2, random_state=42)

X_train, X_test, y_train, y_test = load_data()

supervised_models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(max_depth=10),
    "Random Forest": RandomForestClassifier(n_estimators=100),
    "SVM": SVC(probability=True),
    "KNN": KNeighborsClassifier(),
    "Naive Bayes": GaussianNB(),
    "Gradient Boosting": GradientBoostingClassifier()
}


tab1, tab2 = st.tabs(["📊 Supervised Learning", "🧠 Unsupervised Learning"])

with tab1:
    st.subheader("Supervised Classification")
    model_name = st.selectbox("Select Algorithm", supervised_models.keys())

    if st.button("Run Supervised Model"):
        with st.spinner("Training model..."):
            model = supervised_models[model_name]
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

        st.text("Classification Report")
        st.text(classification_report(y_test, y_pred))

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(4,4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Greys", ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)

        # ROC Curve
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:,1]
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            roc_auc = auc(fpr, tpr)

            fig, ax = plt.subplots(figsize=(4,4))
            ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
            ax.plot([0,1],[0,1],'--')
            ax.legend()
            st.pyplot(fig)


with tab2:
    st.subheader("Unsupervised Clustering")
    algo = st.selectbox("Select Clustering Algorithm", ["KMeans", "Hierarchical", "DBSCAN"])

    if st.button("Run Clustering"):
        with st.spinner("Running clustering..."):
          
            subset_size = min(5000, X_train.shape[0])
            subset = X_train[np.random.choice(X_train.shape[0], subset_size, replace=False)]

           
            X_pca = PCA(n_components=2).fit_transform(subset)

            if algo == "KMeans":
                labels = KMeans(n_clusters=3, n_init=10, random_state=42).fit_predict(subset)
            elif algo == "Hierarchical":
                labels = AgglomerativeClustering(n_clusters=3).fit_predict(subset)
            else: 
                labels = DBSCAN(eps=1.5, min_samples=10).fit_predict(subset)

    
            try:
                score = silhouette_score(subset, labels)
                st.write(f"Silhouette Score: {score:.2f}")
            except:
                st.write("Silhouette Score not defined (some clusters may be singletons)")

      
            fig, ax = plt.subplots(figsize=(5,4))
            sns.scatterplot(
                x=X_pca[:,0],
                y=X_pca[:,1],
                hue=labels,
                palette="muted",
                legend=False,
                ax=ax
            )
            ax.set_title(f"{algo} Clustering (PCA projection)")
            st.pyplot(fig)