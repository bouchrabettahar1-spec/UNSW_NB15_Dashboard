import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

st.title("Self-Training on UNSW-NB15 Dataset")

# 1️⃣ تحميل البيانات مباشرة من المسار المحلي
csv_path = r"C:\Users\bouchra\Desktop\ML_TP\UNSW_NB15_testing-set.csv"
df = pd.read_csv(csv_path)
st.write("Preview of Dataset:", df.head())

# 2️⃣ تجهيز البيانات
target_col = 'attack_cat'

# تحويل الأعمدة النصية إلى أرقام
for col in df.select_dtypes(include=['object']).columns:
    if col != target_col:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

X = df.drop(columns=[target_col])
y = LabelEncoder().fit_transform(df[target_col])

# 3️⃣ إنشاء بيانات غير معلّمة (50%)
rng = np.random.RandomState(42)
mask = rng.rand(len(y)) < 0.5
y_unlabeled = y.copy()
y_unlabeled[mask] = -1  # -1 تعني غير معلّم

# تقسيم البيانات إلى تدريب واختبار
X_train, X_test, y_train, y_test = train_test_split(X, y_unlabeled, test_size=0.3, random_state=42)
y_test_true = y[X_test.index]

# 4️⃣ إعداد النموذج وHyperparameters
base_clf = RandomForestClassifier(n_estimators=100, random_state=42)
self_training_model = base_clf

confidence_threshold = st.slider("Confidence Threshold", 0.80, 1.0, 0.99)
max_iterations = st.slider("Max Iterations", 1, 20, 10)

st.write(f"Starting Self-Training with threshold={confidence_threshold} and max_iterations={max_iterations}")

added_counts = []

# 5️⃣ حلقة Self-Training
for i in range(max_iterations):
    labeled_mask = y_train != -1
    if np.sum(labeled_mask) == 0:
        break
    self_training_model.fit(X_train[labeled_mask], y_train[labeled_mask])
    
    unlabeled_mask = y_train == -1
    if np.sum(unlabeled_mask) == 0:
        break
    
    probs = self_training_model.predict_proba(X_train[unlabeled_mask])
    preds = np.argmax(probs, axis=1)
    max_probs = np.max(probs, axis=1)
    
    confident_idx = np.where(max_probs >= confidence_threshold)[0]
    added_counts.append(len(confident_idx))
    
    if len(confident_idx) == 0:
        break
    
    y_train_indices = np.where(unlabeled_mask)[0][confident_idx]
    y_train[y_train_indices] = preds[confident_idx]
    
    st.write(f"Iteration {i+1}: Added {len(confident_idx)} pseudo-labeled samples")

# 6️⃣ التقييم
y_pred = self_training_model.predict(X_test)
accuracy = accuracy_score(y_test_true, y_pred)
st.success(f"Final Accuracy: {accuracy:.4f}")

st.subheader("Classification Report")
st.text(classification_report(y_test_true, y_pred))

st.subheader("Pseudo-labels Added per Iteration")
st.bar_chart(added_counts)