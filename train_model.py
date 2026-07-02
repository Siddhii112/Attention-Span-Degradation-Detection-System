import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# =========================
# LOAD DATA
# =========================
df = pd.read_csv("student_learning_interaction_dataset.csv")

# Clean columns
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

# =========================
# TARGET CREATION (FIXED)
# =========================
def categorize_attention(x):
    if x < 0.10:
        return "Low"
    elif x < 0.20:
        return "Medium"
    else:
        return "High"

df['attention_category'] = df['attention_score'].apply(categorize_attention)

# Encode target
le_target = LabelEncoder()
df['attention_category'] = le_target.fit_transform(df['attention_category'])

# DEBUG: Check mapping
print("Label Mapping:")
for i, label in enumerate(le_target.classes_):
    print(i, "->", label)

# =========================
# FEATURES
# =========================
features = [
    'time_spent_minutes',
    'pages_visited',
    'video_watched_percent',
    'click_events',
    'assignment_score'
]

X = df[features]
y = df['attention_category']

# =========================
# SCALING
# =========================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


#HeatMap
heatmap_data = df[features]
# Create correlation matrix
corr = heatmap_data.corr()

# Plot heatmap
fig, ax = plt.subplots(figsize=(6, 4))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
plt.show()
# =========================
# SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# =========================
# MODEL
# =========================
rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=10,
    class_weight='balanced',
    random_state=42
)

rf.fit(X_train, y_train)

# =========================
# EVALUATION
# =========================
y_pred = rf.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

print("\nClass Distribution:\n", df['attention_category'].value_counts())

# =========================
# SAVE MODELS
# =========================
os.makedirs("models", exist_ok=True)

joblib.dump(rf, "models/rf_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")
joblib.dump(le_target, "models/label_encoder.pkl")

print("\n✅ Models saved successfully!")


print("Classes:", le_target.classes_)


