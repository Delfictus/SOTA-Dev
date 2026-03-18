import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

DATA_FILE = 'training_features.csv'
MODEL_OUT = 'prism4d_pocket_ranker.pkl'

try:
    df = pd.read_csv(DATA_FILE)
except FileNotFoundError:
    print(f"Error: {DATA_FILE} not found. Run build_ml_dataset.py first.")
    exit()

print(f"Loaded {len(df)} data points from {df['Target'].nunique()} targets")
print(f"  Hits: {df['Is_Hit'].sum()}, Decoys: {(df['Is_Hit'] == 0).sum()}")

# One-hot encode classification
df_model = pd.get_dummies(df, columns=['Classification'])

# All numeric columns except identifiers and labels
exclude = {'Target', 'Site_ID', 'DCC', 'Is_Hit'}
features = [c for c in df_model.columns if c not in exclude]

# Fill missing spatial features with -1 sentinel (RandomForest handles this well)
X = df_model[features].apply(pd.to_numeric, errors='coerce').fillna(-1)
y = df_model['Is_Hit']

print(f"  Features ({len(features)}): {features}")

model = RandomForestClassifier(
    n_estimators=300, max_depth=12, random_state=42,
    class_weight='balanced', min_samples_leaf=2
)
model.fit(X, y)

# Feature importance ranking
importances = sorted(zip(features, model.feature_importances_), key=lambda x: -x[1])
print(f"\nFeature importance:")
for feat, imp in importances:
    bar = "#" * int(imp * 50)
    print(f"  {feat:<25s} {imp:.3f} {bar}")

# Training accuracy (on training set — for sanity check only)
y_pred = model.predict(X)
print(f"\nTraining accuracy: {accuracy_score(y, y_pred):.1%}")
print(classification_report(y, y_pred, target_names=['Decoy', 'Hit'], zero_division=0))

with open(MODEL_OUT, 'wb') as f:
    pickle.dump({'model': model, 'features': features}, f)

print(f"Model saved: {MODEL_OUT}")
