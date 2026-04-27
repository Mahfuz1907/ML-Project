#  SPAM DETECTOR - MODEL TRAINING SCRIPT

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, confusion_matrix, classification_report,
                              roc_auc_score, roc_curve)
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download NLTK data
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)

# STEP 1: LOAD DATASET
print("=" * 55)
print("  SPAM DETECTOR - Training Pipeline")
print("=" * 55)

print("\n[1/6] Loading dataset...")

try:
    df = pd.read_csv('spam.csv', encoding='latin-1')
    df = df[['v1', 'v2']]
    df.columns = ['label', 'message']
except FileNotFoundError:
    print("\n  [!] spam.csv not found. Generating sample dataset for demo...")

print(f"  Dataset loaded: {len(df)} messages")
print(f"  Spam: {(df['label']=='spam').sum()} | Ham: {(df['label']=='ham').sum()}")

# STEP 2: PREPROCESSING
print("\n[2/6] Preprocessing text...")

stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def preprocess(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)      # remove URLs
    text = re.sub(r'\d+', '', text)                  # remove numbers
    text = re.sub(r'[^a-z\s]', '', text)             # remove punctuation
    tokens = text.split()
    tokens = [stemmer.stem(t) for t in tokens if t not in stop_words and len(t) > 2]
    return ' '.join(tokens)

df['cleaned'] = df['message'].apply(preprocess)
df['label_num'] = df['label'].map({'spam': 1, 'ham': 0})

print(f"  Preprocessing complete. Sample:")
print(f"  Original : {df['message'].iloc[0][:60]}...")
print(f"  Cleaned  : {df['cleaned'].iloc[0][:60]}...")

# STEP 3: FEATURE EXTRACTION (TF-IDF)
print("\n[3/6] Extracting TF-IDF features...")

X = df['cleaned']
y = df['label_num']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf  = vectorizer.transform(X_test)

print(f"  Train: {X_train_tfidf.shape[0]} | Test: {X_test_tfidf.shape[0]}")
print(f"  Vocabulary size: {X_train_tfidf.shape[1]} features")

# STEP 4: TRAIN THREE MODELS
print("\n[4/6] Training models...")

models = {
    'Naive Bayes':        MultinomialNB(alpha=0.1),
    'Logistic Regression': LogisticRegression(max_iter=1000, C=1.0),
    'SVM (Linear)':       LinearSVC(C=1.0, max_iter=1000)
}

results = {}
trained_models = {}

for name, model in models.items():
    model.fit(X_train_tfidf, y_train)
    y_pred = model.predict(X_test_tfidf)

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score(y_test, y_pred, zero_division=0)
    f1   = f1_score(y_test, y_pred, zero_division=0)

    results[name] = {'Accuracy': acc, 'Precision': prec, 'Recall': rec, 'F1-Score': f1}
    trained_models[name] = model

    print(f"  {name:25s} | Acc: {acc:.4f} | F1: {f1:.4f}")

# STEP 5: SAVE BEST MODEL + VECTORIZER
print("\n[5/6] Saving best model...")

best_model_name = max(results, key=lambda k: results[k]['F1-Score'])
best_model = trained_models[best_model_name]

print(f"  Best model: {best_model_name} (F1={results[best_model_name]['F1-Score']:.4f})")

with open('model.pkl', 'wb') as f:
    pickle.dump(best_model, f)

with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

print("  Saved: model.pkl and vectorizer.pkl")

# STEP 6: GENERATE REPORT FIGURES
print("\n[6/6] Generating evaluation plots...")

os.makedirs('plots', exist_ok=True)

# --- Figure 1: Model Comparison Bar Chart ---
fig, ax = plt.subplots(figsize=(10, 5))
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
x = np.arange(len(metrics))
width = 0.25
colors = ['#185FA5', '#1D9E75', '#D85A30']

for i, (name, scores) in enumerate(results.items()):
    vals = [scores[m] for m in metrics]
    bars = ax.bar(x + i * width, vals, width, label=name, color=colors[i], alpha=0.85)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{val:.2f}', ha='center', va='bottom', fontsize=8)

ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
ax.set_xticks(x + width)
ax.set_xticklabels(metrics)
ax.set_ylim(0, 1.1)
ax.set_ylabel('Score')
ax.legend()
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('plots/model_comparison.png', dpi=150)
plt.close()
print("  Saved: plots/model_comparison.png")

# --- Figure 2: Confusion Matrix (best model) ---
y_pred_best = trained_models[best_model_name].predict(X_test_tfidf)
cm = confusion_matrix(y_test, y_pred_best)
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'], ax=ax)
ax.set_title(f'Confusion Matrix — {best_model_name}', fontsize=13, fontweight='bold')
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
plt.tight_layout()
plt.savefig('plots/confusion_matrix.png', dpi=150)
plt.close()
print("  Saved: plots/confusion_matrix.png")

# --- Figure 3: ROC Curve (Naive Bayes has predict_proba) ---
nb_model = trained_models['Naive Bayes']
y_proba = nb_model.predict_proba(X_test_tfidf)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_proba)
auc = roc_auc_score(y_test, y_proba)

fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(fpr, tpr, color='#185FA5', lw=2, label=f'ROC Curve (AUC = {auc:.3f})')
ax.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=1)
ax.fill_between(fpr, tpr, alpha=0.1, color='#185FA5')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curve — Naive Bayes', fontsize=13, fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('plots/roc_curve.png', dpi=150)
plt.close()
print("  Saved: plots/roc_curve.png")

# --- Print Full Classification Report ---
print("\n" + "=" * 55)
print(f"  CLASSIFICATION REPORT — {best_model_name}")
print("=" * 55)
print(classification_report(y_test, y_pred_best, target_names=['Ham', 'Spam']))

print("\n  FINAL RESULTS TABLE")
print(f"  {'Model':<25} {'Accuracy':>9} {'Precision':>10} {'Recall':>8} {'F1':>8}")
print("  " + "-" * 63)
for name, s in results.items():
    print(f"  {name:<25} {s['Accuracy']:>9.4f} {s['Precision']:>10.4f} {s['Recall']:>8.4f} {s['F1-Score']:>8.4f}")

print("\n  Training complete! Run: streamlit run app.py\n")