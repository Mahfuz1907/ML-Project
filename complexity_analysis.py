import time
import tracemalloc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.datasets import make_classification
import os

os.makedirs('plots', exist_ok=True)

print("=" * 55)
print("  COMPLEXITY ANALYSIS — For Report Section 8.2")
print("=" * 55)

# TIME COMPLEXITY ANALYSIS
# Measure training time as dataset size grows
print("\nMeasuring training time vs. dataset size...")

# Simulate different dataset sizes
sizes = [500, 1000, 2000, 3000, 4000, 5000]
nb_times, lr_times, svm_times = [], [], []

# Generate synthetic TF-IDF-like data (sparse float matrix)
np.random.seed(42)

for n in sizes:
    # Synthetic sparse feature matrix simulating TF-IDF output
    X = np.random.rand(n, 5000) * np.random.binomial(1, 0.05, (n, 5000))
    y = np.random.randint(0, 2, n)

    # Naive Bayes — needs non-negative values (already ok)
    X_nb = np.abs(X)

    t0 = time.time()
    MultinomialNB().fit(X_nb, y)
    nb_times.append(time.time() - t0)

    t0 = time.time()
    LogisticRegression(max_iter=100).fit(X, y)
    lr_times.append(time.time() - t0)

    t0 = time.time()
    LinearSVC(max_iter=100).fit(X, y)
    svm_times.append(time.time() - t0)

    print(f"  n={n}: NB={nb_times[-1]*1000:.1f}ms | LR={lr_times[-1]*1000:.1f}ms | SVM={svm_times[-1]*1000:.1f}ms")

# Plot training time
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(sizes, [t*1000 for t in nb_times],  'o-', color='#185FA5', label='Naive Bayes',        lw=2)
ax.plot(sizes, [t*1000 for t in lr_times],  's-', color='#1D9E75', label='Logistic Regression', lw=2)
ax.plot(sizes, [t*1000 for t in svm_times], '^-', color='#D85A30', label='SVM (Linear)',        lw=2)
ax.set_xlabel('Training set size (samples)')
ax.set_ylabel('Training time (ms)')
ax.set_title('Time Complexity: Training Time vs. Dataset Size', fontsize=13, fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('plots/time_complexity.png', dpi=150)
plt.close()
print("\n  Saved: plots/time_complexity.png")

# MEMORY USAGE ANALYSIS
print("\nMeasuring memory usage...")

def measure_memory(model_class, X, y, **kwargs):
    tracemalloc.start()
    model = model_class(**kwargs)
    model.fit(X, y)
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak / 1024  # KB

n = 3000
X_test_mem = np.random.rand(n, 5000) * np.random.binomial(1, 0.05, (n, 5000))
y_test_mem = np.random.randint(0, 2, n)

nb_mem  = measure_memory(MultinomialNB,        np.abs(X_test_mem), y_test_mem)
lr_mem  = measure_memory(LogisticRegression,   X_test_mem,         y_test_mem, max_iter=100)
svm_mem = measure_memory(LinearSVC,            X_test_mem,         y_test_mem, max_iter=100)

print(f"  Naive Bayes:         {nb_mem:.0f} KB")
print(f"  Logistic Regression: {lr_mem:.0f} KB")
print(f"  SVM (Linear):        {svm_mem:.0f} KB")

# Plot memory
fig, ax = plt.subplots(figsize=(7, 4))
models_mem = ['Naive Bayes', 'Logistic\nRegression', 'SVM (Linear)']
mem_vals   = [nb_mem, lr_mem, svm_mem]
colors_mem = ['#185FA5', '#1D9E75', '#D85A30']
bars = ax.bar(models_mem, mem_vals, color=colors_mem, alpha=0.85, width=0.5)
for bar, val in zip(bars, mem_vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 30,
            f'{val:.0f} KB', ha='center', va='bottom', fontsize=10, fontweight='bold')
ax.set_ylabel('Peak memory usage (KB)')
ax.set_title('Space Complexity: Memory Usage Comparison', fontsize=13, fontweight='bold')
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('plots/memory_usage.png', dpi=150)
plt.close()
print("  Saved: plots/memory_usage.png")

# PRINT REPORT-READY COMPLEXITY TABLE
print("\n" + "=" * 55)
print("  COMPLEXITY SUMMARY — Copy into Report Section 8.2")
print("=" * 55)
print(f"""
  Model               Training Time  Space    Time Complexity
  -----------------------------------------------------------------
  Naive Bayes         Fastest        Lowest   O(N * d) — linear
  Logistic Regression Medium         Medium   O(N * d * iter)
  SVM (Linear)        Slower         Highest  O(N * d) to O(N² * d)

  N = number of training samples
  d = number of TF-IDF features (5000 in this project)

  KEY TRADE-OFF (P2 — Conflicting Requirements):
  - Naive Bayes: fastest but assumes word independence
  - SVM: more accurate on complex patterns but slower to train
  - Logistic Regression: best balance of speed & interpretability
""")
print("  Complexity analysis done!\n")