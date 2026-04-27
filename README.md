# Spam Detector — Machine Learning Project

## Step 1 — Download the Dataset

1. Go to: https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset
2. Download `spam.csv`
3. Place it in this folder (same folder as `train_model.py`)

---

## Step 2 — Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Step 3 — Train the Model

```bash
python train_model.py
```

This will:

- Load and preprocess the dataset
- Extract TF-IDF features
- Train Naive Bayes, Logistic Regression, and SVM
- Save the best model as `model.pkl`
- Generate evaluation plots in `plots/` folder

---

## Step 4 — Run the Web App

```bash
streamlit run app.py
```

Open http://localhost:8501 in your browser.

---

## Step 5 — Generate Complexity Analysis (for report Section 8.2)

```bash
python complexity_analysis.py
```

---

## Report Figures (auto-generated in plots/ folder)

| File                 | Used In Report Section            |
| -------------------- | --------------------------------- |
| model_comparison.png | Section 8.1 — Results             |
| confusion_matrix.png | Section 8.1 — Results             |
| roc_curve.png        | Section 8.1 — Results             |
| time_complexity.png  | Section 8.2 — Complexity Analysis |
| memory_usage.png     | Section 8.2 — Complexity Analysis |
