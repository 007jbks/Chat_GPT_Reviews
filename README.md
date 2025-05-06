# 🧠 Binary Classification of Review Dataset using Logistic Regression and TF-IDF

This project performs binary classification of textual reviews (e.g., product or movie reviews) using **Logistic Regression** and **TF-IDF vectorization**. The goal is to classify each review as either **positive** or **negative** based on its content.


## 🧾 Dataset Format

The dataset should be a CSV file with the following columns:
- `review`: The text of the review.
- `label`: The sentiment label (0 for negative, 1 for positive).

Example:

| review                    | label |
|---------------------------|-------|
| "I loved the product!"    | 1     |
| "It was a waste of money" | 0     |

## 🧪 Model Workflow

1. **Load and Clean Data**
2. **Preprocess Reviews**
   - Lowercasing
   - Removing punctuation and stopwords (optional)
3. **Feature Extraction using TF-IDF**
4. **Train Logistic Regression Model**
5. **Evaluate with Accuracy, Precision, Recall, and F1-score**

## 📈 Evaluation Metrics

- **Accuracy**
- **Precision**
- **Recall**
- **F1 Score**
  
