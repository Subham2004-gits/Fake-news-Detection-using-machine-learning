📰 Fake News Detection Using Machine Learning
📌 Project Overview

This project focuses on building a machine learning-based Fake News Detection system that classifies news articles as Real or Fake using Natural Language Processing (NLP) techniques and multiple machine learning models.

The system:

Cleans and preprocesses news text
Extracts features using TF-IDF and Word2Vec
Trains multiple classifiers
Compares their performance using Accuracy, Precision, Recall, and F1-score
Selects the best-performing model

🎯 Objective

To design a high-accuracy fake news classifier using:
Traditional Machine Learning algorithms
NLP preprocessing
Statistical feature extraction techniques

📂 Dataset

The project uses:
True.csv → Real news articles
Fake.csv → Fake news articles

Both datasets are:
Merged
Cleaned
Shuffled
Split into:
Training set
Test set
Validation set

🧠 NLP Pipeline
1. Text Preprocessing

Lowercasing
Removing punctuation & numbers
Stopword removal
Lemmatization
Tokenization

2. Feature Engineering

Two methods were used:
TF-IDF Vectorizer
Word2Vec Embeddings

🤖 Models Implemented

Each model was trained using both TF-IDF and Word2Vec features:
Logistic Regression
Random Forest Classifier
Support Vector Machine (SVM)
Naive Bayes

📊 Results (Test Dataset)
🔹 TF-IDF Features
| Model               | Accuracy   | Precision  | Recall     | F1 Score   |
| ------------------- | ---------- | ---------- | ---------- | ---------- |
| Logistic Regression | 99.14%     | 98.92%     | 99.29%     | 99.10%     |
| Random Forest       | 99.95%     | 99.97%     | 99.92%     | 99.95%     |
| SVM                 | 99.67%     | 99.55%     | 99.77%     | 99.66%     |
| Naive Bayes         | 93.31%     | 93.26%     | 92.67%     | 92.97%     |

🔹 Word2Vec Features
| Model               | Accuracy                 | Precision | Recall | F1 Score |
| ------------------- | ------------------------ | --------- | ------ | -------- |
| Logistic Regression | 97.13%                   | 96.46%    | 97.57% | 97.01%   |
| Random Forest       | ~98% (lower than TF-IDF) |           |        |          |
| Others              | Lower than TF-IDF        |           |        |          |

🧪 Validation Results
🔹 TF-IDF (Validation Set)
| Model               | Accuracy   | Precision  | Recall     | F1 Score   |
| ------------------- | ---------- | ---------- | ---------- | ---------- |
| Logistic Regression | 99.00%     | 98.65%     | 99.27%     | 98.96%     |
| Random Forest       | 99.95%     | 99.95%     | 99.95%     | 99.95%     |
| SVM                 | 99.65%     | 99.46%     | 99.81%     | 99.64%     |
| Naive Bayes         | 93.10%     | 92.84%     | 92.64%     | 92.74%     |

🏆 Best Model
✅ Random Forest + TF-IDF

Why?
Highest accuracy: ~99.95%
Best F1-score
Extremely low false positives and false negatives
Very stable on both test and validation sets
Handles feature importance and non-linearity very well

📈 Evaluation Techniques Used

Accuracy Score
Precision
Recall
F1 Score
Confusion Matrix
ROC Curve
Bar charts for comparison

⚙️ How to Run the Project

Install dependencies:
pip install pandas numpy scikit-learn nltk gensim matplotlib seaborn wordcloud


Download NLTK resources:
import nltk
nltk.download('stopwords')
nltk.download('wordnet')


Place dataset files:

True.csv
Fake.csv


Run:
jupyter notebook


Open:
SDP END SEM CODE.ipynb


Run all cells.
🚀 Future Improvements

Add:
LSTM / GRU
BERT / Transformers

Build:
Web interface (Flask / Spring Boot)

Add:
Real-time news URL prediction

Use:
Explainable AI (SHAP / LIME)

🏁 Conclusion
This project proves that classical ML + TF-IDF can outperform many deep models for structured text classification.
The Random Forest + TF-IDF model achieves near-perfect classification performance and is production-ready.
