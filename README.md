# Enhancing Social Media Sentiment Analysis with Emoji Embeddings

**Technologies:** Python, NLTK, scikit-learn

A machine-learning approach that leverages emoji semantic content to improve hate-speech detection in social media, particularly Twitter data.

---

## Table of Contents
1. [Overview](#overview)
2. [Key Features](#key-features)
3. [Dataset](#dataset)
4. [Methodology](#methodology)
5. [Results](#results)
6. [Installation](#installation)
7. [Usage](#usage)
8. [Future Work](#future-work)

---

## Overview
Social media text often contains emojis that convey significant emotional context, yet traditional sentiment analysis models frequently ignore or improperly process them. This project addresses this limitation by incorporating emoji embeddings into the text analysis pipeline, resulting in more accurate hate-speech detection.

## Key Features
- **Emoji Semantic Preservation:** Convert emojis to textual representations that preserve emotional context
- **Comprehensive Text Preprocessing:** Tokenization, lemmatization, and special character handling
- **Advanced Vectorization:** TF-IDF with n-gram features (1–3) to capture contextual relationships
- **Model Optimization:** GridSearchCV-tuned Logistic Regression classifier
- **High Accuracy:** 95% accuracy and 94% precision for hate-speech detection

## Dataset
- **Size:** 28,000+ tweets labeled for hate speech
- **Class distribution:** 26,558 non-hate (93.4%), 1,865 hate (6.6%)

## Methodology
### Data Preprocessing
- **Text Cleaning:** remove URLs, usernames, special characters; lowercase conversion
- **Tokenization & Lemmatization:** via NLTK
- **Emoji Processing:** convert emojis to textual aliases, preserving semantic content

### Feature Engineering
- Hashtag extraction and processing
- TF-IDF vectorization (n-gram range 1–3)
- Feature matrix with 360,000+ dimensions

### Model Development
- **Algorithm:** Logistic Regression
- **Hyperparameter Tuning:** GridSearchCV
- **Rationale:** effective for high-dimensional sparse text, interpretable, efficient

## Results
- **Accuracy:** 95%
- **Precision (hate speech):** 94%
- **F1-score (weighted):** 93%

**Confusion Matrix**
```
[[5291    6]
 [ 286  102]]
```

**Classification Report**
```
              precision    recall  f1-score   support

           0       0.95      1.00      0.97      5297
           1       0.94      0.26      0.41       388

    accuracy                           0.95      5685
   macro avg       0.95      0.63      0.69      5685
weighted avg       0.95      0.95      0.93      5685
```

## Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/emoji-sentiment-analysis.git
cd emoji-sentiment-analysis

# (Optional) Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage
```python
from model import EmojiSentimentAnalyzer

analyzer = EmojiSentimentAnalyzer()
print(analyzer.predict("Just got the best news ever! 🎉🤩 #LifeIsGood"))  # 0: Non-hate, 1: Hate
```

## Future Work
- Extend to multi-class sentiment classification
- Incorporate deep-learning models (e.g. BERT) with emoji embeddings
- Add sequence-based emoji analysis for complex emotional signals
- Develop a web API for real-time monitoring
- Explore cross-platform emoji interpretation differences

## Acknowledgements
- Twitter data used for research purposes only
- Thanks to the NLTK and scikit-learn communities for their excellent tools
