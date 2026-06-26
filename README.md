---
title: TNL6323-Project
emoji: 🦀
colorFrom: yellow
colorTo: blue
sdk: docker
pinned: false
short_description: TNL6323-Project
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference

<h1 align="center">TNL6323-Project</h1>

## 📋 Project Overview

**Subject:** TNL6323 Natural Language Processing

**Title:** Restaurant & Food Review Sentiment Analysis

**Category:** Food Review

**Description:** Sushi Restaurant & Food Review Sentiment

## 🛠️ Technologies Used

- **Language:** Python
- **Framework:** Flask
- **Machine Learning:** Scikit-learn
- **NLP:** TF-IDF, Logistic Regression, VADER Sentiment
- **Transformer Model:** CardiffNLP RoBERTa Sentiment Model
- **Frontend:** HTML, CSS, JavaScript
- **Data Processing:** Pandas, NumPy
- **Model Storage:** Joblib

## ✨ Key Features

- 🔍 Real-time restaurant review sentiment classification
- 😊 Emoji extraction and sentiment preservation
- 📊 Interactive sentiment analytics dashboard
- 🍣 Food review sentiment analysis
- 👨‍🍳 Aspect-based detection for:
  - Food
  - Service
  - Ambiance
  - Price
- 🤖 Dual-model prediction:
  - Classical Machine Learning Model
  - Transformer-Based Model
- 📈 Confidence score visualization
- 📋 Dataset insights and review statistics
- 🌐 Web-based user interface using Flask


## 📁 Project Structure

```text
TNL6323-Project/
│
├── app.py                     # Main Flask application
├── preprocess.py              # Text cleaning and preprocessing
├── features.py                # Custom VADER feature extraction
├── Dockerfile                 # Docker container configuration
│
├── data/
│   └── analytics.json         # Dashboard analytics data
│
├── models/
│   ├── sentiment_model.joblib # Trained sentiment model
│   └── metrics.json           # Model evaluation metrics
│
├── static/
│   ├── app.js                 # Frontend functionality
│   └── style.css              # Website styling
│
├── templates/
│   └── index.html             # Main web interface
│
└── requirements.txt           # Required Python packages
```

## 👥 Contributors

<table>
    <tbody>
        <tr>
            <td align="center" valign="top" width="25%">
                <a href="https://github.com/sekjs19" target="_blank">
                    <img src="https://avatars.githubusercontent.com/u/134927115?v=4" width="100px;" alt="sekjs19 Avatar"/><br />
                    <sub><b>sekjs19</b></sub>
                </a>
            </td>
            <td align="center" valign="top" width="25%">
                <a href="https://github.com/yanyee0814" target="_blank">
                    <img src="https://avatars.githubusercontent.com/u/208572964?v=4" width="100px;" alt="yanyee0814 Avatar"/><br />
                    <sub><b>yanyee0814</b></sub>
                </a>
            </td>
            <td align="center" valign="top" width="25%">
                <a href="https://github.com/Jacob7179" target="_blank">
                    <img src="https://avatars.githubusercontent.com/u/70430960?v=4" width="100px;" alt="Jacob7179 Avatar"/><br />
                    <sub><b>Jacob7179</b></sub>
                </a>
            </td>
            <td align="center" valign="top" width="25%">
                <a href="https://github.com/t1an-wei" target="_blank">
                    <img src="https://avatars.githubusercontent.com/u/242118479?v=4" width="100px;" alt="t1an-wei Avatar"/><br />
                    <sub><b>t1an-wei</b></sub>
                </a>
            </td>
        </tr>
    </tbody>
</table>