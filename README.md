# 🔍 Sentiment Analysis Project

![GitHub](https://img.shields.io/github/license/AminePro7/sentiment-analysis)
![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![Flask](https://img.shields.io/badge/Flask-2.x-green.svg)

A comprehensive sentiment analysis tool that uses both traditional Machine Learning algorithms and Recurrent Neural Networks to classify text sentiment.

## 📋 Overview

This project implements sentiment analysis using multiple approaches:
- **Traditional ML Models**: Logistic Regression, SVM, Naive Bayes
- **Deep Learning**: Recurrent Neural Networks (RNN)
- **Web Interface**: Flask-based application for real-time sentiment prediction

## ✨ Features

- 📊 Multiple model comparison (ML vs Deep Learning)
- 🔄 Real-time sentiment prediction through web interface
- 📝 Text preprocessing pipeline for Twitter data
- 💾 Pre-trained models for immediate use
- 📈 Detailed analysis and visualization of results

## 🛠️ Technologies Used

- **Python**: Core programming language
- **Flask**: Web framework
- **TensorFlow/Keras**: Deep learning models
- **NLTK**: Natural Language Processing
- **scikit-learn**: Machine learning algorithms
- **Pandas/NumPy**: Data manipulation
- **Matplotlib/Seaborn**: Data visualization

## 🚀 Installation

```bash
# Clone the repository
git clone https://github.com/AminePro7/sentiment-analysis.git
cd sentiment-analysis

# Install dependencies
pip install -r requirements.txt

# Run the application
python app.py
```

## 📊 Dataset

The project uses the Twitter sentiment analysis dataset containing 1.6 million tweets labeled as positive or negative.

## 🖥️ Usage

1. Start the Flask application:
   ```bash
   python app.py
   ```

2. Open your browser and navigate to `http://localhost:5000`

3. Enter text in the input field and click "Analyze" to get sentiment predictions from all models

## 📁 Project Structure

```
sentiment-analysis/
├── app.py                 # Flask web application
├── SA.py                  # Sentiment analysis implementation
├── templates/             # HTML templates
│   └── index.html         # Main web interface
├── static/                # CSS and JavaScript files
│   └── style.css          # Styling for web interface
├── results/               # Saved models
└── sentiment-analysis-ml-rnn.ipynb  # Jupyter notebook with model training
```

## 🔍 Model Performance

The project compares the performance of different models:

| Model | Accuracy | F1 Score |
|-------|----------|----------|
| Logistic Regression | ~84% | ~0.84 |
| SVM | ~83% | ~0.83 |
| Naive Bayes | ~82% | ~0.82 |
| RNN | ~86% | ~0.86 |

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👤 Author

- GitHub: [@AminePro7](https://github.com/AminePro7)

## 🙏 Acknowledgements

- Twitter sentiment analysis dataset
- NLTK and scikit-learn communities
- TensorFlow and Keras documentation 