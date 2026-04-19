# Phishing URL Detection System

A machine learning based system that detects whether a given URL is **Safe** or **Phishing** with a confidence score.

## About the Project

Phishing attacks trick users into visiting fake websites that look real.
This system analyzes the structure of a URL and predicts whether it is
malicious or safe using a trained Gradient Boosting classifier.

## How It Works

1. A dataset of labeled URLs (Safe / Phishing) is used for training
2. 13 structural features are extracted from each URL
3. A Gradient Boosting model is trained on these features
4. For any new URL entered by the user, features are extracted on the fly
5. The model predicts Safe or Phishing along with a confidence percentage

## Features Used

| Feature | Description |
|---|---|
| url_length | Total length of the URL |
| isHttps | Whether URL uses HTTPS |
| nb_dots | Number of dots in URL |
| nb_hyphens | Number of hyphens |
| at_symbol | Presence of @ symbol |
| sensitive_words_count | Count of words like login, verify, bank |
| valid_url | Whether URL has proper structure |
| path_length | Length of the URL path |
| nb_www | Presence of www |
| nb_com | Count of .com in URL |
| nb_and | Count of & symbols |
| nb_or | Count of | symbols |
| nb_underscore | Count of underscores |

## Technologies Used

- Python 3
- scikit-learn (Gradient Boosting Classifier)
- pandas
- numpy
- matplotlib
- joblib

## How to Run

1. Clone this repository
git clone https://github.com/Suhas-2005/Phishing-url-detection.git

2. Install required libraries
pip install pandas scikit-learn matplotlib numpy joblib

3. Run the script
python phishing_detector.py

4. Enter any URL when prompted
Enter a URL to check (or 'quit' to exit): https://google.com
URL        : https://google.com
Verdict    : Safe
Confidence : 82.0%

## Model Performance

| Metric | Value |
|---|---|
| Algorithm | Gradient Boosting Classifier |
| Accuracy | 78.23% |
| Safe URL recall | 91% |
| Phishing recall | 52% |

## Project Structure
Phishing-url-detection/
├── phishing_detector.py        # Main script
├── phishing_url_dataset.csv    # Training dataset
├── feature_importance.png      # Feature importance chart
└── README.md                   # Project documentation

## Sample Output
URL        : http://paypa1-login.com
Verdict    : Phishing
Confidence : 96.29%
URL        : https://leetcode.com/problemset/
Verdict    : Safe
Confidence : 86.0%

## Limitations

- Model accuracy is 78% — can be improved with a larger dataset
- Some legitimate URLs with suspicious patterns may be flagged
