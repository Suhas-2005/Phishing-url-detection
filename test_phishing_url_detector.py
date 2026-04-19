import unittest

from phishing_url_detector import PhishingURLDetector, extract_url_features


class TestPhishingURLDetector(unittest.TestCase):
    def test_extract_url_features_flags_suspicious_url(self):
        features = extract_url_features("http://verify-account-update.example.com/login")
        self.assertEqual(len(features), 9)
        self.assertEqual(features[6], 1.0)

    def test_train_and_predict(self):
        urls = [
            "https://docs.python.org/3/library/unittest.html",
            "https://www.wikipedia.org",
            "http://secure-login-update.example.net/verify",
            "http://bit.ly/reset-account",
        ]
        labels = [0, 0, 1, 1]

        detector = PhishingURLDetector.train(urls, labels)
        pred = detector.predict("http://verify-bank-account.example.com/login")
        self.assertIn(pred, [0, 1])

        batch = detector.predict_batch(["https://www.openai.com", "http://tinyurl.com/free-prize"])
        self.assertEqual(len(batch), 2)


if __name__ == "__main__":
    unittest.main()
