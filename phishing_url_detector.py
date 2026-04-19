from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, List, Sequence
from urllib.parse import urlparse

from sklearn.ensemble import GradientBoostingClassifier


SUSPICIOUS_KEYWORDS = ("login", "verify", "secure", "account", "update", "bank")
SHORTENERS = (
    "bit.ly",
    "tinyurl.com",
    "t.co",
    "goo.gl",
    "ow.ly",
    "buff.ly",
)


@dataclass
class PhishingURLDetector:
    model: GradientBoostingClassifier

    @classmethod
    def train(cls, urls: Sequence[str], labels: Sequence[int]) -> "PhishingURLDetector":
        if len(urls) != len(labels):
            raise ValueError("urls and labels must have equal length")
        if len(urls) < 2:
            raise ValueError("at least two samples are required for training")

        features = [extract_url_features(url) for url in urls]
        model = GradientBoostingClassifier(random_state=42)
        model.fit(features, labels)
        return cls(model=model)

    def predict(self, url: str) -> int:
        return int(self.model.predict([extract_url_features(url)])[0])

    def predict_batch(self, urls: Iterable[str]) -> List[int]:
        features = [extract_url_features(url) for url in urls]
        return [int(value) for value in self.model.predict(features)]


def extract_url_features(url: str) -> List[float]:
    parsed = urlparse(url if re.match(r"^[a-zA-Z]+://", url) else f"http://{url}")
    hostname = (parsed.hostname or "").lower()
    path = (parsed.path or "").lower()
    query = (parsed.query or "").lower()
    full = f"{hostname}{path}{query}"

    return [
        float(len(url)),
        float(sum(ch.isdigit() for ch in url)),
        float(url.count(".")),
        float(url.count("-")),
        float("@" in url),
        float(url.startswith("https://")),
        float(any(keyword in full for keyword in SUSPICIOUS_KEYWORDS)),
        float(hostname.count(".") > 2),
        float(any(hostname.endswith(shortener) for shortener in SHORTENERS)),
    ]


if __name__ == "__main__":
    training_urls = [
        "https://www.google.com/search?q=phishing",
        "https://github.com/login",
        "http://secure-login-account-update.example.com/verify",
        "http://bit.ly/2xYz9Ab",
    ]
    training_labels = [0, 0, 1, 1]

    detector = PhishingURLDetector.train(training_urls, training_labels)
    sample = "http://verify-account-security-update.example.net/login"
    prediction = detector.predict(sample)
    print(f"URL: {sample}")
    print("Prediction:", "phishing" if prediction == 1 else "legitimate")
