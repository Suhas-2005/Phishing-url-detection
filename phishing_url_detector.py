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
SUSPICIOUS_KEYWORDS_FEATURE_INDEX = 6
MODEL_RANDOM_STATE = 42
SUBDOMAIN_THRESHOLD = 2
FEATURE_COUNT = 9


def _normalize_url(url: str) -> str:
    if re.match(r"^[a-zA-Z]+://", url):
        return url
    return f"http://{url}"


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
        model = GradientBoostingClassifier(random_state=MODEL_RANDOM_STATE)
        model.fit(features, labels)
        return cls(model=model)

    def predict(self, url: str) -> int:
        return int(self.model.predict([extract_url_features(url)])[0])

    def predict_batch(self, urls: Iterable[str]) -> List[int]:
        features = [extract_url_features(url) for url in urls]
        return [int(value) for value in self.model.predict(features)]


def extract_url_features(url: str) -> List[float]:
    parsed = urlparse(_normalize_url(url))
    hostname = (parsed.hostname or "").lower()
    path = (parsed.path or "").lower()
    query = (parsed.query or "").lower()
    full = f"{hostname}{path}{query}"

    features = [
        float(len(url)),
        float(sum(ch.isdigit() for ch in url)),
        float(url.count(".")),
        float(url.count("-")),
        float("@" in url),
        float(url.startswith("https://")),
        float(any(keyword in full for keyword in SUSPICIOUS_KEYWORDS)),
        float(hostname.count(".") > SUBDOMAIN_THRESHOLD),
        float(any(hostname.endswith(shortener) for shortener in SHORTENERS)),
    ]
    if len(features) != FEATURE_COUNT:
        raise RuntimeError(
            f"Feature count mismatch: expected {FEATURE_COUNT}, got {len(features)}"
        )
    return features


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
