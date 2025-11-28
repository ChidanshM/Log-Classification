# app/regex_classifier.py

import re
from typing import Tuple, Optional


class RegexClassifier:
    """
    High-quality rule-based classifier for strong-signal log patterns.
    Returns (label, confidence) or (None, 0.0) when no confident rule matches.
    """

    def __init__(self):
        self.rules = [

            # ---------------- AUTH FAILURE ----------------
            (re.compile(r"failed login|invalid credential|authentication failed|password attempt failed", re.I),
             "authentication_failure"),

           
            # ---------------- AUTH SUCCESS ----------------
(
    re.compile(
        r"success(ful)? authentication|"
        r"authenticated successfully|"
        r"successfully authenticated|"
        r"accepted password|"
        r"login succeeded|"
        r"session established",
        re.I
    ),
    "authentication_success"
),


            # ---------------- NETWORK ERRORS ----------------
            (re.compile(r"connection reset|network unreachable|dns|packet loss|gateway|route", re.I),
             "network_error"),

            # ---------------- SERVICE TIMEOUTS ----------------
            (re.compile(r"timeout|timed out|exceeded.*timeout|request stalled", re.I),
             "service_timeout"),

            # ---------------- RESOURCE EXHAUSTION ----------------
            (re.compile(r"out of memory|oom|cpu starvation|load average exceeded|thread pool exhausted", re.I),
             "resource_exhaustion"),

            # ---------------- FILESYSTEM ERRORS ----------------
            (re.compile(r"disk quota|no space left|file not found|corruption|filesystem", re.I),
             "filesystem_error"),

            # ---------------- CONFIGURATION ERRORS ----------------
            (re.compile(r"configuration.*(missing|invalid|malformed)|unknown field|parameter.*missing", re.I),
             "configuration_error"),

            # ---------------- API REQUEST ----------------
            (re.compile(r"GET|POST|PUT|DELETE|PATCH|OPTIONS", re.I),
             "api_request"),

            # ---------------- API ERROR ----------------
            (re.compile(r"api error|500|error during api call|exception.*api", re.I),
             "api_error"),

            # ---------------- DATABASE ERROR ----------------
            (re.compile(r"database|sql|deadlock|relation.*does not exist|transaction.*fail", re.I),
             "database_error"),

            # ---------------- SECURITY ALERT ----------------
            (re.compile(r"unauthorized|security alert|violation|restricted|blocked by firewall|suspicious", re.I),
             "security_alert"),
        ]

    def predict(self, text: str) -> Tuple[Optional[str], float]:
        for pattern, label in self.rules:
            if pattern.search(text):
                return label, 1.0  # high-confidence rule match
        return None, 0.0
