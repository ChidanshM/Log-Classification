# app/regex_classifier.py

import re
from typing import Tuple, Optional


class RegexClassifier:
    """
    Enhanced rule-based classifier with priority-ordered, highly specific patterns.
    Returns (label, confidence) or (None, 0.0) when no confident rule matches.
    
    Improvements:
    - More specific patterns with better coverage
    - Priority ordering (specific patterns first)
    - Multi-pattern support with confidence scoring
    - Context-aware matching
    """

    def __init__(self):
        # priority for rules (from high to low)
        self.rules = [
            
            # auth failure
            # High-priority auth failure patterns
            (re.compile(
                r"(?:failed login|login failed|authentication failed|"
                r"invalid credential|password (?:attempt )?failed|"
                r"MFA (?:verification )?failed|SSO login denied|"
                r"kerberos authentication failed|user.*blocked after failed|"
                r"token invalid|access denied)",
                re.I
            ), "authentication_failure", 1.0),
            
            # sucess authentication
            
            (re.compile(
                r"(?:success(?:ful(?:ly)?)? authenticat(?:ed|ion)|"
                r"authenticat(?:ed|ion) succeed(?:ed)?|"
                r"accepted password|login succeed(?:ed)?|"
                r"session established|token validated|"
                r"MFA (?:challenge )?passed|token (?:successfully )?verified)",
                re.I
            ), "authentication_success", 1.0),
            
            # security alert
            (re.compile(
                r"(?:security alert|unauthorized access|"
                r"privilege escalation|malware|"
                r"blocked by firewall|suspicious (?:token|login|behavior)|"
                r"anomalous login|untrusted (?:binary|script)|"
                r"certificate validation fail|"
                r"multiple failed.*login.*from|"
                r"port scan|intrusion|breach)",
                re.I
            ), "security_alert", 1.0),
            
            # Api failures
            (re.compile(
                r"(?:api (?:error|exception|crash(?:ed)?|timeout)|"
                r"(?:GET|POST|PUT|DELETE|PATCH).*(?:error|crash(?:ed)?|returned error|failed)|"
                r"gateway returned (?:50[0-9]|502|503|504)|"
                r"api responded? with (?:50[0-9]|40[0-9]|conflict)|"
                r"malformed api request|"
                r"unexpected api exception)",
                re.I
            ), "api_error", 0.95),
            
            # db errors
            (re.compile(
                r"(?:database (?:connection (?:refused|failed)|unreachable|error|migration failed)|"
                r"sql (?:error|syntax)|"
                r"(?:postgres|mysql|mongodb|redis).*(?:error|failed|refused|unreachable|dropped)|"
                r"relation.*does not exist|"
                r"(?:transaction|query) (?:deadlock|timeout|failed)|"
                r"duplicate (?:key|entry)|"
                r"integrity (?:failure|constraint))",
                re.I
            ), "database_error", 0.95),
            
            # resources
            (re.compile(
                r"(?:out of memory|oom(?:[ -]killed)?|"
                r"memory (?:allocation failure|exhaustion|critically low)|"
                r"thread pool (?:exhausted|saturation)|"
                r"cpu (?:starvation|starved)|"
                r"(?:load average|system load) exceeded|"
                r"(?:open )?file (?:descriptor|handle)(?:s)? (?:leak|exhausted?|limit exceeded)|"
                r"too many open file|"
                r"disk quota exceeded|"
                r"cache (?:eviction storm|thrashing)|"
                r"insufficient memory)",
                re.I
            ), "resource_exhaustion", 0.95),
            
            # fs errors
            (re.compile(
                r"(?:filesystem (?:error|inconsisten(?:cy|t))|"
                r"(?:disk quota|no space left|permission denied)|"
                r"file (?:not found|corruption|descriptor leak)|"
                r"(?:unable to|failed to) (?:write|read|delete|access)|"
                r"symbolic link loop|"
                r"directory.*not accessible)",
                re.I
            ), "filesystem_error", 0.95),
            
            #network errors
            (re.compile(
                r"(?:network (?:unreachable|timeout|congestion|error)|"
                r"connection (?:reset|refused|aborted)|"
                r"dns (?:resolution|lookup) failed|"
                r"packet loss exceeded|"
                r"(?:high )?(?:network )?latency|"
                r"ssl handshake failed|"
                r"tls handshake|"
                r"route.*unavailable|"
                r"gateway.*unreachable|"
                r"tcp (?:connection|session) (?:aborted|reset))",
                re.I
            ), "network_error", 0.95),
            
            # service timeout
            (re.compile(
                r"(?:(?:request|service|operation|connection|task|health probe) (?:timed out|timeout)|"
                r"timeout (?:contacting|accessing|after|reaching)|"
                r"exceeded timeout|"
                r"request stalled|"
                r"(?:timed? ?out|timeout).*(?:waiting|after|contacting))",
                re.I
            ), "service_timeout", 0.95),
            
            # conf errors
            (re.compile(
                r"(?:configuration (?:error|mismatch|missing)|"
                r"(?:missing|invalid|unknown|unsupported) (?:key|field|parameter|configuration)|"
                r"(?:invalid|malformed) (?:yaml|json|xml)|"
                r"(?:yaml|json) syntax|"
                r"incorrect permission.*config|"
                r"error parsing config|"
                r"(?:missing|invalid) environment variable)",
                re.I
            ), "configuration_error", 0.95),
            
            #api requests
            (re.compile(
                r"(?:incoming request|"
                r"(?:GET|POST|PUT|DELETE|PATCH|OPTIONS|HEAD) (?:/|\\w+/).*(?:initiated|received|from|processed|started|check executed))",
                re.I
            ), "api_request", 0.90),
        ]

    def predict(self, text: str) -> Tuple[Optional[str], float]:
        """
        Match against ordered rules and return first high-confidence match.
        
        Returns:
            (label, confidence) or (None, 0.0) if no match
        """
        for pattern, label, confidence in self.rules:
            if pattern.search(text):
                return label, confidence
        return None, 0.0
