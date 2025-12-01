# app/feature_extractor.py

import re
import numpy as np
from typing import List, Dict


class LogFeatureExtractor:
    """
    Extracts domain-specific features from log messages to enhance ML classification.
    
    Features extracted:
    - Numeric patterns (counts, IDs, error codes)
    - Error severity indicators
    - Service/component names
    - Network indicators (IPs, ports, URLs)
    - File system patterns
    - Temporal indicators
    - Special characters and structure
    """
    
    def __init__(self):
        # Common service/component keywords
        self.service_keywords = [
            'api', 'database', 'db', 'redis', 'mysql', 'postgres', 'mongodb',
            'auth', 'authentication', 'login', 'session', 'token',
            'network', 'dns', 'gateway', 'firewall', 'proxy',
            'filesystem', 'disk', 'file', 'directory',
            'cpu', 'memory', 'thread', 'process', 'worker',
            'config', 'configuration', 'service', 'cluster'
        ]
        
        # HDFS-specific keywords
        self.hdfs_keywords = [
            'namesystem', 'blockmap', 'packetresponder', 'datanode', 'namenode',
            'dfs', 'hadoop', 'exception', 'stacktrace', 'replicat', 'terminat'
        ]
        
        # Severity keywords
        self.severity_keywords = {
            'critical': ['critical', 'fatal', 'emergency', 'panic'],
            'error': ['error', 'err', 'failed', 'failure', 'exception', 'crash'],
            'warning': ['warn', 'warning', 'degraded', 'timeout'],
            'info': ['info', 'success', 'ok', 'accepted', 'completed']
        }
    
    def extract_features(self, text: str) -> np.ndarray:
        """
        Extract all features from a log message.
        
        Returns:
            numpy array of features (shape: (n_features,))
        """
        features = []
        
        # LENGTH FEATURES
        features.append(len(text))  # Total length
        features.append(len(text.split()))  # Word count
        
        # NUMERIC FEATURES
        numbers = re.findall(r'\d+', text)
        features.append(len(numbers))  # Count of numeric tokens
        features.append(1 if any(int(n) >= 400 and int(n) < 600 for n in numbers if n.isdigit()) else 0)  # HTTP error codes
        
        # ERROR CODE PATTERNS
        features.append(1 if re.search(r'\b(?:40[0-9]|50[0-9])\b', text) else 0)  # HTTP status codes
        features.append(1 if re.search(r'\bexit code\s*:?\s*[1-9]', text, re.I) else 0)  # Non-zero exit codes
        
        # SEVERITY INDICATORS
        for severity_type, keywords in self.severity_keywords.items():
            has_severity = any(kw in text.lower() for kw in keywords)
            features.append(1 if has_severity else 0)
        
        # SERVICE/COMPONENT INDICATORS
        for keyword in self.service_keywords:
            features.append(1 if keyword in text.lower() else 0)
            
        # HDFS INDICATORS
        for keyword in self.hdfs_keywords:
            features.append(1 if keyword in text.lower() else 0)
            
        # HDFS PATTERNS
        features.append(1 if re.search(r'blk_[-0-9]+', text) else 0) # Block ID
        features.append(1 if re.search(r'src:\s*/', text) else 0) # Source IP pattern
        features.append(1 if re.search(r'dest:\s*/', text) else 0) # Dest IP pattern
        
        # NETWORK INDICATORS
        features.append(1 if re.search(r'\b(?:\d{1,3}\.){3}\d{1,3}\b', text) else 0)  # IP address
        features.append(1 if re.search(r':\d{2,5}(?:\s|$)', text) else 0)  # Port number
        features.append(1 if re.search(r'https?://', text, re.I) else 0)  # URL
        
        # FILE SYSTEM INDICATORS
        features.append(1 if re.search(r'[/\\](?:[\w-]+[/\\])*[\w-]+', text) else 0)  # File paths
        features.append(1 if re.search(r'\.\w{2,4}(?:\s|$)', text) else 0)  # File extensions
        
        # HTTP METHOD INDICATORS
        http_methods = ['GET', 'POST', 'PUT', 'DELETE', 'PATCH', 'HEAD', 'OPTIONS']
        for method in http_methods:
            features.append(1 if method in text.upper() else 0)
        
        # SPECIAL CHARACTERS
        features.append(text.count(':'))  # Colons (structured logs)
        features.append(text.count('['))  # Brackets
        features.append(text.count('('))  # Parentheses
        features.append(text.count('{'))  # Curly braces
        features.append(text.count('"'))  # Quotes
        features.append(text.count("'"))  # Single quotes
        
        # UPPERCASE RATIO
        if len(text) > 0:
            uppercase_ratio = sum(1 for c in text if c.isupper()) / len(text)
            features.append(uppercase_ratio)
        else:
            features.append(0.0)
        
        # AUTHENTICATION-SPECIFIC
        features.append(1 if re.search(r'\b(?:user|login|password|credential|token|session)\b', text, re.I) else 0)
        
        # DATABASE-SPECIFIC
        features.append(1 if re.search(r'\b(?:sql|query|transaction|table|schema|relation)\b', text, re.I) else 0)
        
        # RESOURCE-SPECIFIC
        features.append(1 if re.search(r'\b(?:memory|cpu|disk|thread|process|worker)\b', text, re.I) else 0)
        
        return np.array(features, dtype=np.float32)
    
    def get_feature_names(self) -> List[str]:
        """Returns names of all features for interpretability."""
        names = [
            'text_length', 'word_count', 'numeric_count', 'has_http_error_code',
            'has_http_status', 'has_exit_code',
            'severity_critical', 'severity_error', 'severity_warning', 'severity_info'
        ]
        
        names.extend([f'service_{kw}' for kw in self.service_keywords])
        names.extend([f'hdfs_{kw}' for kw in self.hdfs_keywords])
        names.extend(['has_block_id', 'has_src_ip', 'has_dest_ip'])
        
        names.extend(['has_ip_address', 'has_port', 'has_url',
                     'has_file_path', 'has_file_extension'])
        
        names.extend([f'http_{method.lower()}' for method in 
                     ['GET', 'POST', 'PUT', 'DELETE', 'PATCH', 'HEAD', 'OPTIONS']])
        
        names.extend(['colon_count', 'bracket_count', 'paren_count', 
                     'brace_count', 'doublequote_count', 'singlequote_count',
                     'uppercase_ratio',
                     'auth_related', 'db_related', 'resource_related'])
        
        return names
    
    def extract_batch(self, texts: List[str]) -> np.ndarray:
        """
        Extract features for multiple log messages.
        
        Returns:
            numpy array of shape (n_samples, n_features)
        """
        return np.array([self.extract_features(text) for text in texts])
