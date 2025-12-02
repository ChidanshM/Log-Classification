import re
import numpy as np
from typing import List, Dict


class LogFeatureExtractor:
	"""
	This code picks out important parts of a log message so a machine-learning model can understand it better.

    It looks for:
    - Numbers (like IDs or error codes)
    - How serious the error is
    - Names of services or parts of the system
    - Network info (IP addresses, ports, links)
    - File paths
    - Time information
    - Special characters or patterns

	"""
	
	def __init__(self):
		# Common service or component names
		self.service_keywords = [
			'api', 'database', 'db', 'redis', 'mysql', 'postgres', 'mongodb',
			'auth', 'authentication', 'login', 'session', 'token',
			'network', 'dns', 'gateway', 'firewall', 'proxy',
			'filesystem', 'disk', 'file', 'directory',
			'cpu', 'memory', 'thread', 'process', 'worker',
			'config', 'configuration', 'service', 'cluster'
		]
		
		# HDFS-specific words
		self.hdfs_keywords = [
			'namesystem', 'blockmap', 'packetresponder', 'datanode', 'namenode',
			'dfs', 'hadoop', 'exception', 'stacktrace', 'replicat', 'terminat'
		]
		
		# Severity 
		self.severity_keywords = {
			'critical': ['critical', 'fatal', 'emergency', 'panic'],
			'error': ['error', 'err', 'failed', 'failure', 'exception', 'crash'],
			'warning': ['warn', 'warning', 'degraded', 'timeout'],
			'info': ['info', 'success', 'ok', 'accepted', 'completed']
		}
	
	def extract_features(self, text: str) -> np.ndarray:
		"""
		Extract all features from a log message and returns np array of features
		"""
		features = []
		
		
		features.append(len(text)) 
		features.append(len(text.split()))  
		
		
		numbers = re.findall(r'\d+', text)
		features.append(len(numbers))  
		#http error codes
		features.append(1 if any(int(n) >= 400 and int(n) < 600 for n in numbers if n.isdigit()) else 0)  
		
		#http status codes
		features.append(1 if re.search(r'\b(?:40[0-9]|50[0-9])\b', text) else 0)  
		#non - zeror exit codes 
		features.append(1 if re.search(r'\bexit code\s*:?\s*[1-9]', text, re.I) else 0)  
		
		# severity levels
		for severity_type, keywords in self.severity_keywords.items():
			has_severity = any(kw in text.lower() for kw in keywords)
			features.append(1 if has_severity else 0)
		
		# service or component 
		for keyword in self.service_keywords:
			features.append(1 if keyword in text.lower() else 0)
			
		# HDFS
		for keyword in self.hdfs_keywords:
			features.append(1 if keyword in text.lower() else 0)
			
		# HDFS specific patterns: block IDs, src/dest
		features.append(1 if re.search(r'blk_[-0-9]+', text) else 0) 
		features.append(1 if re.search(r'src:\s*/', text) else 0)  
		features.append(1 if re.search(r'dest:\s*/', text) else 0) 
		
		# network related features: IPs, ports, URLs
		features.append(1 if re.search(r'\b(?:\d{1,3}\.){3}\d{1,3}\b', text) else 0)  
		features.append(1 if re.search(r':\d{2,5}(?:\s|$)', text) else 0) 
		features.append(1 if re.search(r'https?://', text, re.I) else 0) 
		
		# file systems related errors: file paths and extensions
		features.append(1 if re.search(r'[/\\](?:[\w-]+[/\\])*[\w-]+', text) else 0)  
		features.append(1 if re.search(r'\.\w{2,4}(?:\s|$)', text) else 0) 
		
		# http methods
		http_methods = ['GET', 'POST', 'PUT', 'DELETE', 'PATCH', 'HEAD', 'OPTIONS']
		for method in http_methods:
			features.append(1 if method in text.upper() else 0)
		
		# special characters
		features.append(text.count(':'))  
		features.append(text.count('[')) 
		features.append(text.count('('))  
		features.append(text.count('{')) 
		features.append(text.count('"')) 
		features.append(text.count("'"))  
		
		# uppercase letter ratio
		if len(text) > 0:
			uppercase_ratio = sum(1 for c in text if c.isupper()) / len(text)
			features.append(uppercase_ratio)
		else:
			features.append(0.0)
		
		# auth specific
		features.append(1 if re.search(r'\b(?:user|login|password|credential|token|session)\b', text, re.I) else 0)
		
		# DB specific
		features.append(1 if re.search(r'\b(?:sql|query|transaction|table|schema|relation)\b', text, re.I) else 0)
		
		# resource specific 
		features.append(1 if re.search(r'\b(?:memory|cpu|disk|thread|process|worker)\b', text, re.I) else 0)
		
		return np.array(features, dtype=np.float32)
	
	def get_feature_names(self) -> List[str]:
		"""Returns the list of feature names so it's easy to understand what the model is using."""
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
		Extract features for multiple log messages.returns 2D np array
		"""
		
		return np.array([self.extract_features(text) for text in texts])
