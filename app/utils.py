import re

def preprocess_log(text: str) -> str:
    """
    Normalize log text by masking dynamic entities to improve ML generalization.
    """
    # Mask Block IDs (blk_...)
    text = re.sub(r'blk_[-0-9]+', 'BLOCK_ID', text)
    
    # Mask IP addresses
    text = re.sub(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', 'IP_ADDRESS', text)
    
    # Mask file paths (simple version)
    text = re.sub(r'/[a-zA-Z0-9_/.-]+', 'FILE_PATH', text)
    
    return text
