from typing import Dict, Any

try:
    from drain3 import TemplateMiner
    from drain3.file_persistence import FilePersistence
    DRAIN3_AVAILABLE = True
except ImportError:
    DRAIN3_AVAILABLE = False


class LogParser:
    """
    Wraps Drain3 log parser. If Drain3 is not installed, it simply returns raw logs.
    """

    def __init__(self):
        if DRAIN3_AVAILABLE:
            persistence = FilePersistence("drain3_state.bin")
            self.template_miner = TemplateMiner(persistence)
        else:
            self.template_miner = None

    def parse(self, raw_log: str) -> Dict[str, Any]:
        """
        Returns: {
            "message": raw_log,
            "template": extracted template (if available)
        }
        """
        if self.template_miner:
            result = self.template_miner.add_log_message(raw_log)
            template = result.get("template_mined")
        else:
            template = raw_log  # fallback

        return {
            "message": raw_log,
            "template": template
        }
