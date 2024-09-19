import logging
import json
from datetime import datetime

class StructuredLogger:
    def __init__(self, name):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(message)s'))
        self.logger.addHandler(handler)

    def log(self, level, message, **kwargs):
        log_data = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': level,
            'message': message,
            **kwargs
        }
        self.logger.log(getattr(logging, level), json.dumps(log_data))

    def info(self, message, **kwargs):
        self.log('INFO', message, **kwargs)

    def error(self, message, **kwargs):
        self.log('ERROR', message, **kwargs)

    def warning(self, message, **kwargs):
        self.log('WARNING', message, **kwargs)

logger = StructuredLogger(__name__)

def setup_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

def log_model_performance(model_name, metrics):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = {
        "timestamp": timestamp,
        "model_name": model_name,
        "metrics": metrics
    }
    logging.info(json.dumps(log_entry))

def analyze_performance_trend(model_name):
    with open('model_performance.log', 'r') as f:
        logs = f.readlines()
    
    performance_data = []
    for log in logs:
        entry = json.loads(log.split(' - ')[-1])
        if entry['model_name'] == model_name:
            performance_data.append(entry)
    
    # Analyze trend (e.g., plot performance over time)
    # This is a placeholder for more sophisticated trend analysis
    return performance_data