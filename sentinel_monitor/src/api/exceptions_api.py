from sentinel_monitor.src.utils.logging_utils import setup_logger

logger = setup_logger(__name__)

class RedisConnectionError(Exception):
    """Exceção levantada quando há problemas de conexão com o Redis."""
    pass