import logging
import logging.config

from pathlib import Path

__all__ = ["ConsensusClassifier", "LymphxpressClassifier"]

init_path = Path(__file__).resolve()

# avoid circular imports
from lymphxpress.clf import ConsensusClassifier, LymphxpressClassifier  # noqa: E402

logging.config.fileConfig(
    init_path.parent / "log.conf",
    disable_existing_loggers=False,
)
logging.getLogger().setLevel(logging.INFO)

__version__ = "0.1.0"
