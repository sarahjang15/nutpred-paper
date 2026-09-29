import logging

# Set up logging configuration for the package
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('nutpred.log')
    ]
)

__all__ = [
    "build_ingredient_features",
    "build_knowledge_guided_features",
    "cleaning",
    "constants",
    "metrics",
    "predict",
    "prepare",
    "preprocess",
    "viz",
]
__version__ = "0.0.0"

# Create a logger for the package
logger = logging.getLogger(__name__)
logger.info(f"nutpred package version {__version__} initialized")
