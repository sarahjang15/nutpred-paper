"""nutpred: predicting calcium, fiber and iron of packaged foods from the
ingredient list and the macronutrient panel."""
import logging

# A library does not configure logging; run.py and the CLIs do.
logging.getLogger(__name__).addHandler(logging.NullHandler())

__version__ = "0.3.0"

__all__ = [
    "build_ingredient_features",
    "build_knowledge_guided_features",
    "cleaning",
    "constants",
    "metrics",
    "predict",
    "prepare",
    "preprocess",
    "split",
    "viz",
]
