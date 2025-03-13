"""
Utility module for miscellaneous functions such as logging configuration.
"""

import logging

def setup_logging(level=logging.INFO):
    """
    Initialize and configure global logging settings for the analyzer.
    Call this once at startup to set the desired logging level and format.
    """
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
