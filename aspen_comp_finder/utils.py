"""
Utility functions for the Aspen Comparative Property Finder package.
"""

import logging
import re
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("comp_finder")


def clean_filename(filename):
    """Create a clean, safe filename from potentially unsafe string."""
    if not isinstance(filename, str):
        return "unknown"

    # Replace any non-alphanumeric characters with underscores
    safe_name = re.sub(r"[^\w\s]", "_", filename)
    # Replace spaces with underscores
    safe_name = re.sub(r"\s+", "_", safe_name)
    return safe_name


def get_numeric_columns():
    """Return standard numeric columns for real estate data."""
    return [
        "price_per_sqft",
        "walk_time_to_gondola_min",
        "walk_time_to_mill_st_min",
        "total_sqft",
        "bedrooms",
        "total_baths",
        "lot_sqft",
        "year_built",
        "demo_score",
        "condition_confidence",
        "lot_buildout_gap",
    ]


def safe_get(data, key, default="N/A"):
    """Safely get a value from a Series or dict with a default fallback."""
    if isinstance(data, pd.Series):
        return data.get(key, default)
    elif isinstance(data, dict):
        return data.get(key, default)
    return default


def format_currency(value, decimals=2):
    """Format a number as currency."""
    if isinstance(value, (int, float)):
        return f"${value:,.{decimals}f}"
    return str(value)


def ensure_directory_exists(directory):
    """Ensure a directory exists, create it if it doesn't."""
    import os

    if not os.path.exists(directory):
        os.makedirs(directory)
        logger.info(f"Created directory: {directory}")
    return directory
