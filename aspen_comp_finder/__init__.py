#!/usr/bin/env python3
"""
Aspen Comp Finder
----------------
A modular system for finding comparable real estate properties in Aspen.
"""

import logging

# Set up package-level logging
logger = logging.getLogger("aspen_comp_finder")
logger.setLevel(logging.INFO)

# Add a handler if none exists
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

# Import key classes for easier access
from .classic_finder import ClassicCompFinder
from .data.data_loader import PropertyDataLoader
from .pricing.price_adjuster import PriceAdjuster
from .premium.premium_calculator import PremiumCalculator
from .scoring.scoring_engine import CompScoringEngine
from .filters.filters import PropertyFilter

# Version
__version__ = '0.1.0'