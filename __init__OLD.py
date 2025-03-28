"""
Aspen Comparative Property Finder Package
-----------------------------------------
A package for finding and analyzing comparable real estate properties
in Aspen with enhanced analytics capabilities.
"""

__version__ = "1.0.0"

# Import main classes for easy access
from .comp_finder import EnhancedCompFinder
from .data_loader import DataLoader
from .address_matcher import AddressMatcher

# Convenience function for address-based analysis
from .comp_finder import run_comp_analysis_by_address
