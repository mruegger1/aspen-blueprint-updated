"""
ClassicCompFinder: Main orchestrator for the classic comparable property finder.

This module integrates the various components of the aspen_comp_finder package
to provide a unified interface for finding comparable properties using the
"classic" algorithm.
"""

import os
import json
import logging
import pandas as pd
from pathlib import Path

# Import components from the specialized modules as classes
try:
    # First try the direct imports (if called directly)
    from aspen_comp_finder.data.data_loader import PropertyDataLoader
    from aspen_comp_finder.pricing.price_adjuster import PriceAdjuster
    from aspen_comp_finder.premium.premium_calculator import PremiumCalculator
    from aspen_comp_finder.filters.filters import PropertyFilter
    from aspen_comp_finder.scoring.scoring_engine import CompScoringEngine
except ImportError:
    # Fallback to relative imports (if imported as part of the package)
    from .data.data_loader import PropertyDataLoader
    from .pricing.price_adjuster import PriceAdjuster
    from .premium.premium_calculator import PremiumCalculator
    from .filters.filters import PropertyFilter
    from .scoring.scoring_engine import CompScoringEngine


class ClassicCompFinder:
    """
    ClassicCompFinder uses the classic algorithm to find comparable properties.
    
    It orchestrates the various components of the aspen_comp_finder package:
    - Loading and preprocessing data
    - Adjusting prices for time
    - Applying building and street premiums
    - Filtering properties based on criteria
    - Scoring properties based on similarity
    """
    
    def __init__(self, csv_path=None, weights=None, premiums=None):
        """
        Initialize the ClassicCompFinder with data and configuration.
        
        Args:
            csv_path (str): Path to the CSV file with property data.
            weights (dict): Scoring weights for different property attributes.
            premiums (dict): Premium factors for buildings, streets, and time.
        """
        self.logger = logging.getLogger(__name__)
        
        # Load configuration files if not provided
        if weights is None:
            weights_path = self._find_config_file("weights.json")
            if weights_path:
                try:
                    with open(weights_path, "r") as f:
                        weights = json.load(f)
                except Exception as e:
                    self.logger.warning(f"Could not load weights from {weights_path}: {e}")
                    weights = {}
            else:
                weights = {}
        
        if premiums is None:
            premiums_path = self._find_config_file("premiums.json")
            if premiums_path:
                try:
                    with open(premiums_path, "r") as f:
                        premiums = json.load(f)
                except Exception as e:
                    self.logger.warning(f"Could not load premiums from {premiums_path}: {e}")
                    premiums = {}
            else:
                premiums = {}
        
        # Initialize component classes
        self.data_loader = PropertyDataLoader()
        self.price_adjuster = PriceAdjuster()
        self.premium_calculator = PremiumCalculator()
        self.property_filter = PropertyFilter()
        self.scoring_engine = CompScoringEngine(weights=weights.get("scoring_weights", {}))
        
        # Load property data
        if csv_path:
            try:
                self.data, self.file_path = self.data_loader.load_data(csv_path)
                self.logger.info(f"Loaded {len(self.data)} properties from {csv_path}")
            except Exception as e:
                self.logger.error(f"Error loading data from {csv_path}: {e}")
                self.data = pd.DataFrame()
                self.file_path = None
        else:
            self.data = pd.DataFrame()
            self.file_path = None
        
        self.weights = weights
        self.premiums = premiums
    
    def _find_config_file(self, filename):
        """
        Look for a configuration file in common locations.
        
        Args:
            filename (str): Name of the configuration file.
            
        Returns:
            str: Path to the configuration file, or None if not found.
        """
        # Try to find the config file in common locations
        possible_paths = [
            # Current directory
            os.path.join(os.getcwd(), filename),
            # Config directory
            os.path.join(os.getcwd(), "config", filename),
            # Package directory
            os.path.join(os.path.dirname(__file__), "config", filename),
            # Project root
            os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "config", filename),
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                return path
        
        return None
    
    def find_classic_comps(self, bedrooms=None, bathrooms=None, property_type=None, 
                          area=None, str_eligible=None, min_comps=3, limit=5, 
                          relax_filters_if_needed=True):
        """
        Find comparable properties based on the given criteria.
        
        Args:
            bedrooms (int): Number of bedrooms.
            bathrooms (float): Number of bathrooms.
            property_type (str): Type of property (e.g., "Condo", "Single Family").
            area (str): Area in Aspen (e.g., "Core", "West End").
            str_eligible (bool): Whether the property is eligible for short-term rentals.
            min_comps (int): Minimum number of comps to find before relaxing filters.
            limit (int): Maximum number of comps to return.
            relax_filters_if_needed (bool): Whether to relax filters if not enough comps found.
            
        Returns:
            dict: Dictionary with comps, stats, and metadata.
        """
        if self.data.empty:
            self.logger.warning("No data available. Please provide a CSV path.")
            return {"comps": pd.DataFrame(), "stats": {}, "filter_relaxation_applied": []}
        
        # Create a criteria dictionary for all the provided filters
        criteria = {
            "bedrooms": bedrooms,
            "bathrooms": bathrooms,
            "property_type": property_type,
            "area": area,
            "str_eligible": str_eligible,
            "min_comps": min_comps,
            "limit": limit
        }
        
        # Remove None values
        criteria = {k: v for k, v in criteria.items() if v is not None}
        
        self.logger.info(f"Finding comps with criteria: {criteria}")
        
        # Apply time-based price adjustments
        df = self.price_adjuster.adjust_prices_for_time(self.data)
        
        # Apply building and street premiums
        df = self.premium_calculator.apply_building_premium(df)
        df = self.premium_calculator.apply_street_premium(df)
        
        # Apply filters
        filtered_df = self.property_filter.filter_basic_criteria(df, criteria)
        self.logger.info(f"Found {len(filtered_df)} properties after basic filtering")
        
        # Relax filters if needed
        relax_steps = []
        if relax_filters_if_needed and len(filtered_df) < min_comps:
            self.logger.info(f"Not enough comps found ({len(filtered_df)}). Relaxing filters...")
            filtered_df, relax_steps = self.property_filter.apply_progressive_relaxation(df, criteria, min_comps)
            self.logger.info(f"Found {len(filtered_df)} properties after relaxing filters")
        
        # Create subject property from criteria
        subject = {k: v for k, v in criteria.items() if k not in ["min_comps", "limit"]}
        
        # Score the comps
        scored_df = self.scoring_engine.score(subject, filtered_df)
        
        # Sort by match score and limit results
        top_comps = scored_df.sort_values("match_score", ascending=False).head(limit)
        
        # Add column aliases for compatibility with display code
        if len(top_comps) > 0:
            # Map 'total_baths' to 'bathrooms' for display
            if 'total_baths' in top_comps.columns and 'bathrooms' not in top_comps.columns:
                top_comps['bathrooms'] = top_comps['total_baths']
            
            # Map 'resolved_property_type' to 'property_type' for display
            if 'resolved_property_type' in top_comps.columns and 'property_type' not in top_comps.columns:
                top_comps['property_type'] = top_comps['resolved_property_type']
                
            # Map price columns if needed
            if 'adjusted_sold_price' in top_comps.columns and 'price' not in top_comps.columns:
                top_comps['price'] = top_comps['adjusted_sold_price']
                
            # Map date columns if needed
            if 'sold_date' in top_comps.columns and 'date' not in top_comps.columns:
                top_comps['date'] = top_comps['sold_date']
                
            # Ensure full_address exists
            if 'full_address' not in top_comps.columns and 'address' in top_comps.columns:
                top_comps['full_address'] = top_comps['address']
                
            # Ensure price_per_sqft exists
            if 'price_per_sqft' not in top_comps.columns and 'total_sqft' in top_comps.columns and 'price' in top_comps.columns:
                top_comps['price_per_sqft'] = top_comps['price'] / top_comps['total_sqft']
        
        # Calculate statistics with exact key names for display
        stats = {}
        if len(top_comps) > 0:
            # Ensure we have the price and price_per_sqft columns for stats
            if 'price' in top_comps.columns:
                stats['avg_price'] = top_comps['price'].mean()
                stats['min_price'] = top_comps['price'].min()
                stats['max_price'] = top_comps['price'].max()
            
            if 'price_per_sqft' in top_comps.columns:
                stats['avg_price_per_sqft'] = top_comps['price_per_sqft'].mean()
                stats['min_price_per_sqft'] = top_comps['price_per_sqft'].min()
                stats['max_price_per_sqft'] = top_comps['price_per_sqft'].max()
            
            if 'match_score' in top_comps.columns:
                stats['avg_match_score'] = top_comps['match_score'].mean()
                stats['min_match_score'] = top_comps['match_score'].min()
                stats['max_match_score'] = top_comps['match_score'].max()
        
        self.logger.info(f"Returning {len(top_comps)} comps")
        
        # Ensure the dataframe is ready for CSV export
        # Make sure all columns contain serializable values
        for col in top_comps.columns:
            if top_comps[col].dtype == 'object':
                top_comps[col] = top_comps[col].astype(str)
        
        return {
            "comps": top_comps,
            "stats": stats,
            "subject": subject,
            "filter_relaxation_applied": relax_steps
        }