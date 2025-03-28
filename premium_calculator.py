#!/usr/bin/env python3
"""
Premium Calculator for Aspen Comp Finder
---------------------------------------
Applies building and street premium adjustments to property data.
"""

import pandas as pd
import logging
import json
import os
from ..data.data_loader import PropertyDataLoader

# Set up logging
logger = logging.getLogger("aspen_comp_finder.premium_calculator")

class PremiumCalculator:
    """Handles building and street premium calculations for properties."""
    
    def __init__(self, building_premiums=None, street_premiums=None, config_path=None):
        """
        Initialize the premium calculator.
        
        Args:
            building_premiums (dict): Dictionary of building premiums
            street_premiums (dict): Dictionary of street premiums
            config_path (str): Path to config file containing premium data
        """
        self.building_premiums = building_premiums or {}
        self.street_premiums = street_premiums or {}
        self.data_loader = PropertyDataLoader()
        
        # If no premiums provided, try to load from config
        if (not self.building_premiums or not self.street_premiums) and config_path:
            self._load_config(config_path)
        
        # If still no premiums, use default hardcoded values
        if not self.building_premiums:
            self._set_default_building_premiums()
        
        if not self.street_premiums:
            self._set_default_street_premiums()
    
    def _set_default_building_premiums(self):
        """Set default building premium values."""
        self.building_premiums = {
            "Monarch on the park": 0.604,
            "DurCondo": 0.48,
            "MtQueen": 0.424,
            "DurMews": 0.343,
            "TwnSteAs": 0.267,
            "NorthNell": 0.172,
            "DerBerghof": 0.146,
            "Dolomite": 0.128,
            "Aspen Alps": 0.106,
            "Aspen Square": 0.085,
            "ObermeyerPlace": 0.068,
            "FifthAve": 0.063,
            "ShadowMtn": 0.052,
            "210Cooper": 0.04,
            "SouthPt": 0.029
        }
    
    def _set_default_street_premiums(self):
        """Set default street premium values."""
        self.street_premiums = {
            "galena": 0.271,
            "monarch": 0.186,
            "durant": 0.032,
            "mill": 0.025,
            "cooper": 0.02,
            "hyman": -0.016,
            "aspen": -0.059,
            "hopkins": -0.075,
            "main": -0.203
        }
    
    def _load_config(self, config_path):
        """
        Load premium data from a config file.
        
        Args:
            config_path (str): Path to JSON config file
        """
        try:
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    
                    if 'building_premiums' in config:
                        self.building_premiums = config['building_premiums']
                        logger.info(f"Loaded building premiums from {config_path}")
                    
                    if 'street_premiums' in config:
                        self.street_premiums = config['street_premiums']
                        logger.info(f"Loaded street premiums from {config_path}")
        except Exception as e:
            logger.warning(f"Error loading premiums from {config_path}: {str(e)}")
            # Fall back to default values
            self._set_default_building_premiums()
            self._set_default_street_premiums()
    
    def apply_building_premium(self, df):
        """
        Apply building premium adjustments to property data.
        
        Args:
            df (DataFrame): Property data
            
        Returns:
            DataFrame: Data with building premium column
        """
        logger.info("=== Applying Building Premium Adjustments ===")
        
        # Make a copy of the dataframe to avoid modifying the original
        df = df.copy()
        
        # Create building premium column
        df['building_premium'] = 0.0
        
        # Check if sub_loc column exists
        if 'sub_loc' not in df.columns:
            logger.warning("sub_loc column not found, skipping building premium adjustments")
            return df
        
        # Map building premiums
        for idx, row in df.iterrows():
            if pd.notna(row['sub_loc']):
                sub_loc = row['sub_loc'].strip()
                
                # Try exact match first
                if sub_loc in self.building_premiums:
                    df.at[idx, 'building_premium'] = self.building_premiums[sub_loc]
                else:
                    # Try fuzzy match
                    for building, premium in self.building_premiums.items():
                        if building.lower() in sub_loc.lower() or sub_loc.lower() in building.lower():
                            df.at[idx, 'building_premium'] = premium
                            break
        
        # Log buildings found
        buildings_found = df[df['building_premium'] > 0]['sub_loc'].unique()
        logger.info(f"Applied building premiums to {len(buildings_found)} buildings")
        
        return df
    
    def apply_street_premium(self, df):
        """
        Apply street premium adjustments to property data.
        
        Args:
            df (DataFrame): Property data
            
        Returns:
            DataFrame: Data with street premium column
        """
        logger.info("=== Applying Street Premium Adjustments ===")
        
        # Make a copy of the dataframe to avoid modifying the original
        df = df.copy()
        
        # Create street premium column
        df['street_premium'] = 0.0
        
        # Extract street names from addresses
        address_col = None
        for col in ['full_address', 'address', 'street_address', 'location']:
            if col in df.columns:
                address_col = col
                break
        
        if not address_col:
            logger.warning("No address column found, skipping street premium adjustments")
            return df
        
        # Extract street names and apply premiums
        df['extracted_street'] = df[address_col].apply(self.data_loader.extract_street_name)
        
        # Apply street premiums
        for idx, row in df.iterrows():
            if pd.notna(row['extracted_street']) and row['extracted_street']:
                street = row['extracted_street'].lower()
                
                # Try exact match first
                if street in self.street_premiums:
                    df.at[idx, 'street_premium'] = self.street_premiums[street]
                else:
                    # Try fuzzy match
                    for premium_street, premium in self.street_premiums.items():
                        if premium_street in street or street in premium_street:
                            df.at[idx, 'street_premium'] = premium
                            break
        
        # Log streets found
        streets_found = df[df['street_premium'] != 0]['extracted_street'].unique()
        logger.info(f"Applied street premiums to {len(streets_found)} streets")
        
        return df