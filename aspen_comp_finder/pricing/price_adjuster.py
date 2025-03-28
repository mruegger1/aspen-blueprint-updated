#!/usr/bin/env python3
"""
Price Adjuster for Aspen Comp Finder
------------------------------------
Applies quarterly price adjustments to normalize property prices over time.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import logging
import json
import os

# Set up logging
logger = logging.getLogger("aspen_comp_finder.price_adjuster")

class PriceAdjuster:
    """Handles quarterly time-based price adjustments to account for market changes."""
    
    def __init__(self, quarterly_appreciation=None, config_path=None):
        """
        Initialize the price adjuster.
        
        Args:
            quarterly_appreciation (dict): Dictionary of quarterly appreciation rates
            config_path (str): Path to config file containing quarterly appreciation rates
        """
        self.quarterly_appreciation = quarterly_appreciation or {}
        
        # If no rates provided, try to load from config
        if not self.quarterly_appreciation and config_path:
            self._load_config(config_path)
        
        # If still no rates, use default hardcoded values
        if not self.quarterly_appreciation:
            self._set_default_rates()
    
    def _set_default_rates(self):
        """Set default quarterly appreciation rates."""
        self.quarterly_appreciation = {
            "2021Q1": 0.032,
            "2021Q2": 0.045,
            "2021Q3": 0.051,
            "2021Q4": 0.048,
            "2022Q1": 0.037,
            "2022Q2": 0.029,
            "2022Q3": 0.025,
            "2022Q4": 0.023,
            "2023Q1": 0.023,
            "2023Q2": 0.020,
            "2023Q3": 0.018,
            "2023Q4": 0.020,
            "2024Q1": 0.015,
            "2024Q2": 0.012,
            "2024Q3": 0.016,
            "2024Q4": 0.014,
            "2025Q1": 0.000  # Current quarter - no adjustment
        }
    
    def _load_config(self, config_path):
        """
        Load quarterly appreciation rates from a config file.
        
        Args:
            config_path (str): Path to JSON config file
        """
        try:
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    if 'quarterly_appreciation' in config:
                        self.quarterly_appreciation = config['quarterly_appreciation']
                        logger.info(f"Loaded quarterly appreciation rates from {config_path}")
        except Exception as e:
            logger.warning(f"Error loading quarterly appreciation rates from {config_path}: {str(e)}")
            # Fall back to default rates
            self._set_default_rates()
    
    def get_quarter_from_date(self, date):
        """
        Extract quarter identifier from a date (format: YYYYQN).
        
        Args:
            date (datetime): Date to extract quarter from
            
        Returns:
            str: Quarter identifier (e.g., "2023Q2")
        """
        year = date.year
        quarter = (date.month - 1) // 3 + 1
        return f"{year}Q{quarter}"
    
    def adjust_prices_for_time(self, df):
        """
        Apply quarterly time-based price adjustments to account for market appreciation/depreciation.
        
        Args:
            df (DataFrame): Property data
            
        Returns:
            DataFrame: Data with adjusted prices
        """
        logger.info("=== Price Adjustment For Time (Quarterly) ===")
        
        # Make a copy of the dataframe to avoid modifying the original
        df = df.copy()
        
        # Create adjusted price columns if they don't exist
        if 'adjusted_sold_price_time' not in df.columns:
            df['adjusted_sold_price_time'] = df['adjusted_sold_price'].copy()
        
        if 'adjusted_price_per_sqft_time' not in df.columns:
            # Make sure we have price_per_sqft column
            if 'price_per_sqft' not in df.columns and 'total_sqft' in df.columns:
                df['price_per_sqft'] = df['adjusted_sold_price'] / df['total_sqft']
            
            df['adjusted_price_per_sqft_time'] = df['price_per_sqft'].copy()
        
        # Get current quarter
        current_date = datetime.now()
        current_quarter = self.get_quarter_from_date(current_date)
        
        logger.info(f"Current quarter: {current_quarter}")
        logger.info("Applying quarterly appreciation rates to normalize prices")
        
        # Process each row
        for idx, row in df.iterrows():
            if pd.notna(row.get('sold_date_dt')):
                sale_date = row['sold_date_dt']
                sale_quarter = self.get_quarter_from_date(sale_date)
                
                # Skip adjustment if already in current quarter
                if sale_quarter == current_quarter:
                    continue
                
                # Calculate cumulative adjustment factor
                adjustment_factor = self._calculate_adjustment_factor(sale_date, current_quarter)
                
                # Apply adjustment
                df.at[idx, 'adjusted_sold_price_time'] = row['adjusted_sold_price'] * adjustment_factor
                if 'price_per_sqft' in df.columns and pd.notna(row['price_per_sqft']):
                    df.at[idx, 'adjusted_price_per_sqft_time'] = row['price_per_sqft'] * adjustment_factor
                
                # Add quarter info for reference
                df.at[idx, 'sale_quarter'] = sale_quarter
                df.at[idx, 'quarters_adjustment'] = adjustment_factor
        
        return df
    
    def _calculate_adjustment_factor(self, sale_date, current_quarter):
        """
        Calculate the cumulative adjustment factor between sale date and current quarter.
        
        Args:
            sale_date (datetime): Date of sale
            current_quarter (str): Current quarter identifier (YYYYQN)
            
        Returns:
            float: Cumulative adjustment factor
        """
        # Initialize adjustment factor
        adjustment_factor = 1.0
        
        # Get all quarters between sale quarter and current quarter
        quarters = []
        year = sale_date.year
        q = (sale_date.month - 1) // 3 + 1
        
        while f"{year}Q{q}" != current_quarter:
            q += 1
            if q > 4:
                q = 1
                year += 1
            quarterly_key = f"{year}Q{q}"
            quarters.append(quarterly_key)
        
        # Apply appreciation for each quarter
        for q in quarters:
            if q in self.quarterly_appreciation:
                adjustment_factor *= (1 + self.quarterly_appreciation[q])
            else:
                # If quarter not found, use default appreciation
                adjustment_factor *= 1.01  # 1% default quarterly appreciation
        
        return adjustment_factor