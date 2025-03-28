#!/usr/bin/env python3
"""
Data Loader for Aspen Comp Finder
---------------------------------
Loads and preprocesses property data from CSV files.
"""

import os
import pandas as pd
import logging
from datetime import datetime

# Set up logging
logger = logging.getLogger("aspen_comp_finder.data_loader")

class PropertyDataLoader:
    """Handles loading and basic preprocessing of property data."""
    
    def __init__(self):
        """Initialize the property data loader."""
        pass
    
    def load_data(self, csv_path=None):
        """
        Load property data from CSV file.
        
        Args:
            csv_path (str): Path to CSV file
            
        Returns:
            tuple: (DataFrame of properties, path to file)
        """
        # Define potential CSV file locations
        potential_paths = []
        
        # Add the provided path if it exists
        if csv_path and os.path.exists(csv_path):
            potential_paths.append(csv_path)
        
        # Add default locations
        default_paths = [
            "aspen_mvp_final_scored.csv",
            "data/aspen_mvp_final_scored.csv",
            "../data/aspen_mvp_final_scored.csv",
            "../../data/aspen_mvp_final_scored.csv",
            os.path.expanduser("~/Desktop/Aspen Real Estate Analysis Master/Python Analysis/Data/mvp/aspen_mvp_final_scored.csv")
        ]
        
        for path in default_paths:
            if os.path.exists(path):
                potential_paths.append(path)
        
        # Try to load from the first available path
        if not potential_paths:
            raise FileNotFoundError("Could not find property data CSV file. Please provide a valid path.")
        
        # Load the first available file
        file_path = potential_paths[0]
        logger.info(f"Loading data from: {file_path}")
        
        try:
            data = pd.read_csv(file_path)
            logger.info(f"Loaded {len(data)} properties from CSV")
            
            # Ensure critical columns exist
            required_cols = ['bedrooms', 'total_baths', 'adjusted_sold_price', 'total_sqft', 'sold_date']
            missing_cols = [col for col in required_cols if col not in data.columns]
            
            if missing_cols:
                logger.warning(f"Missing required columns: {', '.join(missing_cols)}")
            
            # Convert date columns
            data = self._preprocess_dates(data)
            
            return data, file_path
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def _preprocess_dates(self, data):
        """
        Convert date columns to datetime objects.
        
        Args:
            data (DataFrame): Property data
            
        Returns:
            DataFrame: Data with processed date columns
        """
        date_columns = ['sold_date', 'list_date', 'close_date']
        for col in date_columns:
            if col in data.columns:
                data[f"{col}_dt"] = pd.to_datetime(data[col], errors="coerce")
        
        return data
    
    def extract_street_name(self, address):
        """
        Extract street name from a property address.
        
        Args:
            address (str): Property address
            
        Returns:
            str: Extracted street name or empty string if not found
        """
        import re
        
        if not address or not isinstance(address, str):
            return ""
        
        # Try to extract street name
        # Common patterns: "123 Main St", "123 N Main St"
        address = address.lower().strip()
        
        # Remove any unit numbers
        address = re.sub(r'\bunit\s+\w+\b', '', address)
        address = re.sub(r'\bapt\s+\w+\b', '', address)
        address = re.sub(r'\b#\s*\w+\b', '', address)
        
        # Extract parts
        parts = address.split()
        
        # Skip number and directions
        directions = ['n', 's', 'e', 'w', 'north', 'south', 'east', 'west']
        start_idx = 0
        
        # Skip the number
        if len(parts) > 0 and parts[0].isdigit():
            start_idx = 1
        
        # Skip direction if present
        if len(parts) > start_idx and parts[start_idx].lower() in directions:
            start_idx += 1
        
        # The next part should be the street name
        if len(parts) > start_idx:
            return parts[start_idx].lower()
        
        return ""