"""
Data loading and preprocessing functionality for real estate analysis.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import datetime
from utils import logger, get_numeric_columns


class DataLoader:
    """Handles loading and preprocessing of real estate data."""

    @staticmethod
    def find_data_file(csv_path=None):
        """Find the data file from common locations or provided path."""
        if csv_path is None:
            # Try different possible locations for the file
            possible_paths = [
                "aspen_mvp_final_scored.csv",
                "../aspen_mvp_final_scored.csv",
                "../../aspen_mvp_final_scored.csv",
                "Data/mvp/aspen_mvp_final_scored.csv",
                "../Data/mvp/aspen_mvp_final_scored.csv",
                Path(__file__).parent / "Data/mvp/aspen_mvp_final_scored.csv",
                Path(__file__).parent.parent / "Data/mvp/aspen_mvp_final_scored.csv",
                "../../Python Analysis/Data/mvp/aspen_mvp_final_scored.csv",
                # Legacy paths
                "real_estate_with_condition_TEST.csv",
                "../real_estate_with_condition_TEST.csv",
                "../../real_estate_with_condition_TEST.csv",
            ]

            for path in possible_paths:
                try:
                    data = pd.read_csv(path)
                    logger.info(f"Found data file at: {path}")
                    return data, path
                except FileNotFoundError:
                    continue

            raise FileNotFoundError(
                "Could not find real estate data file in common locations."
            )
        else:
            data = pd.read_csv(csv_path)
            return data, csv_path

    @staticmethod
    def preprocess_data(data):
        """Preprocess the data by converting columns to appropriate types."""
        # Check for enriched dataset features
        has_demo_score = "demo_score" in data.columns
        has_condition_data = "improved_condition" in data.columns
        has_tdr_data = "tdr_eligible_flag_lookup" in data.columns

        if has_demo_score and has_condition_data and has_tdr_data:
            logger.info(
                "Using fully enriched dataset with condition, demo scores, and TDR data"
            )
        else:
            logger.info(
                "Using basic dataset - some advanced filtering options may not be available"
            )

        # Ensure key columns are numeric
        numeric_columns = get_numeric_columns()

        for col in numeric_columns:
            if col in data.columns:
                data[col] = pd.to_numeric(data[col], errors="coerce")

        return data

    @staticmethod
    def process_dates(data):
        """Process date columns and extract year information."""
        # Try to parse sold_date or listing_date
        date_cols = ["sold_date", "listing_date"]

        for col in date_cols:
            if col in data.columns:
                # Handle various date formats
                try:
                    data[f"{col}_dt"] = pd.to_datetime(data[col], errors="coerce")
                except:
                    # Handle special date formats (e.g., "7/20/23")
                    try:
                        # Extract date patterns like m/d/yy to datetime
                        data[f"{col}_dt"] = data[col].apply(DataLoader._parse_date_str)
                    except:
                        logger.warning(f"Could not parse {col} column to dates")

                # Extract year
                if f"{col}_dt" in data.columns:
                    data[f"{col}_year"] = data[f"{col}_dt"].dt.year

        # If no date columns found, create a dummy year column for price adjustment
        if not any(f"{col}_year" in data.columns for col in date_cols):
            logger.warning("No date columns found, using current year for all records")
            current_year = datetime.datetime.now().year
            data["transaction_year"] = current_year
        else:
            # Prefer sold_date if available, otherwise use listing_date
            if "sold_date_year" in data.columns:
                data["transaction_year"] = data["sold_date_year"]
            elif "listing_date_year" in data.columns:
                data["transaction_year"] = data["listing_date_year"]

        # Fill missing years with median year - avoiding inplace for future pandas compatibility
        if "transaction_year" in data.columns:
            median_year = data["transaction_year"].median()
            data["transaction_year"] = data["transaction_year"].fillna(median_year)

        return data

    @staticmethod
    def _parse_date_str(date_str):
        """Parse date strings in various formats."""
        if not isinstance(date_str, str):
            return pd.NaT

        # Try to match patterns like "7/20/23" or "7/20/2023"
        if "/" in date_str:
            parts = date_str.split("/")
            if len(parts) == 3:
                month, day, year = parts
                # Handle 2-digit years
                if len(year) == 2:
                    year = f"20{year}"  # Assume 20xx for 2-digit years
                try:
                    return pd.Timestamp(int(year), int(month), int(day))
                except:
                    return pd.NaT

        return pd.NaT
