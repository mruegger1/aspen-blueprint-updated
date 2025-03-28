"""
Price adjustment utilities for normalizing real estate prices across time periods.
"""

import datetime
from utils import logger


class PriceAdjuster:
    """Handles time-based price adjustments for real estate data."""

    @staticmethod
    def add_time_adjusted_prices(data):
        """
        Add time-adjusted price columns to normalize historical prices.
        Uses estimated annual appreciation rates for Aspen real estate.

        Args:
            data (DataFrame): The property data to adjust

        Returns:
            DataFrame: The data with adjusted price columns added
        """
        # Define annual appreciation rates (estimated)
        # These should be adjusted based on actual market data
        appreciation_rates = {
            2021: 0.18,  # 18% appreciation from 2021 to 2022
            2022: 0.12,  # 12% appreciation from 2022 to 2023
            2023: 0.09,  # 9% appreciation from 2023 to 2024
            2024: 0.06,  # 6% appreciation from 2024 to 2025
            2025: 0.00,  # Current year (no adjustment)
        }

        # Get current year
        current_year = datetime.datetime.now().year

        # Calculate cumulative adjustment factors
        adjustment_factors = {}
        cumulative = 1.0

        for year in range(min(appreciation_rates.keys()), current_year + 1):
            if year in appreciation_rates:
                if year < current_year:
                    cumulative *= 1 + appreciation_rates[year]
            adjustment_factors[year] = cumulative

        # Apply adjustment factors to prices if we have transaction years
        if "transaction_year" in data.columns:
            data["price_adjustment_factor"] = data["transaction_year"].apply(
                lambda y: adjustment_factors.get(int(y), 1.0) if not pd.isna(y) else 1.0
            )

            # Create adjusted price columns
            if "price_per_sqft" in data.columns:
                data["adjusted_price_per_sqft_time"] = (
                    data["price_per_sqft"] * data["price_adjustment_factor"]
                )

            # Log the adjustment process
            logger.info("=== Price Adjustment For Time ===")
            logger.info(
                "Applying the following appreciation rates to normalize historical prices:"
            )
            for year, rate in appreciation_rates.items():
                logger.info(f"{year}: {rate * 100:.1f}%")

        return data


# Add any missing imports
import pandas as pd
