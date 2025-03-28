"""
Analysis and reporting utilities for real estate property data.
"""

import pandas as pd
import datetime
from utils import logger, clean_filename, format_currency


class PropertyAnalyzer:
    """Handles the analysis of real estate properties and their comparables."""

    @staticmethod
    def analyze_comps(comp_df, time_adjust=True, price_col=None):
        """
        Analyze comparable properties and provide summary statistics.

        Args:
            comp_df (pd.DataFrame): DataFrame of comparable properties
            time_adjust (bool): Whether time adjustment is enabled
            price_col (str): Column to use for price analysis

        Returns:
            dict: Dictionary containing summary statistics
        """
        if comp_df.empty:
            return {"error": "No comparable properties found"}

        # Set default price column based on time adjustment
        if price_col is None:
            price_col = (
                "adjusted_price_per_sqft_time" if time_adjust else "price_per_sqft"
            )

        # Default to price_per_sqft if the specified column doesn't exist
        if price_col not in comp_df.columns:
            price_col = "price_per_sqft"

        # Calculate core statistics
        stats = {
            "count": len(comp_df),
            price_col: {
                "mean": comp_df[price_col].mean(),
                "median": comp_df[price_col].median(),
                "min": comp_df[price_col].min(),
                "max": comp_df[price_col].max(),
                "std": comp_df[price_col].std(),
            },
            "total_sqft": {
                "mean": comp_df["total_sqft"].mean(),
                "median": comp_df["total_sqft"].median(),
                "min": comp_df["total_sqft"].min(),
                "max": comp_df["total_sqft"].max(),
            },
        }

        # Add enhanced property condition statistics
        if "improved_condition" in comp_df.columns:
            stats["improved_condition_counts"] = (
                comp_df["improved_condition"].value_counts().to_dict()
            )
        elif "property_condition" in comp_df.columns:
            stats["condition_counts"] = (
                comp_df["property_condition"].value_counts().to_dict()
            )

        # Add condition confidence statistics
        if "condition_confidence" in comp_df.columns:
            stats["condition_confidence"] = {
                "mean": comp_df["condition_confidence"].mean(),
                "counts": comp_df["condition_confidence"].value_counts().to_dict(),
            }

        # Add demolition score statistics
        if "demo_score" in comp_df.columns:
            stats["demo_score"] = {
                "mean": comp_df["demo_score"].mean(),
                "median": comp_df["demo_score"].median(),
                "min": comp_df["demo_score"].min(),
                "max": comp_df["demo_score"].max(),
            }
            stats["value_add_counts"] = (
                comp_df["value_add_type"].value_counts().to_dict()
            )

        # Add TDR statistics
        if "tdr_eligible_flag_lookup" in comp_df.columns:
            stats["tdr_eligible_counts"] = (
                comp_df["tdr_eligible_flag_lookup"].value_counts().to_dict()
            )

        if "lot_buildout_gap" in comp_df.columns:
            stats["lot_buildout_gap"] = {
                "mean": comp_df["lot_buildout_gap"].mean(),
                "median": comp_df["lot_buildout_gap"].median(),
                "min": comp_df["lot_buildout_gap"].min(),
                "max": comp_df["lot_buildout_gap"].max(),
            }

        # Add STR eligibility distribution
        if "str_eligible" in comp_df.columns:
            stats["str_eligible_counts"] = (
                comp_df["str_eligible"].value_counts().to_dict()
            )

        # Add year distribution
        if "transaction_year" in comp_df.columns:
            stats["year_counts"] = comp_df["transaction_year"].value_counts().to_dict()

        return stats

    @staticmethod
    def compare_to_market(comp_df, market_df, time_adjust=True, price_col=None):
        """
        Compare comparable properties to the broader market or a defined segment.

        Args:
            comp_df (pd.DataFrame): DataFrame of comparable properties
            market_df (pd.DataFrame): DataFrame representing the market
            time_adjust (bool): Whether time adjustment is enabled
            price_col (str): Column to use for price analysis

        Returns:
            dict: Comparison metrics between comps and market
        """
        if comp_df.empty:
            return {"error": "No comparable properties found"}

        # Set default price column based on time adjustment
        if price_col is None:
            price_col = (
                "adjusted_price_per_sqft_time" if time_adjust else "price_per_sqft"
            )

        # Default to price_per_sqft if the specified column doesn't exist
        if price_col not in comp_df.columns:
            price_col = "price_per_sqft"

        # Skip if market segment is empty
        if market_df.empty:
            return {"error": "No properties found in the specified market segment"}

        # Calculate comparison metrics
        comp_avg_price = comp_df[price_col].mean()
        market_avg_price = market_df[price_col].mean()

        comparison = {
            "comp_count": len(comp_df),
            "market_count": len(market_df),
            f"comp_avg_{price_col}": comp_avg_price,
            f"market_avg_{price_col}": market_avg_price,
            "price_difference": comp_avg_price - market_avg_price,
            "price_difference_percent": ((comp_avg_price / market_avg_price) - 1) * 100
            if market_avg_price > 0
            else 0,
        }

        # Add demo score comparison if available
        if "demo_score" in comp_df.columns and "demo_score" in market_df.columns:
            comp_avg_demo = comp_df["demo_score"].mean()
            market_avg_demo = market_df["demo_score"].mean()
            comparison["comp_avg_demo_score"] = comp_avg_demo
            comparison["market_avg_demo_score"] = market_avg_demo
            comparison["demo_score_difference"] = comp_avg_demo - market_avg_demo

        # Add value add type distribution comparison
        if (
            "value_add_type" in comp_df.columns
            and "value_add_type" in market_df.columns
        ):
            comp_value_types = (
                comp_df["value_add_type"].value_counts(normalize=True).to_dict()
            )
            market_value_types = (
                market_df["value_add_type"].value_counts(normalize=True).to_dict()
            )
            comparison["comp_value_add_distribution"] = comp_value_types
            comparison["market_value_add_distribution"] = market_value_types

        # Add buildout potential comparison
        if (
            "lot_buildout_gap" in comp_df.columns
            and "lot_buildout_gap" in market_df.columns
        ):
            comp_avg_gap = comp_df["lot_buildout_gap"].mean()
            market_avg_gap = market_df["lot_buildout_gap"].mean()
            comparison["comp_avg_buildout_gap"] = comp_avg_gap
            comparison["market_avg_buildout_gap"] = market_avg_gap
            comparison["buildout_gap_difference"] = comp_avg_gap - market_avg_gap

        # Add year distribution comparison
        if (
            "transaction_year" in comp_df.columns
            and "transaction_year" in market_df.columns
        ):
            comp_years = (
                comp_df["transaction_year"].value_counts(normalize=True).to_dict()
            )
            market_years = (
                market_df["transaction_year"].value_counts(normalize=True).to_dict()
            )
            comparison["comp_year_distribution"] = comp_years
            comparison["market_year_distribution"] = market_years

        return comparison

    @staticmethod
    def export_results(comp_df, filename=None):
        """
        Export comparison results to CSV.

        Args:
            comp_df (pd.DataFrame): DataFrame of comparable properties
            filename (str, optional): Output filename, defaults to 'comp_results_{timestamp}.csv'

        Returns:
            str: Path to the saved CSV file
        """
        if comp_df.empty:
            logger.warning("No data to export")
            return None

        # Generate default filename if not provided
        if filename is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"comp_results_{timestamp}.csv"

        # Ensure the filename has .csv extension
        if not filename.endswith(".csv"):
            filename += ".csv"

        # Create a copy for exporting
        export_df = comp_df.copy()

        # Save to CSV
        try:
            export_df.to_csv(filename, index=False)
            logger.info(f"Results exported to {filename}")
            return filename
        except Exception as e:
            logger.error(f"Error exporting results: {e}")
            return None

    @staticmethod
    def print_comp_analysis_summary(result):
        """
        Print a formatted summary of the comp analysis results.

        Args:
            result (dict): The analysis result from run_comp_analysis_by_address
        """
        if not result.get("success", False):
            logger.error(result.get("error", "Unknown error in comp analysis"))
            return

        prop = result["property"]
        analysis = result["analysis"]
        market = result["market_comparison"]

        print("\n" + "=" * 80)
        print(f"PROPERTY ANALYSIS: {prop['address']}")
        print("=" * 80)

        # Property details
        print("\nSUBJECT PROPERTY DETAILS:")
        print(f"  Bedrooms: {prop['bedrooms']}")
        print(f"  Bathrooms: {prop['bathrooms']}")
        print(f"  Size: {prop['total_sqft']} sq ft")

        if isinstance(prop["asking_price"], (int, float)):
            print(f"  Price: ${prop['asking_price']:,.0f}")
        else:
            print(f"  Price: {prop['asking_price']}")

        if isinstance(prop["price_per_sqft"], (int, float)):
            print(f"  Price/SqFt: ${prop['price_per_sqft']:.2f}")
        else:
            print(f"  Price/SqFt: {prop['price_per_sqft']}")

        # Enhanced data
        if "demo_score" in prop:
            print(f"  Demo Score: {prop['demo_score']}")
        if "value_add_type" in prop:
            print(f"  Value-Add Type: {prop['value_add_type']}")
        if "condition" in prop:
            print(f"  Condition: {prop['condition']}")
        if "tdr_eligible" in prop:
            print(f"  TDR Eligible: {prop['tdr_eligible']}")
        if "lot_buildout_gap" in prop and isinstance(
            prop["lot_buildout_gap"], (int, float)
        ):
            print(f"  Buildout Potential: {prop['lot_buildout_gap']:.0f} sq ft")
        if "str_eligible" in prop:
            print(f"  STR Eligible: {prop['str_eligible']}")

        # Analysis results
        print("\nCOMPARABLE PROPERTIES ANALYSIS:")
        print(f"  Number of comps: {analysis.get('count', 0)}")

        price_col = next(
            (
                col
                for col in ["adjusted_price_per_sqft_time", "price_per_sqft"]
                if col in analysis
            ),
            None,
        )

        if price_col and price_col in analysis:
            price_stats = analysis[price_col]
            print(f"  Average {price_col}: ${price_stats['mean']:.2f}")
            print(f"  Median {price_col}: ${price_stats['median']:.2f}")
            print(f"  Range: ${price_stats['min']:.2f} - ${price_stats['max']:.2f}")

        # Demo score analysis
        if "demo_score" in analysis:
            demo_stats = analysis["demo_score"]
            print(f"\n  Average Demo Score: {demo_stats['mean']:.1f}")
            print(
                f"  Demo Score Range: {demo_stats['min']:.1f} - {demo_stats['max']:.1f}"
            )

        # Value-add distribution
        if "value_add_counts" in analysis:
            print("\n  Value-Add Types:")
            for value_type, count in analysis["value_add_counts"].items():
                print(f"    {value_type}: {count} properties")

        # Market comparison
        if market and market.get("comp_count", 0) > 0:
            print("\nMARKET COMPARISON:")
            if "price_difference_percent" in market:
                diff_pct = market["price_difference_percent"]
                if diff_pct > 0:
                    print(f"  Subject/comps are {diff_pct:.1f}% ABOVE market average")
                else:
                    print(
                        f"  Subject/comps are {abs(diff_pct):.1f}% BELOW market average"
                    )

            if "comp_avg_demo_score" in market and "market_avg_demo_score" in market:
                print(f"  Comp Avg Demo Score: {market['comp_avg_demo_score']:.1f}")
                print(f"  Market Avg Demo Score: {market['market_avg_demo_score']:.1f}")

        # Export file
        if "export_file" in result:
            print(f"\nResults exported to: {result['export_file']}")

        print("\n" + "=" * 80)

        # If few comps were found, provide guidance
        if analysis.get("count", 0) < 3:
            print(
                "\nNOTE: Few comparable properties were found with strict matching criteria."
            )
            print("Consider relaxing filters for a more comprehensive analysis:")
            print("  - Allow a wider price range")
            print("  - Include properties with different condition ratings")
            print("  - Expand the square footage range")

            if "filter_relaxation_applied" in result:
                print(
                    "\nFilters automatically relaxed:",
                    ", ".join(result["filter_relaxation_applied"]),
                )
