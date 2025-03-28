"""
Command line interface for the Enhanced Comparative Property Finder.
"""

import argparse
import traceback

from .comp_finder import EnhancedCompFinder, run_comp_analysis_by_address
from .analysis import PropertyAnalyzer
from .utils import logger


def get_parser():
    """Create the command line argument parser."""
    parser = argparse.ArgumentParser(description='Enhanced Comparative Property Finder')
    parser.add_argument('--address', type=str, help='Property address to analyze')
    parser.add_argument('--csv', type=str, help='Path to CSV file with property data')
    parser.add_argument('--limit', type=int, default=5, help='Maximum number of comps to return')
    parser.add_argument('--min-comps', type=int, default=3, help='Minimum number of comps to try to find')
    parser.add_argument('--similarity', type=float, default=0.7, 
                        help='Similarity threshold for matching (0-1)')
    parser.add_argument('--price-diff', type=float, default=0.35,
                        help='Maximum price difference percentage (0-1)')
    parser.add_argument('--sqft-diff', type=int, default=500,
                        help='Maximum square footage difference')
    parser.add_argument('--no-export', action='store_true', 
                        help='Do not export results to CSV')
    parser.add_argument('--demo-score', type=float, default=None,
                        help='Filter by minimum demo score (1-5)')
    parser.add_argument('--str-eligible', type=str, choices=['Yes', 'No'], default=None,
                        help='Filter by STR eligibility')
    parser.add_argument('--tdr-eligible', type=str, choices=['Yes', 'No'], default=None,
                        help='Filter by TDR eligibility')
    
    return parser


def run_cli():
    """Run the command line interface."""
    parser = get_parser()
    args = parser.parse_args()
    
    try:
        if args.address:
            # Run analysis by address
            result = run_comp_analysis_by_address(
                address=args.address,
                csv_path=args.csv,
                limit=args.limit,
                export_results=not args.no_export
            )
            # Return result for testing
            return result
        else:
            # If no address provided, run demo mode
            run_demo_mode(args)
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        traceback.print_exc()
        return {"error": str(e), "success": False}


def run_demo_mode(args):
    """Run the demo mode with criteria based on arguments."""
    finder = EnhancedCompFinder(csv_path=args.csv, time_adjust=True)
    
    print("\n=== Demo Score and Redevelopment Analysis ===")
    
    # Build criteria from arguments or use defaults
    criteria = {}
    
    if args.demo_score is not None:
        criteria["min_demo_score"] = args.demo_score
    else:
        criteria["min_demo_score"] = 4.0  # Default: High redevelopment score
    
    if args.str_eligible is not None:
        criteria["str_eligible"] = args.str_eligible
    else:
        criteria["str_eligible"] = "Yes"  # Default: STR eligible
    
    if args.tdr_eligible is not None:
        criteria["tdr_eligible"] = args.tdr_eligible
    else:
        criteria["tdr_eligible"] = "Yes"  # Default: TDR eligible
    
    criteria["min_lot_buildout_gap"] = 1000  # At least 1000 SF of buildout potential
    
    # Find properties matching criteria
    comps = finder.find_comps(
        criteria, limit=args.limit, sort_by="lot_buildout_gap", ascending=False
    )
    print(f"Found {len(comps)} teardown/redevelopment opportunities")
    
    if len(comps) == 0:
        print("No properties found with the specified criteria.")
        print("Try relaxing the criteria, such as:")
        print("  --demo-score 3.0")
        print("  --str-eligible No")
        print("  --tdr-eligible No")
        return
    
    # Analyze the comps
    analysis = PropertyAnalyzer.analyze_comps(comps, time_adjust=finder.time_adjust)
    
    # Print demo score statistics
    if "demo_score" in analysis:
        print(f"\nRedevelopment Metrics:")
        print(f"Average Demo Score: {analysis['demo_score']['mean']:.1f}/5.0")
        print(
            f"Score Range: {analysis['demo_score']['min']:.1f} - {analysis['demo_score']['max']:.1f}"
        )

    # Print value-add distribution
    if "value_add_counts" in analysis:
        print("\nValue-Add Distribution:")
        for value_type, count in analysis["value_add_counts"].items():
            print(f"  {value_type}: {count} properties")

    # Print buildout gap statistics
    if "lot_buildout_gap" in analysis:
        print(f"\nBuildout Potential:")
        print(
            f"Average Buildout Gap: {analysis['lot_buildout_gap']['mean']:.0f} SF"
        )
        print(
            f"Max Buildout Potential: {analysis['lot_buildout_gap']['max']:.0f} SF"
        )

    # Compare to broader market
    market_comparison = PropertyAnalyzer.compare_to_market(comps, finder.data, time_adjust=finder.time_adjust)
    print("\n=== Comparison to Overall Market ===")
    print(f"Comp Properties: {market_comparison.get('comp_count', 0)}")
    print(f"Market Properties: {market_comparison.get('market_count', 0)}")

    if "comp_avg_demo_score" in market_comparison:
        print(f"Comp Avg Demo Score: {market_comparison['comp_avg_demo_score']:.1f}/5.0")
        print(
            f"Market Avg Demo Score: {market_comparison['market_avg_demo_score']:.1f}/5.0"
        )

    if "comp_avg_buildout_gap" in market_comparison:
        print(
            f"Comp Avg Buildout Gap: {market_comparison['comp_avg_buildout_gap']:.0f} SF"
        )
        print(
            f"Market Avg Buildout Gap: {market_comparison['market_avg_buildout_gap']:.0f} SF"
        )

    # Show example property
    print("\n=== Example Property from Results ===")
    if not comps.empty:
        example = comps.iloc[0]
        print(f"Address: {example['full_address']}")
        print(f"Price: ${example.get('asking_price', 0):,.0f}")
        print(
            f"Size: {example['total_sqft']} sqft, {example['bedrooms']} bed, {example['total_baths']} bath"
        )

        if "demo_score" in example:
            print(
                f"Demo Score: {example['demo_score']}/5.0 ({example['value_add_type']})"
            )

        if "improved_condition" in example:
            print(
                f"Condition: {example['improved_condition']} (Confidence: {example.get('condition_confidence', 'N/A')})"
            )

        if "lot_buildout_gap" in example:
            print(f"Buildout Potential: {example['lot_buildout_gap']:.0f} SF")

        if "tdr_summary_note" in example:
            print(f"TDR Summary: {example['tdr_summary_note']}")

        if "agent_notes" in example:
            print(f"Agent Notes: {example['agent_notes']}")

    # Export results to CSV if requested
    if not args.no_export:
        PropertyAnalyzer.export_results(comps, "high_demo_score_properties.csv")
    
    # If sample addresses are available, suggest them to the user
    if not comps.empty and 'full_address' in comps.columns:
        print("\nSample addresses you can analyze:")
        for i, addr in enumerate(comps['full_address'].head(3)):
            print(f"  {i+1}. {addr}")
        print(f"\nTry running: python -m aspen_comp_finder.cli --address \"[address]\"")