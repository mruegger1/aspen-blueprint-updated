#!/usr/bin/env python3
"""
Command line interface for finding comparable properties.
"""

import argparse
import json
import logging
import sys
import os
from pathlib import Path
import pandas as pd

# Add the parent directory to the system path to import the package
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.insert(0, project_root)

# Try importing directly from the module - adjust these imports if needed
try:
    from src.aspen_comp_finder.classic_finder import ClassicCompFinder
except ImportError as e:
    print(f"Error importing ClassicCompFinder: {e}")
    print("Trying alternative import...")
    try:
        # Alternative import if the module structure is different
        sys.path.insert(0, os.path.join(project_root, 'src'))
        from aspen_comp_finder.classic_finder import ClassicCompFinder
    except ImportError as e2:
        print(f"Alternative import also failed: {e2}")
        print("\nDebug information:")
        print(f"Current directory: {os.getcwd()}")
        print(f"Python path: {sys.path}")
        
        # Find where the module might be
        for root, dirs, files in os.walk(project_root):
            if "classic_finder.py" in files:
                print(f"Found classic_finder.py in: {root}")
        
        sys.exit(1)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Find comparable properties in Aspen.")
    
    parser.add_argument("--csv-path", type=str, 
                        default=os.path.join(project_root, "data", "aspen_mvp_final_scored.csv"),
                        help="Path to the CSV file with property data")
    
    parser.add_argument("--bedrooms", type=int, required=True,
                        help="Number of bedrooms")
    
    parser.add_argument("--bathrooms", type=float, required=True,
                        help="Number of bathrooms")
    
    parser.add_argument("--property-type", type=str, required=True,
                        help="Property type (e.g., 'Condo', 'Single Family')")
    
    parser.add_argument("--area", type=str, required=True,
                        help="Area (e.g., 'Core', 'West End')")
    
    parser.add_argument("--str-eligible", action="store_true",
                        help="Property is eligible for short-term rentals")
    
    parser.add_argument("--limit", type=int, default=5,
                        help="Maximum number of comps to return")
    
    parser.add_argument("--output", type=str,
                        help="Path to output CSV file (optional)")
    
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug logging")
    
    return parser.parse_args()

def setup_logging(debug=False):
    """Configure logging."""
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

def main():
    """Run the comp finder."""
    args = parse_args()
    setup_logging(args.debug)
    
    # Log some debug info about paths
    logging.debug(f"Script directory: {script_dir}")
    logging.debug(f"Project root: {project_root}")
    logging.debug(f"CSV path: {args.csv_path}")
    
    try:
        # Check if the CSV file exists
        if not os.path.exists(args.csv_path):
            csv_paths = [
                args.csv_path,
                os.path.join(project_root, args.csv_path),
                os.path.join(project_root, "data", os.path.basename(args.csv_path))
            ]
            
            for path in csv_paths:
                if os.path.exists(path):
                    logging.info(f"Found CSV at: {path}")
                    args.csv_path = path
                    break
            else:
                logging.error(f"CSV file not found at: {args.csv_path}")
                print(f"ERROR: CSV file not found at: {args.csv_path}")
                sys.exit(1)
        
        # Initialize the comp finder
        logging.info("Initializing ClassicCompFinder...")
        finder = ClassicCompFinder(csv_path=args.csv_path)
        
        # Find comparable properties
        logging.info("Finding comps...")
        print(f"Finding comps for: {args.bedrooms}BR, {args.bathrooms}BA, {args.property_type}, {args.area}, STR: {args.str_eligible}")
        
        results = finder.find_classic_comps(
            bedrooms=args.bedrooms,
            bathrooms=args.bathrooms,
            property_type=args.property_type,
            area=args.area,
            str_eligible=args.str_eligible,
            limit=args.limit
        )
        
        # Extract results
        comps = results["comps"]
        stats = results["stats"]
        
        # DEBUG: Print the keys in the stats dictionary
        print("\nDEBUG - Stats keys:", list(stats.keys()))
        
        # Print results to console
        print("\n=== TOP COMPARABLE PROPERTIES ===")
        
        if len(comps) == 0:
            print("No comparable properties found.")
        else:
            for i, (_, comp) in enumerate(comps.iterrows(), 1):
                print(f"{i}. {comp['full_address']} - Match Score: {comp['match_score']:.1f}")
                print(f"   {comp['bedrooms']}BR, {comp['bathrooms']}BA, {comp['property_type']}, {comp['area']}")
                print(f"   Price: ${comp['price']:,.0f} ({comp['price_per_sqft']:.0f}/sqft)")
                print(f"   Date: {comp['date']}")
                print()
        
        # Print stats
        print("\n=== STATISTICS ===")
        
        # Use try/except blocks to handle missing keys gracefully
        try:
            print(f"Average Price: ${stats.get('avg_price', 0):,.0f}")
        except (KeyError, TypeError):
            print("Average Price: Not available")
            
        try:
            print(f"Average Price/SqFt: ${stats.get('avg_price_per_sqft', 0):,.0f}")
        except (KeyError, TypeError):
            print("Average Price/SqFt: Not available")
            
        try:
            print(f"Price Range: ${stats.get('min_price', 0):,.0f} - ${stats.get('max_price', 0):,.0f}")
        except (KeyError, TypeError):
            print("Price Range: Not available")
        
        # Save to CSV if output file specified
        if args.output:
            output_path = args.output
            comps.to_csv(output_path, index=False)
            print(f"\nSaved {len(comps)} comps to {output_path}")
            
            # Verify the file was created
            if os.path.exists(output_path):
                file_size = os.path.getsize(output_path)
                print(f"CSV file created successfully: {file_size} bytes")
            else:
                print("WARNING: Failed to create CSV file")
        
    except Exception as e:
        logging.exception("Error in comp finder")
        print(f"ERROR: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()