import streamlit as st
import pandas as pd
from io import StringIO
import numpy as np # Needed for price/sqft calculation safety

# Placeholder for ClassicCompFinder if the actual import fails
try:
    # Attempt to import the real class
    from aspen_comp_finder.classic_finder import ClassicCompFinder
except ImportError:
    st.warning("Could not import `ClassicCompFinder`. Using a dummy class for demonstration.")
    # Create a dummy class that mimics the expected behavior for UI development
    class ClassicCompFinder:
        def __init__(self, csv_path=None, data=None):
            if csv_path:
                try:
                    self.data = pd.read_csv(csv_path)
                except Exception as e:
                    st.error(f"Dummy class: Failed to load CSV from {csv_path}: {e}")
                    self.data = pd.DataFrame() # Start empty if load fails
            elif data is not None:
                self.data = data.copy() # Use a copy to avoid modifying original data
            else:
                self.data = pd.DataFrame() # Start empty

            # --- Ensure Essential Columns & Data Types ---
            # Define expected columns and rough types (None means any)
            expected_schema = {
                'bedrooms': 'int64', 'bathrooms': 'float64',
                'property_type': None, 'resolved_property_type': None, # Keep original and resolved
                'area': None, 'str_eligible': None,
                'adjusted_sold_price_time': 'float64', 'sold_price': 'float64',
                'sqft': 'float64', 'total_sqft': 'float64',
                'match_score': 'float64', 'address': None,
                'price_per_sqft': 'float64', 'sold_date': 'datetime64[ns]',
                'year_built': 'int64', 'year_remodeled': 'float64', # Add year cols
                'latitude': 'float64', 'longitude': 'float64' # Add geo cols
            }

            # Add missing columns with default empty lists
            for col, dtype in expected_schema.items():
                if col not in self.data.columns:
                    self.data[col] = [] # Add as empty

            # Attempt to convert types, handling potential errors
            for col, dtype in expected_schema.items():
                if dtype and col in self.data.columns:
                    try:
                        if 'datetime' in dtype:
                            self.data[col] = pd.to_datetime(self.data[col], errors='coerce')
                        elif 'int' in dtype:
                             # Convert float to int safely (handle NaN)
                             self.data[col] = pd.to_numeric(self.data[col], errors='coerce').astype('Int64')
                        else: # float or other
                            self.data[col] = pd.to_numeric(self.data[col], errors='coerce')
                    except Exception as e:
                        st.warning(f"Dummy class: Could not convert column '{col}' to {dtype}: {e}. Keeping original type.")

            # --- Add SqFt Fallback ---
            if 'sqft' not in self.data.columns or self.data['sqft'].isnull().all():
                 if 'total_sqft' in self.data.columns:
                      st.info("Dummy class: Using 'total_sqft' as fallback for 'sqft'.")
                      self.data['sqft'] = self.data['total_sqft']

            # --- Add Price Fallback ---
            if 'adjusted_sold_price_time' not in self.data.columns or self.data['adjusted_sold_price_time'].isnull().all():
                 if 'sold_price' in self.data.columns:
                      st.info("Dummy class: Using 'sold_price' as fallback for 'adjusted_sold_price_time'.")
                      self.data['adjusted_sold_price_time'] = self.data['sold_price']

            # --- Generate Dummy Data ONLY if DataFrame is still empty ---
            if self.data.empty:
                st.info("Dummy class: Generating sample data as DataFrame was empty.")
                defaults = {
                    'bedrooms': [2, 3, 4, 2, 3, 5, 1, 3, 4, 2, 10, 4],
                    'bathrooms': [2.0, 2.5, 3.0, 1.5, 2.0, 4.0, 1.0, 2.5, 3.5, 2.0, 10.0, 5.0],
                    'resolved_property_type': ['Condominium', 'Single Family Home', 'Townhouse', 'Duplex'] * 3,
                    'property_type': ['Residential'] * 12, # More realistic original
                    'area': ['Core', 'West End', 'Red Mountain', 'Smuggler'] * 3,
                    'str_eligible': [True, False] * 6,
                    'adjusted_sold_price_time': [500000, 1500000, 2500000, 800000, 1200000, 3500000, 400000, 950000, 1800000, 700000, 50000000, 15000000],
                    'sqft': [1000, 2000, 3000, 1200, 1800, 4000, 800, 1500, 2500, 1100, 10000, 5000],
                    'match_score': [0.8, 0.9, 0.7, 0.85, 0.95, 0.6, 0.75, 0.88, 0.78, 0.92, 0.99, 0.89],
                    'address': [f'12{i} Main St' for i in range(12)],
                    'sold_date': pd.to_datetime(['2023-01-15', '2023-02-20', '2023-03-10', '2023-04-05', '2023-05-12', '2023-06-18', '2023-07-22', '2023-08-30', '2023-09-15', '2023-10-25', '2023-11-01', '2023-12-10']),
                    'year_built': [1980, 1995, 2010, 1975, 2000, 2018, 1965, 1988, 2005, 1992, 2022, 2015]
                }
                # Calculate price/sqft
                defaults['price_per_sqft'] = [p / s if s > 0 else 0 for p, s in zip(defaults['adjusted_sold_price_time'], defaults['sqft'])]
                self.data = pd.DataFrame(defaults)
                # Add other expected columns as empty/NaN
                for col, dtype in expected_schema.items():
                     if col not in self.data.columns:
                          self.data[col] = pd.NA if 'int' in str(dtype) or 'float' in str(dtype) else None if 'datetime' not in str(dtype) else pd.NaT
                          if dtype:
                               try:
                                    if 'datetime' in dtype: self.data[col] = pd.to_datetime(self.data[col])
                                    elif 'int' in dtype: self.data[col] = self.data[col].astype('Int64')
                                    else: self.data[col] = pd.to_numeric(self.data[col])
                               except: pass # Ignore conversion errors on empty addition

        # --- MODIFIED: Dummy find_classic_comps with NEW Filters ---
        def find_classic_comps(self, **kwargs):
            st.warning("Using Dummy `find_classic_comps`. Results simulate filtering but may not be fully accurate.")
            limit = kwargs.get('limit', 10)
            filtered_df = self.data.copy()

            # Price Filter
            price_col = 'adjusted_sold_price_time' # Primary price column
            if price_col not in filtered_df.columns or filtered_df[price_col].isnull().all():
                 price_col = 'sold_price' # Fallback
            if price_col in filtered_df.columns:
                 min_price = kwargs.get('min_price', 0)
                 max_price = kwargs.get('max_price', float('inf'))
                 filtered_df = filtered_df[filtered_df[price_col].between(min_price, max_price, inclusive='both')]
            else:
                 st.warning("Dummy Finder: No price column ('adjusted_sold_price_time' or 'sold_price') found for filtering.")

            # Bedrooms Filter
            if 'bedrooms' in filtered_df.columns:
                 min_beds = kwargs.get('bedrooms_min', 0)
                 max_beds = kwargs.get('bedrooms_max', float('inf'))
                 filtered_df = filtered_df[filtered_df['bedrooms'].between(min_beds, max_beds, inclusive='both')]

            # Bathrooms Filter
            if 'bathrooms' in filtered_df.columns:
                 min_baths = kwargs.get('bathrooms_min', 0.0)
                 max_baths = kwargs.get('bathrooms_max', float('inf'))
                 filtered_df = filtered_df[filtered_df['bathrooms'].between(min_baths, max_baths, inclusive='both')]

            # Property Type Filter (List)
            prop_type_col = 'resolved_property_type' # Primary type column
            if prop_type_col not in filtered_df.columns:
                 prop_type_col = 'property_type' # Fallback
            if prop_type_col in filtered_df.columns and 'property_types' in kwargs:
                 selected_types = kwargs['property_types']
                 if isinstance(selected_types, list) and selected_types: # Ensure it's a non-empty list
                      filtered_df = filtered_df[filtered_df[prop_type_col].isin(selected_types)]
            elif 'property_types' in kwargs:
                 st.warning(f"Dummy Finder: Property type column ('{prop_type_col}') not found for filtering.")

            # Area Filter (List)
            area_col = self._get_area_col_name() # Find the area column dynamically
            if area_col and 'area' in kwargs and area_col in filtered_df.columns: # Key in kwargs is still 'area'
                 selected_areas = kwargs['area']
                 if isinstance(selected_areas, list) and selected_areas: # Check if it's a non-empty list
                      filtered_df = filtered_df[filtered_df[area_col].isin(selected_areas)]
            elif 'area' in kwargs:
                 st.warning(f"Dummy Finder: Area column ('{area_col}') not found for filtering.")

            # STR Filter
            if 'str_eligible' in filtered_df.columns and kwargs.get('str_eligible') is True:
                 # Handle boolean or numeric representation (e.g., 1 for True)
                 if pd.api.types.is_bool_dtype(filtered_df['str_eligible']):
                      filtered_df = filtered_df[filtered_df['str_eligible'] == True]
                 elif pd.api.types.is_numeric_dtype(filtered_df['str_eligible']):
                      filtered_df = filtered_df[filtered_df['str_eligible'] == 1]
                 else: # Attempt string conversion
                      filtered_df = filtered_df[filtered_df['str_eligible'].astype(str).str.lower().isin(['true', 'yes', '1'])]

            # --- Calculate Dummy Stats on Filtered Data ---
            stats = {}
            if not filtered_df.empty:
                 if price_col in filtered_df.columns:
                      stats['average_price'] = filtered_df[price_col].mean()
                 else:
                      stats['average_price'] = 0

                 sqft_col = 'sqft' if 'sqft' in filtered_df.columns else 'total_sqft' if 'total_sqft' in filtered_df.columns else None
                 if price_col in filtered_df.columns and sqft_col and sqft_col in filtered_df.columns:
                    # Avoid division by zero or NaN results, using numpy for safety
                    valid_for_ppsf = filtered_df[[price_col, sqft_col]].dropna()
                    valid_for_ppsf = valid_for_ppsf[valid_for_ppsf[sqft_col] > 0]
                    if not valid_for_ppsf.empty:
                         stats['average_price_per_sqft'] = np.mean(valid_for_ppsf[price_col] / valid_for_ppsf[sqft_col])
                    else:
                         stats['average_price_per_sqft'] = 0
                 else:
                      stats['average_price_per_sqft'] = 0
            else:
                stats = {'average_price': 0, 'average_price_per_sqft': 0}


            # Apply limit (in a real scenario, this happens after scoring/sorting)
            # For dummy, just take head
            comps = filtered_df.head(limit).copy()

            # Ensure match_score exists for display
            if 'match_score' not in comps.columns:
                # Assign dummy scores if missing - less relevant now with stricter filters
                comps['match_score'] = np.linspace(0.9, 0.7, len(comps)) if len(comps) > 0 else []

            return {"comps": comps, "stats": stats}

        def _get_area_col_name(self):
            """Helper to find the likely area column name."""
            possible_area_columns = ['area', 'Area', 'AREA', 'neighborhood', 'Neighborhood', 'location', 'Location', 'major_area', 'Major Area']
            if hasattr(self, 'data') and self.data is not None:
                 for col in possible_area_columns:
                      if col in self.data.columns:
                           return col
            return 'area' # Default fallback

# --- Page Config ---
st.set_page_config(page_title="Aspen Blueprint - Comp Finder", layout="wide")
st.title("🌼 Aspen Blueprint - Comparable Property Finder")
st.markdown("""
Upload your Aspen dataset (or use the default), then adjust filters to find relevant comparable sales.
""")

# --- Load Data Once ---
@st.cache_data
def load_data(uploaded_file=None):
    """
    Loads data using ClassicCompFinder.
    Prioritizes loading via csv_path for the real class.
    Handles dummy class creation if import fails or default path missing.
    Does NOT perform preprocessing here - that happens after loading.
    """
    finder_instance = None
    error_msg = None
    success_msg = None
    using_dummy = False

    # Check if the real class was imported
    try:
        from aspen_comp_finder.classic_finder import ClassicCompFinder
        real_class_imported = True
    except ImportError:
        real_class_imported = False
        # Use the dummy class defined globally in the script
        global ClassicCompFinder # Make sure we're using the dummy one if defined
        using_dummy = True

    if uploaded_file is not None:
        # Handling uploaded files with the REAL class is tricky if it only accepts paths.
        if real_class_imported:
             error_msg = "File uploads are not currently supported with the main ClassicCompFinder. Please use the default dataset."
        # If using the DUMMY class, allow loading from upload
        elif not real_class_imported:
             try:
                  content = uploaded_file.getvalue().decode("utf-8")
                  df = pd.read_csv(StringIO(content))
                  # Initialize DUMMY finder with data=df
                  finder_instance = ClassicCompFinder(data=df)
                  success_msg = f"Uploaded file '{uploaded_file.name}' loaded into DUMMY finder ({len(finder_instance.data)} properties)."
             except Exception as e:
                  error_msg = f"Failed to process uploaded file for DUMMY finder: {e}"
        else:
             error_msg = "Inconsistent state regarding class import and file upload."

    else: # No file uploaded, use default path
        default_path = "data/aspen_mvp_final_scored.csv"
        try:
            # If REAL class is available, initialize with csv_path
            if real_class_imported:
                finder_instance = ClassicCompFinder(csv_path=default_path)
                # Basic check if data loaded (assuming .data attribute exists)
                if not hasattr(finder_instance, 'data') or finder_instance.data is None or finder_instance.data.empty:
                     raise ValueError(f"Real ClassicCompFinder loaded no data from {default_path}")
                success_msg = f"Default dataset loaded via ClassicCompFinder ({len(finder_instance.data)} properties)."

            # If REAL class failed import, use DUMMY class (which might also try path)
            else:
                st.warning("Using DUMMY ClassicCompFinder.")
                finder_instance = ClassicCompFinder(csv_path=default_path) # Dummy might handle path OR generate data
                if not hasattr(finder_instance, 'data') or finder_instance.data is None or finder_instance.data.empty:
                     # If dummy also failed to load/generate, report error
                     raise ValueError(f"Dummy ClassicCompFinder could not load or generate data (path: {default_path}).")
                success_msg = f"DUMMY finder initialized (tried path, may use generated data) ({len(finder_instance.data)} properties)."

        except FileNotFoundError:
             error_msg = f"Default data file not found at '{default_path}'. "
             if not real_class_imported:
                  try:
                       finder_instance = ClassicCompFinder() # Try init empty dummy -> generates data
                       if hasattr(finder_instance, 'data') and not finder_instance.data.empty:
                            success_msg = f"Using DUMMY generated data ({len(finder_instance.data)} rows)."
                       else: error_msg += "Dummy data generation failed."
                  except Exception as e_dummy: error_msg += f"Dummy data generation failed: {e_dummy}"
             else: error_msg += "Cannot proceed without default file for the main finder."

        except Exception as e:
            error_msg = f"Failed to load default data using ClassicCompFinder: {e}. "
            # Try dummy as last resort if not already using it
            if real_class_imported: # Only try if the error wasn't from the dummy already
                 try:
                      st.warning("Attempting to use DUMMY finder as fallback.")
                      finder_instance = ClassicCompFinder() # Init empty dummy
                      if hasattr(finder_instance, 'data') and not finder_instance.data.empty:
                           success_msg = f"Using DUMMY generated data ({len(finder_instance.data)} rows)."
                           error_msg = None # Clear primary error if dummy succeeds
                      else: error_msg += "Dummy fallback failed."
                 except Exception as e_dummy_fb: error_msg += f"Dummy fallback failed: {e_dummy_fb}"

    # Final check
    if finder_instance is None or not hasattr(finder_instance, 'data') or finder_instance.data is None:
        # Ensure error message reflects total failure if needed
        if not error_msg: error_msg = "Data could not be loaded or generated."
        finder_instance = None # Make sure it's None if no data

    return finder_instance, error_msg, success_msg

# --- File Uploader ---
uploaded_file = st.file_uploader("Upload your own CSV dataset (Optional)", type="csv")

# --- Load Data ---
finder, error, success_message = load_data(uploaded_file)

if error:
    st.error(error)
    if finder is None or not hasattr(finder, 'data') or finder.data.empty:
        st.stop()
    else:
        st.warning("Proceeding with dummy data structure due to loading errors.")
elif success_message:
    st.success(success_message)

# Final check before accessing finder.data
if finder is None or not hasattr(finder, 'data') or finder.data is None:
     st.error("Critical error: Finder object not available or has no data. Cannot proceed.")
     st.stop()

# --- Apply Data Preprocessing Directly to finder.data ---
try:
    # Determine the property type column
    proc_type_col = 'resolved_property_type' if 'resolved_property_type' in finder.data.columns else 'property_type'
    if proc_type_col in finder.data.columns:
        initial_count = len(finder.data)
        # Filter out 'Mobile Home'
        if 'Mobile Home' in finder.data[proc_type_col].unique():
            finder.data = finder.data[finder.data[proc_type_col] != 'Mobile Home'].copy() # Modify finder's data
            removed_count = initial_count - len(finder.data)
            if removed_count > 0:
                 st.info(f"Removed {removed_count} Mobile Home entries.")
    else:
        st.warning("Preprocessing: Could not find property type column for Mobile Home removal.")

    # Handle NaNs in key columns if needed (example: fill price NaNs with 0 for filtering)
    proc_price_col = 'adjusted_sold_price_time' if 'adjusted_sold_price_time' in finder.data.columns else 'sold_price'
    if proc_price_col in finder.data.columns and finder.data[proc_price_col].isnull().any():
         st.info(f"Column '{proc_price_col}' contains NaN values, which might affect filtering if not handled by the finder logic.")
    # Add similar handling for beds, baths, sqft if necessary

except Exception as e_proc:
     st.error(f"Error during post-load data preprocessing: {e_proc}")
     st.stop()
# --- End Preprocessing ---

# --- Determine Columns ---
# Use helper or fallback for Area column
area_column = finder._get_area_col_name() if hasattr(finder, '_get_area_col_name') else 'area'
if area_column not in finder.data.columns:
     st.warning(f"Identified area column '{area_column}' not found in data. Area filtering may fail.")
     area_column = None # Disable area filtering if column missing

# Determine Property Type column
type_column = 'resolved_property_type' if 'resolved_property_type' in finder.data.columns else 'property_type'
if type_column not in finder.data.columns:
     st.warning("Neither 'resolved_property_type' nor 'property_type' found. Type filtering disabled.")
     type_column = None # Disable type filtering

# Determine Price column
price_column = 'adjusted_sold_price_time' if 'adjusted_sold_price_time' in finder.data.columns and not finder.data['adjusted_sold_price_time'].isnull().all() else 'sold_price'
if price_column not in finder.data.columns:
     st.warning("Neither 'adjusted_sold_price_time' nor 'sold_price' found. Price filtering disabled.")
     price_column = None


# --- Debug info ---
with st.expander("Debug Data Info"):
    st.write("Available columns:", finder.data.columns.tolist())
    st.info(f"Using '{area_column}' for Area filtering (if available).")
    st.info(f"Using '{type_column}' for Property Type filtering (if available).")
    st.info(f"Using '{price_column}' for Price filtering (if available).")


# --- MODIFIED: User Filters Sidebar ---
st.sidebar.header("Property Filters")

# --- PRICE ---
price_min = st.sidebar.number_input(
    f"Min Price ($) (using '{price_column}')",
    min_value=0, max_value=200_000_000, value=1_000_000, step=100_000,
    disabled=(price_column is None),
    help=f"Filters on the '{price_column}' column."
)
price_max = st.sidebar.number_input(
    "Max Price ($)",
    min_value=0, max_value=200_000_000, value=50_000_000, step=100_000, # Adjusted default max
    disabled=(price_column is None)
)

# --- BEDROOMS ---
beds_min = st.sidebar.number_input("Min Bedrooms", min_value=0, max_value=20, value=2, disabled=('bedrooms' not in finder.data.columns))
beds_max = st.sidebar.number_input("Max Bedrooms", min_value=0, max_value=20, value=5, disabled=('bedrooms' not in finder.data.columns))
# Add validation feedback if min > max
if beds_min > beds_max:
     st.sidebar.warning("Min Bedrooms cannot be greater than Max Bedrooms.")

# --- BATHROOMS ---
baths_min = st.sidebar.number_input("Min Bathrooms", min_value=0.0, max_value=25.0, value=2.0, step=0.5, disabled=('bathrooms' not in finder.data.columns))
baths_max = st.sidebar.number_input("Max Bathrooms", min_value=0.0, max_value=25.0, value=6.0, step=0.5, disabled=('bathrooms' not in finder.data.columns))
# Add validation feedback if min > max
if baths_min > baths_max:
     st.sidebar.warning("Min Bathrooms cannot be greater than Max Bathrooms.")


# --- PROPERTY TYPE ---
available_types = []
if type_column and type_column in finder.data.columns:
    available_types = sorted(finder.data[type_column].dropna().unique().tolist())

# Determine default: try to select common types, or all if few options
default_types = available_types # Default to all initially
common_types = ['Single Family Home', 'Condominium', 'Townhouse', 'Duplex', 'Half Duplex']
subset_default = [t for t in common_types if t in available_types]
if len(subset_default) > 0 and len(subset_default) < len(available_types):
     default_types = subset_default # Default to common types if present

property_types = st.sidebar.multiselect(
    f"Property Types (using '{type_column}')",
    available_types,
    default=default_types, # Default to common types or all
    disabled=(type_column is None or not available_types)
)

# --- AREA ---
area_options = []
if area_column and area_column in finder.data.columns:
    area_options = sorted(finder.data[area_column].dropna().unique().tolist())

selected_areas = st.sidebar.multiselect(
    f"Neighborhood / Area (using '{area_column}')",
    area_options,
    default=area_options, # Default to all areas
    disabled=(area_column is None or not area_options)
)

# --- STR ---
str_col_present = 'str_eligible' in finder.data.columns
str_eligible_filter = st.sidebar.checkbox(
    "Must be STR Eligible",
    value=False,
    disabled=not str_col_present,
    help="Requires 'str_eligible' column to be True or 1."
)
if not str_col_present:
     st.sidebar.caption("'str_eligible' column not found.")


submit = st.sidebar.button("Find Comparable Properties")

# --- Data Analysis (Distribution) --- Display relevant distributions
with st.expander("Dataset Distribution Overview"):
     col1, col2 = st.columns(2)
     with col1:
          if type_column and type_column in finder.data.columns:
               try:
                    type_dist = finder.data[type_column].value_counts().reset_index()
                    type_dist.columns = [type_column.replace('_',' ').title(), 'Count']
                    st.write(f"{type_column.replace('_',' ').title()} Counts:")
                    st.dataframe(type_dist, use_container_width=True, height=200)
               except Exception as e: st.warning(f"Could not display type distribution: {e}")
          else: st.caption("Property type column not available.")
     with col2:
          if area_column and area_column in finder.data.columns:
               try:
                    area_dist = finder.data[area_column].value_counts().reset_index()
                    area_dist.columns = [area_column.replace('_',' ').title(), 'Count']
                    st.write(f"{area_column.replace('_',' ').title()} Counts:")
                    st.dataframe(area_dist, use_container_width=True, height=200)
               except Exception as e: st.warning(f"Could not display area distribution: {e}")
          else: st.caption("Area column not available.")


# --- MODIFIED: Search Logic ---
if submit:
    # --- Input Validation ---
    validation_passed = True
    if beds_min > beds_max:
        st.sidebar.error("Min Bedrooms > Max Bedrooms!")
        validation_passed = False
    if baths_min > baths_max:
        st.sidebar.error("Min Bathrooms > Max Bathrooms!")
        validation_passed = False
    if price_min > price_max:
        st.sidebar.error("Min Price > Max Price!")
        validation_passed = False
    if not property_types:
         st.sidebar.error("Please select at least one Property Type.")
         validation_passed = False
    if not selected_areas:
         st.sidebar.error("Please select at least one Neighborhood / Area.")
         validation_passed = False

    if not validation_passed:
         st.error("Please correct the errors in the filters before submitting.")
         st.stop()
    # --- End Validation ---

    with st.spinner("Finding comps..."):
        # --- Build criteria dictionary ---
        criteria = {
            "limit": 15, # Show a few more potential comps
        }
        # Only add filters if the corresponding column exists
        if price_column:
             criteria["min_price"] = price_min
             criteria["max_price"] = price_max
        if 'bedrooms' in finder.data.columns:
             criteria["bedrooms_min"] = beds_min
             criteria["bedrooms_max"] = beds_max
        if 'bathrooms' in finder.data.columns:
             criteria["bathrooms_min"] = baths_min
             criteria["bathrooms_max"] = baths_max
        if type_column and property_types: # Pass list if column exists and types selected
            criteria["property_types"] = property_types # Key is plural for list
        if area_column and selected_areas: # Pass list if column exists and areas selected
            criteria["area"] = selected_areas # Key is singular 'area' but value is list
        if str_col_present and str_eligible_filter:
            criteria['str_eligible'] = True # Only add if checked

        st.write("Searching with criteria:")
        st.write(criteria)

        try:
            # --- Execute Search ---
            results = finder.find_classic_comps(**criteria)
            comps_df = results.get("comps", pd.DataFrame())
            stats = results.get("stats", {})

            # --- Handle No Results ---
            if comps_df.empty:
                st.warning("No results found with the specified criteria. Try relaxing the filters (e.g., widen price range, include more areas/types).")
                # Placeholder for future "expand search" logic
                st.stop()

            # --- Poor Comp Warning (Placeholder) ---
            avg_comp_price = stats.get('average_price', 0)
            # Simple check: if average comp price is less than half the min requested price
            # This needs refinement, ideally comparing to a SUBJECT property price if available.
            if avg_comp_price > 0 and price_min > 0 and avg_comp_price < (price_min / 2):
                 st.warning("⚠️ Warning: Found comps are significantly below the requested minimum price. Consider adjusting price filters or other criteria for better relevance.")
            # Add similar check for $/sqft if desired

            # --- Display Results ---
            st.subheader(f"Top {len(comps_df)} Comparable Properties Found")
            display_cols = [
                'address', area_column, type_column, # Use dynamic cols
                'bedrooms', 'bathrooms', 'sqft', # Add sqft
                price_column, # Use dynamic price col
                'price_per_sqft', 'sold_date', 'year_built', # Add year
                'str_eligible', 'match_score'
            ]
            display_cols_present = [col for col in display_cols if col and col in comps_df.columns]

            # Reorder essentials if needed (e.g., put price near end)
            if price_column in display_cols_present:
                 display_cols_present.remove(price_column)
                 display_cols_present.insert(6, price_column) # Insert after sqft


            st.dataframe(
                comps_df[display_cols_present],
                column_config={ # Add formatting
                    price_column: st.column_config.NumberColumn(f"{price_column.replace('_',' ').title()}", format="$%d"),
                    "price_per_sqft": st.column_config.NumberColumn("Price/SqFt", format="$%d"),
                    "sqft": st.column_config.NumberColumn("SqFt", format="%d"),
                    "match_score": st.column_config.NumberColumn("Match Score", format="%.2f"),
                    "sold_date": st.column_config.DateColumn("Sold Date", format="MM/DD/YYYY"),
                 },
                use_container_width=True
            )

            # --- Summary Stats ---
            st.subheader("Summary Stats for Found Comps")
            avg_price = stats.get('average_price', 0)
            avg_ppsf = stats.get('average_price_per_sqft', 0)
            avg_score = comps_df['match_score'].mean() if 'match_score' in comps_df.columns and not comps_df.empty else 0

            c1, c2, c3 = st.columns(3)
            with c1: st.metric("Average Price", f"${avg_price:,.0f}" if avg_price else "N/A")
            with c2: st.metric("Price/SqFt Avg", f"${avg_ppsf:,.0f}" if avg_ppsf else "N/A")
            with c3: st.metric("Match Score Avg", f"{avg_score:.2f}" if avg_score else "N/A")


            # --- Analysis Tabs --- (Keep simple for now)
            st.subheader("Analysis of Found Comps")
            tab1, tab2 = st.tabs(["📊 Price Chart", "🗺️ Area/Type Counts"])

            with tab1:
                st.markdown("#### Price Distribution")
                if price_column and price_column in comps_df.columns and not comps_df[price_column].dropna().empty:
                    st.bar_chart(comps_df[[price_column]].dropna())
                    st.caption(f"Using column: `{price_column}`")
                else:
                    st.write("Could not find price data for chart.")

            with tab2:
                 st.markdown("#### Counts by Area & Type")
                 c_a, c_t = st.columns(2)
                 with c_a:
                      if area_column and area_column in comps_df.columns:
                           area_counts = comps_df[area_column].value_counts().rename_axis("Area").reset_index(name="Count")
                           st.dataframe(area_counts, use_container_width=True)
                      else: st.caption("Area data not available.")
                 with c_t:
                      if type_column and type_column in comps_df.columns:
                           type_counts = comps_df[type_column].value_counts().rename_axis("Type").reset_index(name="Count")
                           st.dataframe(type_counts, use_container_width=True)
                      else: st.caption("Type data not available.")


            # --- Download ---
            st.subheader("Export Results")
            try:
                 csv = comps_df.to_csv(index=False).encode("utf-8")
                 st.download_button("Download Results as CSV", csv, f"comps_results_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.csv", "text/csv", key='download-csv')
            except Exception as e:
                 st.error(f"Failed to generate CSV for download: {e}")

            st.toast("Comps ready! 🔍", icon="✅")

        except Exception as e:
            st.error(f"An error occurred during the search: {str(e)}")
            st.exception(e) # Show full traceback for debugging
