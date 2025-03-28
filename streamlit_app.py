import streamlit as st
import pandas as pd
from io import StringIO
# Assuming ClassicCompFinder is in a reachable path or installed
# from aspen_comp_finder.classic_finder import ClassicCompFinder
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
                    # Add dummy columns if they don't exist for demo purposes
                    # Simplify dummy data addition - ensure core columns exist
                    defaults = {
                        'bedrooms': [2,3,4,2,3,5,1,3,4,2],
                        'bathrooms': [2.0,2.5,3.0,1.5,2.0,4.0,1.0,2.5,3.5,2.0],
                        'property_type': ['Condo', 'Single Family', 'Townhouse', 'Duplex'] * 3, # Original column might exist
                        'resolved_property_type': ['Condominium', 'Single Family Home', 'Townhouse', 'Duplex'] * 3, # The target column
                        'area': ['Core', 'West End', 'Red Mountain', 'Smuggler'] * 3,
                        'str_eligible': [True, False] * 6,
                        'adjusted_sold_price_time': [500k, 1.5m, 2.5m, 800k, 1.2m, 3.5m, 400k, 950k, 1.8m, 700k, 2m, 1m],
                        'sqft': [1000, 2000, 3000, 1200, 1800, 4000, 800, 1500, 2500, 1100, 2200, 1400],
                        'match_score': [0.8, 0.9, 0.7, 0.85, 0.95, 0.6, 0.75, 0.88, 0.78, 0.92, 0.82, 0.72],
                        'address': [f'12{i} Main St' for i in range(12)],
                        'price_per_sqft': [500, 750, 833, 667, 667, 875, 500, 633, 720, 636, 909, 714],
                        'sold_date': pd.to_datetime(['2023-01-15', '2023-02-20', '2023-03-10', '2023-04-05', '2023-05-12', '2023-06-18', '2023-07-22', '2023-08-30', '2023-09-15', '2023-10-25', '2023-11-01', '2023-12-10'])
                    }
                    # Convert shorthand prices
                    defaults['adjusted_sold_price_time'] = [int(p.replace('k','000').replace('m','000000')) for p in defaults['adjusted_sold_price_time']]

                    data_len = len(self.data)
                    for col, default_list in defaults.items():
                         if col not in self.data.columns:
                              # Extend default list to match data length if needed
                              extended_list = (default_list * (data_len // len(default_list) + 1))[:data_len]
                              self.data[col] = extended_list

                except Exception as e:
                    st.error(f"Dummy class failed to load CSV or add defaults: {e}")
                    self.data = pd.DataFrame() # Initialize with empty dataframe
            elif data is not None:
                self.data = data
            else:
                self.data = pd.DataFrame() # Initialize with empty dataframe

            # Ensure essential columns exist even if loading fails or data is empty
            core_cols = ['bedrooms', 'bathrooms', 'property_type', 'resolved_property_type', 'area', 'str_eligible', 'adjusted_sold_price_time', 'sqft', 'match_score', 'address', 'price_per_sqft', 'sold_date']
            for col in core_cols:
                 if col not in self.data.columns:
                      self.data[col] = [] # Add empty column if missing


        def find_classic_comps(self, **kwargs):
            # Dummy implementation: return a slice of the data
            st.warning("Using Dummy `find_classic_comps`. Results are not accurately filtered.")
            limit = kwargs.get('limit', 5)
            filtered_df = self.data.copy()

            # --- Simulate Fallback Logic (Fix #2 - Optional) ---
            # This demonstrates the check. The actual filtering below in the dummy
            # might still rely explicitly on 'property_type' key from kwargs.
            # The *real* finder should ideally just use the keys passed in kwargs.
            type_col_in_kwargs = None
            if "resolved_property_type" in kwargs:
                type_col_in_kwargs = "resolved_property_type"
            elif "property_type" in kwargs:
                type_col_in_kwargs = "property_type"

            if "property_type" not in filtered_df.columns and "resolved_property_type" in filtered_df.columns:
                st.info("Simulating fallback: 'property_type' missing, 'resolved_property_type' exists in dummy data frame.")
                # In a real scenario, you might create the column if needed by downstream code:
                # filtered_df["property_type"] = filtered_df["resolved_property_type"]
                # But the goal is that the Streamlit fix makes this unnecessary by passing the correct key in kwargs.

            # --- Basic Filtering Simulation ---
            if 'bedrooms' in kwargs and 'bedrooms' in filtered_df.columns:
                 if kwargs.get('bedrooms_exact'):
                      filtered_df = filtered_df[filtered_df['bedrooms'] == kwargs['bedrooms']]
                 else:
                      filtered_df = filtered_df[filtered_df['bedrooms'].between(kwargs['bedrooms']-1, kwargs['bedrooms']+1)]

            if 'bathrooms' in kwargs and 'bathrooms' in filtered_df.columns:
                 if kwargs.get('bathrooms_exact'):
                      filtered_df = filtered_df[filtered_df['bathrooms'] == kwargs['bathrooms']]
                 else:
                      filtered_df = filtered_df[filtered_df['bathrooms'].between(kwargs['bathrooms']-0.5, kwargs['bathrooms']+0.5)]

            # --- MODIFIED: Use the property type key passed in kwargs ---
            if type_col_in_kwargs and type_col_in_kwargs in filtered_df.columns:
                filtered_df = filtered_df[filtered_df[type_col_in_kwargs] == kwargs[type_col_in_kwargs]]
            elif type_col_in_kwargs:
                st.warning(f"Column '{type_col_in_kwargs}' used for property type filter not found in DataFrame.")


            # --- Handle Area List ---
            area_col = self._get_area_col_name() # Find the area column dynamically
            if area_col and f'{area_col}' in kwargs and area_col in filtered_df.columns:
                 area_criteria = kwargs[f'{area_col}']
                 if isinstance(area_criteria, list) and area_criteria: # Check if it's a non-empty list
                      filtered_df = filtered_df[filtered_df[area_col].isin(area_criteria)]
                 elif isinstance(area_criteria, str): # Handle single string selection
                      filtered_df = filtered_df[filtered_df[area_col] == area_criteria]

            if 'str_eligible' in kwargs and 'str_eligible' in filtered_df.columns:
                 filtered_df = filtered_df[filtered_df['str_eligible'] == kwargs['str_eligible']]

            # --- Calculate Dummy Stats ---
            stats = {}
            if not filtered_df.empty:
                if 'adjusted_sold_price_time' in filtered_df.columns:
                    stats['average_price'] = filtered_df['adjusted_sold_price_time'].mean()
                if 'adjusted_sold_price_time' in filtered_df.columns and 'sqft' in filtered_df.columns and filtered_df['sqft'].gt(0).any():
                    valid_sqft = filtered_df[filtered_df['sqft'] > 0]
                    if not valid_sqft.empty:
                         stats['average_price_per_sqft'] = (valid_sqft['adjusted_sold_price_time'] / valid_sqft['sqft']).mean()
                    else:
                         stats['average_price_per_sqft'] = 0
                else:
                     stats['average_price_per_sqft'] = 0 # Default if columns missing or no valid sqft
            else:
                stats = {'average_price': 0, 'average_price_per_sqft': 0}


            # Apply limit
            comps = filtered_df.head(limit).copy()
            # Ensure match_score exists for display, add dummy if needed
            if 'match_score' not in comps.columns:
                comps['match_score'] = [0.9 - i*0.05 for i in range(len(comps))]


            return {"comps": comps, "stats": stats}

        def _get_area_col_name(self):
            """Helper to find the likely area column name."""
            possible_area_columns = ['area', 'Area', 'AREA', 'neighborhood', 'Neighborhood', 'location', 'Location']
            if hasattr(self, 'data') and self.data is not None: # Check if data exists
                 for col in possible_area_columns:
                      if col in self.data.columns:
                           return col
            return None # Return None if no likely column is found

# --- Page Config ---
st.set_page_config(page_title="Aspen Blueprint - Comp Finder", layout="wide")
st.title("🌼 Aspen Blueprint - Comparable Property Finder")
st.markdown("""
Upload your Aspen dataset (or use the default), then select a property profile to find top comparable sales.
This tool uses the ClassicCompFinder engine.
""")

# --- Load Data Once ---
@st.cache_data
def load_data(uploaded_file=None):
    if uploaded_file is not None:
        try:
            content = uploaded_file.getvalue().decode("utf-8")
            df = pd.read_csv(StringIO(content))
            finder = ClassicCompFinder(data=df)
            return finder, None, f"Uploaded file '{uploaded_file.name}' loaded ({len(finder.data)} properties)"
        except Exception as e:
            return None, f"Failed to process uploaded file: {e}", None
    else:
        try:
            default_path = "data/aspen_mvp_final_scored.csv"
            finder = ClassicCompFinder(csv_path=default_path)
            if not hasattr(finder, 'data') or finder.data is None or finder.data.empty:
                 # Attempt to create dummy finder if real one failed/no data
                 st.warning(f"Default data file not found or empty at '{default_path}'. Attempting to use dummy data.")
                 finder = ClassicCompFinder(data=pd.DataFrame()) # Init with empty, let dummy logic add cols
                 if not finder.data.empty:
                      return finder, None, f"Using dummy data structure ({len(finder.data)} placeholder rows)."
                 else:
                      return None, "Failed to load default data and dummy data generation failed.", None
            return finder, None, f"Default dataset loaded ({len(finder.data)} properties)"
        except Exception as e:
             # Last resort: Try dummy finder
             st.warning(f"Failed to load default data: {e}. Attempting dummy data.")
             try:
                  finder = ClassicCompFinder(data=pd.DataFrame())
                  if not finder.data.empty:
                       return finder, None, f"Using dummy data structure ({len(finder.data)} placeholder rows)."
                  else:
                       return None, "Failed to load default data and dummy data generation failed.", None
             except Exception as e_dummy:
                  return None, f"Failed to load default data and dummy data failed: {e_dummy}", None


# --- File Uploader ---
uploaded_file = st.file_uploader("Upload your own CSV dataset (Optional)", type="csv")

# --- Load Data ---
finder, error, success_message = load_data(uploaded_file)

if error:
    st.error(error)
    st.stop()
elif success_message:
    st.success(success_message)

# Ensure finder is initialized before proceeding
if finder is None or not hasattr(finder, 'data') or finder.data is None or finder.data.empty:
    st.error("Data could not be loaded. Please upload a valid CSV or ensure the default data path is correct.")
    st.stop()

# --- Debug info - show columns ---
with st.expander("Debug Data Info"):
    st.write("Available columns:", finder.data.columns.tolist())
    # Check for expected columns
    expected_cols = ['bedrooms', 'bathrooms', 'resolved_property_type', 'property_type', 'adjusted_sold_price_time', 'sqft', 'str_eligible'] # Added resolved_property_type
    missing_cols = [col for col in expected_cols if col not in finder.data.columns]
    if missing_cols:
        st.warning(f"Potentially missing expected columns: {', '.join(missing_cols)}")

    # Attempt to get area column name
    area_column = finder._get_area_col_name() if hasattr(finder, '_get_area_col_name') else None
    # Fallback if method fails or doesn't exist
    if not area_column or area_column not in finder.data.columns:
         possible_area_columns = ['area', 'Area', 'AREA', 'neighborhood', 'Neighborhood', 'location', 'Location']
         for col in possible_area_columns:
              if col in finder.data.columns:
                   area_column = col
                   break

    if area_column and area_column in finder.data.columns:
        st.info(f"Using column '{area_column}' for area filtering.")
    else:
        st.warning(f"Could not reliably identify an area column (tried likely names like 'area', 'neighborhood', etc.). Area filtering might be affected.")

    # Check for property type columns
    if 'resolved_property_type' in finder.data.columns:
         st.info("'resolved_property_type' column found.")
    else:
         st.warning("'resolved_property_type' column NOT found. Property type filtering may fail.")
    if 'property_type' in finder.data.columns:
         st.info("'property_type' column found (may be generic).")
    else:
         st.warning("'property_type' column NOT found.")


# --- User Filters ---
st.sidebar.header("Property Filters")

bedrooms = st.sidebar.slider("Bedrooms", 1, 10, 3)
exact_beds = st.sidebar.checkbox("Match Exact Bedrooms")

bathrooms = st.sidebar.slider("Bathrooms", 1.0, 10.0, 2.0, step=0.5)
exact_baths = st.sidebar.checkbox("Match Exact Bathrooms")

# --- MODIFIED: Dynamically select property type column ---
# Determine the best column to use for property type filters
type_column = "resolved_property_type" if "resolved_property_type" in finder.data.columns else "property_type"
# <<< MODIFIED

# Use the determined column to populate the dropdown
if type_column in finder.data.columns:
    prop_types = ["Any"] + sorted(finder.data[type_column].dropna().unique().tolist())
    st.sidebar.info(f"Using '{type_column}' for property types.") # Inform user
else:
    prop_types = ["Any"]
    st.sidebar.warning("No suitable 'property_type' or 'resolved_property_type' column found.")
# <<< MODIFIED (Logic to populate prop_types)

property_type = st.sidebar.selectbox("Property Type", prop_types)

# Area Multi-Select (using the previously determined area_column)
selected_areas = []
if area_column and area_column in finder.data.columns:
    areas = sorted(finder.data[area_column].dropna().unique().tolist())
    if areas:
         selected_areas = st.sidebar.multiselect(f"Select Areas (using '{area_column}')", options=areas, default=[])
    else:
         st.sidebar.warning(f"No unique values found in '{area_column}' column.")
else:
    st.sidebar.warning("No suitable 'Area' column found. Area filtering disabled.")

# STR Toggle
str_col_present = 'str_eligible' in finder.data.columns
str_eligible_filter = st.sidebar.checkbox("Must be Short-Term Rental Eligible", value=False, disabled=not str_col_present, help="Only include properties marked as STR Eligible.")
if not str_col_present:
     st.sidebar.caption("'_str_eligible_' column not found in data.")


submit = st.sidebar.button("Find Comparable Properties")

# --- Data Analysis (Distribution) ---
with st.expander("Dataset Property Distribution Overview"):
    col1, col2 = st.columns(2)
    with col1:
        try:
            if 'bedrooms' in finder.data.columns and 'bathrooms' in finder.data.columns:
                bed_bath_distribution = finder.data.groupby(['bedrooms', 'bathrooms']).size().reset_index(name='count').sort_values('count', ascending=False)
                st.write("Bed/Bath combinations:")
                st.dataframe(bed_bath_distribution, use_container_width=True, height=200)
            else:
                 st.warning("Missing 'bedrooms' or 'bathrooms' column for distribution.")
        except Exception as e:
            st.warning(f"Couldn't analyze bed/bath distribution: {e}")
    with col2:
        try:
            # --- MODIFIED: Use the dynamic type_column for distribution analysis ---
            if type_column in finder.data.columns:
                prop_type_distribution = finder.data[type_column].value_counts().reset_index()
                prop_type_distribution.columns = [type_column.replace('_',' ').title(), 'Count'] # Make header nice
                st.write(f"{type_column.replace('_',' ').title()} distribution:")
                st.dataframe(prop_type_distribution, use_container_width=True, height=200)
            else:
                 st.warning("Missing property type column for distribution.")
            # <<< MODIFIED
        except Exception as e:
            st.warning(f"Couldn't analyze property type distribution: {e}")


# --- Search Logic ---
if submit:
    with st.spinner("Finding comps..."):
        # Build criteria dictionary dynamically
        criteria = {
            "bedrooms": bedrooms,
            "bathrooms": bathrooms,
            "min_comps": 3,
            "limit": 10,
        }

        if exact_beds: criteria["bedrooms_exact"] = bedrooms
        if exact_baths: criteria["bathrooms_exact"] = bathrooms

        # --- MODIFIED: Use the dynamic type_column for criteria key ---
        if property_type != "Any" and type_column in finder.data.columns: # Check if column exists
             criteria[type_column] = property_type
        # <<< MODIFIED

        if area_column and selected_areas:
            criteria[area_column] = selected_areas

        if str_col_present and str_eligible_filter:
            criteria['str_eligible'] = True

        st.write("Searching with the following criteria:")
        st.write(criteria)

        try:
            # --- Execute Search ---
            results = finder.find_classic_comps(**criteria)
            comps_df = results.get("comps", pd.DataFrame())
            stats = results.get("stats", {})

            if comps_df.empty:
                st.warning("No results found with the specified criteria.")
                st.stop()

            # --- Display Results ---
            st.subheader(f"Top {len(comps_df)} Comparable Properties Found")

            # --- MODIFIED: Include both property type columns if they exist ---
            display_cols = [
                'address', area_column, # Use dynamic area column
                'resolved_property_type', 'property_type', # Show both for comparison if they exist
                'bedrooms', 'bathrooms', 'sqft', 'adjusted_sold_price_time',
                'price_per_sqft', 'sold_date', 'str_eligible', 'match_score'
            ]
            # <<< MODIFIED (added property_type)

            # Filter display_cols to only those present in comps_df AND not None
            display_cols_present = [col for col in display_cols if col and col in comps_df.columns]

            # Add any missing essential columns back if they exist but weren't in the list
            for essential in ['bedrooms', 'bathrooms', 'adjusted_sold_price_time', 'match_score']:
                 if essential in comps_df.columns and essential not in display_cols_present:
                      display_cols_present.append(essential)

            # Ensure address is first if present
            if 'address' in display_cols_present:
                 display_cols_present.insert(0, display_cols_present.pop(display_cols_present.index('address')))


            st.dataframe(comps_df[display_cols_present], use_container_width=True)
            st.markdown("*ℹ️ Match Score is a weighted similarity score (higher is better).*")

            # --- Summary Stats ---
            st.subheader("Summary Stats for Found Comps")
            avg_price = stats.get('average_price', 0)
            avg_ppsf = stats.get('average_price_per_sqft', 0)
            avg_score = comps_df['match_score'].mean() if 'match_score' in comps_df.columns and not comps_df.empty else 0

            c1, c2, c3 = st.columns(3)
            with c1: st.metric("Average Price", f"${avg_price:,.0f}" if avg_price else "N/A")
            with c2: st.metric("Price/SqFt Avg", f"${avg_ppsf:,.0f}" if avg_ppsf else "N/A")
            with c3: st.metric("Match Score Avg", f"{avg_score:.2f}" if avg_score else "N/A")


            # --- Analysis Tabs ---
            st.subheader("Analysis of Found Comps")
            tab1, tab2, tab3 = st.tabs(["📊 Price Chart", "🗺️ Area Stats", "🔑 STR Summary"])

            with tab1:
                st.markdown("#### Price Distribution")
                price_col_options = ['adjusted_sold_price_time', 'sold_price', 'price']
                price_col = next((col for col in price_col_options if col in comps_df.columns), None)

                if price_col and not comps_df[price_col].dropna().empty:
                    st.bar_chart(comps_df[[price_col]].dropna())
                    st.caption(f"Using column: `{price_col}`")
                else:
                    st.write("Could not find a suitable price column or data for the chart.")

            with tab2:
                st.markdown("#### Comps by Area")
                if area_column and area_column in comps_df.columns:
                    area_counts = comps_df[area_column].value_counts().reset_index()
                    area_counts.columns = ["Area", "Comps Found"]
                    st.dataframe(area_counts, use_container_width=True)
                elif selected_areas:
                     st.info(f"Area filter applied ({len(selected_areas)} areas), but the column '{area_column}' is not in the results DataFrame to show breakdown.")
                else:
                    st.write("No specific areas were selected, or area column not found in results.")

            with tab3:
                st.markdown("#### STR Eligibility Summary")
                if 'str_eligible' in comps_df.columns:
                    try:
                         # Attempt robust mapping for boolean, numeric (0/1), or string
                         if pd.api.types.is_bool_dtype(comps_df['str_eligible']):
                              str_map = {True: 'Eligible', False: 'Not Eligible'}
                         elif pd.api.types.is_numeric_dtype(comps_df['str_eligible']):
                              str_map = {1: 'Eligible', 0: 'Not Eligible'}
                         else: # Assume string or handle other types as 'Unknown'
                              str_map = {'yes': 'Eligible', 'no': 'Not Eligible', 'true': 'Eligible', 'false': 'Not Eligible', 1: 'Eligible', 0: 'Not Eligible'} # Add common string variants

                         # Apply mapping, treat unmappable as 'Unknown'
                         str_series = comps_df['str_eligible'].astype(str).str.lower().map(str_map).fillna('Unknown')
                         str_summary = str_series.value_counts().rename_axis("STR Status").reset_index(name="Count")
                         st.dataframe(str_summary, use_container_width=True)

                    except Exception as e:
                         st.warning(f"Could not summarize STR eligibility: {e}")
                         # Fallback: Show raw counts
                         str_summary = comps_df['str_eligible'].value_counts().rename_axis("STR Status (Raw)").reset_index(name="Count")
                         st.dataframe(str_summary, use_container_width=True)

                else:
                    st.write("'str_eligible' column not found in results.")


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
            st.exception(e)
            st.expander("Troubleshooting Details", expanded=False).write(f"""
            **Error details**: {str(e)}
            **Criteria Used**: {criteria}
            **Type Column Used**: {type_column if 'type_column' in locals() else 'N/A'}
            **Possible issues**:
            - **`ClassicCompFinder` needs update**: Ensure the *real* `find_classic_comps` can handle criteria keys like '{type_column}' and list-based area filtering.
            - **Column names**: Check if `{type_column}` or `{area_column}` actually exist in the loaded data (see Debug info above).
            - **Data format**: Unexpected values in key columns.
            Check the debug info above and ensure the `ClassicCompFinder` class implements the necessary filtering logic.
            """)
