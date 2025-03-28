import streamlit as st
import pandas as pd
from io import StringIO
# Assuming ClassicCompFinder is in a reachable path or installed
# from aspen_comp_finder.classic_finder import ClassicCompFinder
# Placeholder for ClassicCompFinder if the actual import fails
try:
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
                    if 'bedrooms' not in self.data.columns: self.data['bedrooms'] = [2,3,4,2,3,5,1,3,4,2] * (len(self.data)//10 + 1) [:len(self.data)]
                    if 'bathrooms' not in self.data.columns: self.data['bathrooms'] = [2.0,2.5,3.0,1.5,2.0,4.0,1.0,2.5,3.5,2.0] * (len(self.data)//10 + 1) [:len(self.data)]
                    if 'property_type' not in self.data.columns: self.data['property_type'] = ['Condo', 'Single Family', 'Townhouse'] * (len(self.data)//3 + 1) [:len(self.data)]
                    if 'area' not in self.data.columns: self.data['area'] = ['Core', 'West End', 'Red Mountain', 'Smuggler'] * (len(self.data)//4 + 1) [:len(self.data)]
                    if 'str_eligible' not in self.data.columns: self.data['str_eligible'] = [True, False] * (len(self.data)//2 + 1) [:len(self.data)]
                    if 'adjusted_sold_price_time' not in self.data.columns: self.data['adjusted_sold_price_time'] = [500000, 1500000, 2500000, 800000, 1200000, 3500000, 400000, 950000, 1800000, 700000] * (len(self.data)//10 + 1) [:len(self.data)]
                    if 'sqft' not in self.data.columns: self.data['sqft'] = [1000, 2000, 3000, 1200, 1800, 4000, 800, 1500, 2500, 1100] * (len(self.data)//10 + 1) [:len(self.data)]
                    if 'match_score' not in self.data.columns: self.data['match_score'] = [0.8, 0.9, 0.7, 0.85, 0.95, 0.6, 0.75, 0.88, 0.78, 0.92] * (len(self.data)//10 + 1) [:len(self.data)]

                except Exception as e:
                    st.error(f"Dummy class failed to load CSV: {e}")
                    self.data = pd.DataFrame() # Initialize with empty dataframe
            elif data is not None:
                self.data = data
            else:
                self.data = pd.DataFrame() # Initialize with empty dataframe
            # Ensure essential columns exist even if loading fails or data is empty
            for col in ['bedrooms', 'bathrooms', 'property_type', 'area', 'str_eligible', 'adjusted_sold_price_time', 'sqft', 'match_score']:
                 if col not in self.data.columns:
                      self.data[col] = [] # Add empty column if missing


        def find_classic_comps(self, **kwargs):
            # Dummy implementation: return a slice of the data
            # This dummy doesn't actually filter based on exact matches or area lists
            # It's just for UI testing. The real ClassicCompFinder needs the logic.
            st.warning("Using Dummy `find_classic_comps`. Results are not accurately filtered.")
            limit = kwargs.get('limit', 5)
            filtered_df = self.data.copy()

            # --- Basic Filtering Simulation ---
            if 'bedrooms' in kwargs and 'bedrooms' in filtered_df.columns:
                 # Simulate approximate match (e.g., +/- 1) unless exact is specified
                 if kwargs.get('bedrooms_exact'):
                      filtered_df = filtered_df[filtered_df['bedrooms'] == kwargs['bedrooms']]
                 else:
                      filtered_df = filtered_df[filtered_df['bedrooms'].between(kwargs['bedrooms']-1, kwargs['bedrooms']+1)]

            if 'bathrooms' in kwargs and 'bathrooms' in filtered_df.columns:
                 # Simulate approximate match (e.g., +/- 0.5) unless exact is specified
                 if kwargs.get('bathrooms_exact'):
                      filtered_df = filtered_df[filtered_df['bathrooms'] == kwargs['bathrooms']]
                 else:
                      filtered_df = filtered_df[filtered_df['bathrooms'].between(kwargs['bathrooms']-0.5, kwargs['bathrooms']+0.5)]

            if 'property_type' in kwargs and 'property_type' in filtered_df.columns:
                 filtered_df = filtered_df[filtered_df['property_type'] == kwargs['property_type']]

            # --- NEW: Handle Area List ---
            area_col = self._get_area_col_name() # Find the area column dynamically
            if area_col and f'{area_col}' in kwargs and area_col in filtered_df.columns:
                 area_criteria = kwargs[f'{area_col}']
                 if isinstance(area_criteria, list) and area_criteria: # Check if it's a non-empty list
                      filtered_df = filtered_df[filtered_df[area_col].isin(area_criteria)]
                 elif isinstance(area_criteria, str): # Handle single string selection (backward compatibility)
                      filtered_df = filtered_df[filtered_df[area_col] == area_criteria]

            if 'str_eligible' in kwargs and 'str_eligible' in filtered_df.columns:
                 filtered_df = filtered_df[filtered_df['str_eligible'] == kwargs['str_eligible']]

            # --- Calculate Dummy Stats ---
            stats = {}
            if not filtered_df.empty:
                if 'adjusted_sold_price_time' in filtered_df.columns:
                    stats['average_price'] = filtered_df['adjusted_sold_price_time'].mean()
                if 'adjusted_sold_price_time' in filtered_df.columns and 'sqft' in filtered_df.columns and filtered_df['sqft'].gt(0).any():
                    # Avoid division by zero or NaN results
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
            # Read the uploaded file into a pandas DataFrame
            content = uploaded_file.getvalue().decode("utf-8")
            df = pd.read_csv(StringIO(content))
            # Initialize the finder with the DataFrame
            finder = ClassicCompFinder(data=df)
            return finder, None, f"Uploaded file '{uploaded_file.name}' loaded ({len(finder.data)} properties)"
        except Exception as e:
            return None, f"Failed to process uploaded file: {e}", None
    else:
        # Load default data
        try:
            # Adjust path as needed for your project structure
            default_path = "data/aspen_mvp_final_scored.csv"
            finder = ClassicCompFinder(csv_path=default_path)
            if finder.data.empty:
                 # Check if dummy data was loaded due to import error or file not found
                 if hasattr(finder, 'data') and not finder.data.empty:
                      return finder, None, f"Default dataset path ('{default_path}') not found or failed to load. Using dummy data ({len(finder.data)} properties)."
                 else:
                      return None, f"Default data file not found at '{default_path}' and dummy data failed.", None
            return finder, None, f"Default dataset loaded ({len(finder.data)} properties)"
        except Exception as e:
            return None, f"Failed to load default data: {e}", None

# --- File Uploader ---
uploaded_file = st.file_uploader("Upload your own CSV dataset (Optional)", type="csv")

# --- Load Data ---
finder, error, success_message = load_data(uploaded_file)

if error:
    st.error(error)
    st.stop()
elif success_message:
    st.success(success_message)
else:
    st.info("Ready to find comps.") # Should not happen if logic is correct, but good fallback

# Ensure finder is initialized before proceeding
if finder is None or finder.data is None or finder.data.empty:
    st.error("Data could not be loaded. Please upload a valid CSV or ensure the default data path is correct.")
    st.stop()

# --- Debug info - show columns ---
with st.expander("Debug Data Info"):
    st.write("Available columns:", finder.data.columns.tolist())
    # Check for expected columns
    expected_cols = ['bedrooms', 'bathrooms', 'property_type', 'adjusted_sold_price_time', 'sqft', 'str_eligible']
    missing_cols = [col for col in expected_cols if col not in finder.data.columns]
    if missing_cols:
        st.warning(f"Missing expected columns for analysis: {', '.join(missing_cols)}")
    # Attempt to get area column name using the method from the dummy class or real class
    area_column = finder._get_area_col_name() if hasattr(finder, '_get_area_col_name') else 'area' # Fallback
    if area_column not in finder.data.columns:
        st.warning(f"Could not reliably identify an area column (tried likely names like 'area', 'neighborhood', etc.). Area filtering might be affected.")
    else:
        st.info(f"Using column '{area_column}' for area filtering.")


# --- Determine the correct area column name (ensure finder has the method) ---
area_column = None
if hasattr(finder, '_get_area_col_name'):
     area_column = finder._get_area_col_name()

if not area_column or area_column not in finder.data.columns:
     # Fallback if method doesn't exist or doesn't find a column
     possible_area_columns = ['area', 'Area', 'AREA', 'neighborhood', 'Neighborhood', 'location', 'Location']
     for col in possible_area_columns:
          if col in finder.data.columns:
               area_column = col
               break

# --- User Filters ---
st.sidebar.header("Property Filters")

# Zillow Defaults Button (Simplified: Sets default values directly)
# st.sidebar.button("Use Zillow Defaults (3BR/2BA)") - Less effective without session state
# Instead, set defaults directly in sliders:
bedrooms = st.sidebar.slider("Bedrooms", 1, 10, 3) # Default to 3
exact_beds = st.sidebar.checkbox("Match Exact Bedrooms") # <<< New

bathrooms = st.sidebar.slider("Bathrooms", 1.0, 10.0, 2.0, step=0.5) # Default to 2.0, increased max
exact_baths = st.sidebar.checkbox("Match Exact Bathrooms") # <<< New

prop_types = ["Any"] + sorted(finder.data['property_type'].dropna().unique().tolist()) if 'property_type' in finder.data.columns else ["Any"]
property_type = st.sidebar.selectbox("Property Type", prop_types)

# Area Multi-Select
selected_areas = []
if area_column and area_column in finder.data.columns:
    areas = sorted(finder.data[area_column].dropna().unique().tolist())
    selected_areas = st.sidebar.multiselect(f"Select Areas (using '{area_column}' column)", options=areas, default=[]) # <<< Changed to multiselect
    if not areas:
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
# Moved slightly down to be less prominent if user just wants results
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
            if 'property_type' in finder.data.columns:
                prop_type_distribution = finder.data['property_type'].value_counts().reset_index()
                prop_type_distribution.columns = ['Property Type', 'Count']
                st.write("Property types:")
                st.dataframe(prop_type_distribution, use_container_width=True, height=200)
            else:
                 st.warning("Missing 'property_type' column for distribution.")
        except Exception as e:
            st.warning(f"Couldn't analyze property type distribution: {e}")


# --- Search Logic ---
if submit:
    with st.spinner("Finding comps..."):
        # Build criteria dictionary dynamically
        criteria = {
            "bedrooms": bedrooms,
            "bathrooms": bathrooms,
            # property_type handled below
            "min_comps": 3, # Keep min_comps potentially? Or rely on limit?
            "limit": 10, # Increased limit slightly
        }

        # Add exact match criteria if checked
        if exact_beds:
            criteria["bedrooms_exact"] = bedrooms
        if exact_baths:
            criteria["bathrooms_exact"] = bathrooms

        # Add property type if not "Any"
        if property_type != "Any":
             criteria["property_type"] = property_type

        # Add area filtering using the identified column name if areas selected
        if area_column and selected_areas: # Pass the list if not empty
            criteria[area_column] = selected_areas # <<< Pass list

        # Add STR eligibility filter if checked AND column exists
        if str_col_present and str_eligible_filter:
            criteria['str_eligible'] = True # Filter FOR true

        # Print debug information about the final search criteria
        st.write("Searching with the following criteria:")
        st.write(criteria) # Show the actual criteria passed

        try:
            # --- Execute Search ---
            # IMPORTANT: Assumes ClassicCompFinder.find_classic_comps is updated
            # to handle 'bedrooms_exact', 'bathrooms_exact', and list for area key.
            results = finder.find_classic_comps(**criteria)
            comps_df = results.get("comps", pd.DataFrame()) # Use .get for safety
            stats = results.get("stats", {})

            # --- Handle No Results ---
            if comps_df.empty:
                st.warning("No results found with the specified criteria.")
                # Optionally, could try a relaxed search here, but let's keep it simple first
                st.stop() # Stop execution if no comps found

            # --- Display Results ---
            st.subheader(f"Top {len(comps_df)} Comparable Properties Found")
            # Select and order columns for display
            display_cols = [
                'address', 'area', 'property_type', 'bedrooms', 'bathrooms',
                'sqft', 'adjusted_sold_price_time', 'price_per_sqft',
                'sold_date', 'str_eligible', 'match_score'
            ]
            # Filter display_cols to only those present in comps_df
            display_cols_present = [col for col in display_cols if col in comps_df.columns]
            # Add any missing essential columns back if they exist but weren't in the list
            for essential in ['bedrooms', 'bathrooms', 'adjusted_sold_price_time', 'match_score']:
                 if essential in comps_df.columns and essential not in display_cols_present:
                      display_cols_present.append(essential)

            st.dataframe(comps_df[display_cols_present], use_container_width=True)
            st.markdown("*ℹ️ Match Score is a weighted similarity score (higher is better).*")

            # --- Summary Stats ---
            st.subheader("Summary Stats for Found Comps")
            avg_price = stats.get('average_price', 0)
            avg_ppsf = stats.get('average_price_per_sqft', 0)
            avg_score = comps_df['match_score'].mean() if 'match_score' in comps_df.columns and not comps_df.empty else 0

            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Average Price", f"${avg_price:,.0f}" if avg_price else "N/A")
            with c2:
                st.metric("Price/SqFt Avg", f"${avg_ppsf:,.0f}" if avg_ppsf else "N/A")
            with c3:
                st.metric("Match Score Avg", f"{avg_score:.2f}" if avg_score else "N/A")


            # --- Analysis Tabs ---
            st.subheader("Analysis of Found Comps")
            tab1, tab2, tab3 = st.tabs(["📊 Price Chart", "🗺️ Area Stats", "🔑 STR Summary"]) # <<< Use tabs

            with tab1:
                st.markdown("#### Price Distribution")
                price_col_options = ['adjusted_sold_price_time', 'sold_price', 'price']
                price_col = next((col for col in price_col_options if col in comps_df.columns), None)

                if price_col:
                    st.bar_chart(comps_df[[price_col]].dropna()) # Ensure NaNs are dropped for chart
                    st.caption(f"Using column: `{price_col}`")
                else:
                    st.write("Could not find a suitable price column (e.g., 'adjusted_sold_price_time') for the chart.")

            with tab2:
                st.markdown("#### Comps by Area") # <<< New Section
                if area_column and area_column in comps_df.columns:
                    area_counts = comps_df[area_column].value_counts().reset_index()
                    area_counts.columns = ["Area", "Comps Found"]
                    st.dataframe(area_counts, use_container_width=True)
                elif selected_areas:
                     st.info(f"Area filter applied ({len(selected_areas)} areas), but the column '{area_column}' is not in the results DataFrame to show breakdown.")
                else:
                    st.write("No specific areas were selected, or area column not found in results.")

            with tab3:
                st.markdown("#### STR Eligibility Summary") # <<< New Section
                if 'str_eligible' in comps_df.columns:
                    # Attempt to map boolean/numeric to readable strings if needed
                    if comps_df['str_eligible'].dtype == 'bool':
                         str_summary = comps_df['str_eligible'].map({True: 'Eligible', False: 'Not Eligible'}).value_counts().rename_axis("STR Status").reset_index(name="Count")
                    elif pd.api.types.is_numeric_dtype(comps_df['str_eligible'].dtype):
                         # Simple assumption: 1 is Eligible, 0 is Not Eligible
                         str_summary = comps_df['str_eligible'].map({1: 'Eligible', 0: 'Not Eligible', -1: 'Unknown/Other'}).fillna('Unknown/Other').value_counts().rename_axis("STR Status").reset_index(name="Count")
                    else: # Assume string representation already exists
                         str_summary = comps_df['str_eligible'].value_counts().rename_axis("STR Status").reset_index(name="Count")

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
            st.exception(e) # Show full traceback for debugging
            st.expander("Troubleshooting Details", expanded=False).write(f"""
            **Error details**: {str(e)}

            **Criteria Used**: {criteria}

            **Possible issues**:
            - **`ClassicCompFinder` not updated**: The `find_classic_comps` method might not support `bedrooms_exact`, `bathrooms_exact`, or list-based area filtering. **The underlying Python class needs modification.**
            - **Column names**: Mismatch between expected names in `criteria` (like '{area_column}') and actual names in the data or the finder's internal logic.
            - **Data format**: Unexpected values in key columns (e.g., non-numeric beds/baths).

            Check the debug info above and ensure the `ClassicCompFinder` class implements the new filtering logic.
            """)
