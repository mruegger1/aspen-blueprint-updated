import streamlit as st
import pandas as pd
from io import StringIO
from aspen_comp_finder.classic_finder import ClassicCompFinder

st.set_page_config(page_title="Aspen Blueprint - Comp Finder", layout="wide")
st.title("🌼 Aspen Blueprint - Comparable Property Finder")
st.markdown("""
Upload your Aspen dataset, then select a property profile to find top comparable sales.
This MVP auto-loads our scored Aspen dataset and runs the ClassicCompFinder engine.
""")

# --- Load Data Once ---
@st.cache_data
def load_default_data():
    try:
        finder = ClassicCompFinder(csv_path="data/aspen_mvp_final_scored.csv")  # Directly specify the CSV path
        return finder, None
    except Exception as e:
        return None, str(e)

finder, error = load_default_data()

if error:
    st.error(f"Failed to load data: {error}")
    st.stop()
else:
    st.success(f"Default dataset loaded ({len(finder.data)} properties)")
    
    # Debug info - show columns to help diagnose issues
    with st.expander("Debug Data Info"):
        st.write("Available columns:", finder.data.columns.tolist())

# --- Determine the correct area column name ---
area_column = 'area'
possible_area_columns = ['area', 'Area', 'AREA', 'neighborhood', 'Neighborhood', 'location', 'Location']

for col in possible_area_columns:
    if col in finder.data.columns:
        area_column = col
        break

# --- User Filters ---
st.sidebar.header("Property Filters")
bedrooms = st.sidebar.slider("Bedrooms", 1, 10, 2)
bathrooms = st.sidebar.slider("Bathrooms", 1.0, 5.0, 2.0, step=0.5)
property_type = st.sidebar.selectbox("Property Type", ["Condo", "Townhouse", "Single Family"])

# Handle area selection safely
if area_column in finder.data.columns:
    areas = ["All Areas"] + sorted(finder.data[area_column].dropna().unique().tolist())
else:
    st.warning(f"No '{area_column}' column found in data. Area filtering disabled.")
    areas = ["All Areas"]
    
area = st.sidebar.selectbox("Area", areas)
str_eligible = st.sidebar.checkbox("Short-Term Rental Eligible", value=True, help="Check this box if the property is allowed to operate as a short-term rental.")
submit = st.sidebar.button("Find Comparable Properties")

# --- Data Analysis --- 
with st.expander("Dataset Property Distribution"):
    try:
        bed_bath_distribution = finder.data.groupby(['bedrooms', 'bathrooms']).size().reset_index(name='count')
        st.write("Bedrooms/Bathroom combinations in dataset:")
        st.dataframe(bed_bath_distribution, use_container_width=True)
        
        # Show property types distribution
        prop_type_distribution = finder.data['property_type'].value_counts().reset_index()
        prop_type_distribution.columns = ['Property Type', 'Count']
        st.write("Property types in dataset:")
        st.dataframe(prop_type_distribution, use_container_width=True)
    except Exception as e:
        st.warning(f"Couldn't analyze distribution: {e}")

# --- Search Logic ---
if submit:
    with st.spinner("Finding comps..."):
        # Print debug information about our search criteria
        st.write("Searching with the following criteria:")
        debug_criteria = {
            "bedrooms": bedrooms,
            "bathrooms": bathrooms,
            "property_type": property_type,
            "min_comps": 3,
            "limit": 5,
            "str_eligible": str_eligible
        }
        st.write(debug_criteria)
        
        # Get the expected column names from the finder
        try:
            # Try to determine the correct column names by inspecting the dataframe
            bed_col = 'bedrooms'  # Default parameter name expected by find_classic_comps
            bath_col = 'bathrooms'  # Default parameter name expected by find_classic_comps
            type_col = 'property_type'  # Default parameter name expected by find_classic_comps
            
            # Create criteria with the exact parameter names expected by find_classic_comps
            criteria = {
                "bedrooms": bedrooms,
                "bathrooms": bathrooms,
                "property_type": property_type,
                "min_comps": 3,
                "limit": 5,
            }
            
            # Add STR eligibility if that column exists
            if 'str_eligible' in finder.data.columns:
                criteria['str_eligible'] = str_eligible
            
            # Only add area if we're not using "All Areas"
            if area != "All Areas" and area_column in finder.data.columns:
                criteria[area_column] = area
            
            # For debugging
            st.write("Actual search criteria passed to finder:", criteria)
        except Exception as e:
            st.warning(f"Error setting up criteria: {e}")
            criteria = debug_criteria
        
        try:
            # First attempt with standard criteria
            results = finder.find_classic_comps(**criteria)
            comps_df = results["comps"]
            stats = results.get("stats", {})
            
            # If we get no results, try with fewer filters
            if len(comps_df) == 0:
                st.warning("No results found with strict criteria. Trying with fewer filters...")
                minimal_criteria = {
                    "limit": 10,
                    "property_type": property_type
                }
                
                st.write("Relaxed search criteria:", minimal_criteria)
                results = finder.find_classic_comps(**minimal_criteria)
                comps_df = results["comps"]
                stats = results.get("stats", {})
                
                if len(comps_df) > 0:
                    st.success("Found properties with relaxed criteria!")
                else:
                    st.error("Still no results. There may be an issue with the dataset or filtering logic.")
            
            # --- Results Table ---
            st.subheader("Top Comparable Properties")
            st.dataframe(comps_df, use_container_width=True)
            
            # --- Stats ---
            st.subheader("Summary Stats")
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Average Price", f"${stats.get('average_price', 0):,.0f}")
            with c2:
                st.metric("Price/SqFt Avg", f"${stats.get('average_price_per_sqft', 0):,.0f}")
            with c3:
                st.metric("Match Score Avg", f"{comps_df['match_score'].mean():.1f}")
                
            with st.expander("📊 Price Distribution"):
                if 'adjusted_sold_price_time' in comps_df.columns:
                    st.bar_chart(comps_df["adjusted_sold_price_time"])
                else:
                    price_col = next((col for col in comps_df.columns if 'price' in col.lower()), None)
                    if price_col:
                        st.bar_chart(comps_df[price_col])
                    else:
                        st.write("No price column found for chart")
                        
            st.markdown("*ℹ️ Match Score is a weighted similarity score across size, condition, STR status, etc.*")
            
            # --- Download ---
            st.subheader("Export Results")
            csv = comps_df.to_csv(index=False).encode("utf-8")
            st.download_button("Download CSV", csv, "comps_results.csv", "text/csv")
            st.toast("Comps ready! 🔍", icon="✅")
        except Exception as e:
            st.error(f"Search failed: {str(e)}")
            st.expander("Troubleshooting Details", expanded=True).write(f"""
            **Error details**: {str(e)}
            
            Possible issues:
            - Column names don't match what ClassicCompFinder expects
            - Selected criteria returned no results
            - Data format issues
            
            Try modifying your search criteria or check the debug info.
            """)
