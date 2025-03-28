import streamlit as st
import pandas as pd
from io import StringIO
from src.aspen_comp_finder.classic_finder import ClassicCompFinder

st.set_page_config(page_title="Aspen Blueprint - Comp Finder", layout="wide")
st.title("🌼 Aspen Blueprint - Comparable Property Finder")

st.markdown("""
Upload your Aspen dataset, then select a property profile to find top comparable sales.
This MVP auto-loads our scored Aspen dataset and runs the ClassicCompFinder engine.
""")

# --- Load Data Once ---
@st.cache_data

def load_default_data():
    finder = ClassicCompFinder()  # Uses default CSV path logic
    return finder

finder = load_default_data()
st.success(f"Default dataset loaded ({len(finder.data)} properties)")

# --- User Filters ---
st.sidebar.header("Property Filters")

bedrooms = st.sidebar.slider("Bedrooms", 1, 10, 2)
bathrooms = st.sidebar.slider("Bathrooms", 1.0, 5.0, 2.0, step=0.5)
property_type = st.sidebar.selectbox("Property Type", ["Condo", "Townhouse", "Single Family"])

areas = ["All Areas"] + sorted(finder.data["area"].dropna().unique().tolist())
area = st.sidebar.selectbox("Area", areas)

str_eligible = st.sidebar.checkbox("Short-Term Rental Eligible", value=True, help="Check this box if the property is allowed to operate as a short-term rental.")

submit = st.sidebar.button("Find Comparable Properties")

# --- Search Logic ---
if submit:
    with st.spinner("Finding comps..."):
        criteria = {
            "bedrooms": bedrooms,
            "bathrooms": bathrooms,
            "property_type": property_type,
            "area": None if area == "All Areas" else area,
            "str_eligible": str_eligible,
            "min_comps": 3,
            "limit": 5
        }

        try:
            results = finder.find_classic_comps(**criteria)
            comps_df = results["comps"]
            stats = results.get("stats", {})

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

            with st.expander("\ud83d\udcca Price Distribution"):
                st.bar_chart(comps_df["adjusted_sold_price_time"])

            st.markdown("*\u2139\ufe0f Match Score is a weighted similarity score across size, condition, STR status, etc.*")

            # --- Download ---
            st.subheader("Export Results")
            csv = comps_df.to_csv(index=False).encode("utf-8")
            st.download_button("Download CSV", csv, "comps_results.csv", "text/csv")

            st.toast("Comps ready! \ud83d\udd0d", icon="✅")

        except Exception as e:
            st.error(f"Search failed: {str(e)}")
            st.stop()
