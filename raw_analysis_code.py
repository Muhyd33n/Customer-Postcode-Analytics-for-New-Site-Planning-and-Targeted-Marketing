import streamlit as st
import pandas as pd
from math import radians, sin, cos, sqrt, atan2
import folium
from streamlit_folium import st_folium
import re
import matplotlib.pyplot as plt
import os, json, hashlib
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# OpenAI client: OPENAI_API_KEY in already set up in env. 
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Page configuration
st.set_page_config(page_title="Customer Postcode Analysis", page_icon="🔎", layout="wide")

# ---- Session state initialisation ----
if "search_done" not in st.session_state:
    st.session_state.search_done = False
if "results" not in st.session_state:
    st.session_state.results = None
if "merged_df" not in st.session_state:
    st.session_state.merged_df = None
if "center" not in st.session_state:
    st.session_state.center = None
if "params" not in st.session_state:
    st.session_state.params = {}
if "customer_segment_column" not in st.session_state:
    st.session_state.customer_segment_column = None
if "customer_spend_column" not in st.session_state:
    st.session_state.customer_spend_column = None
# full contacts dataframe for global stats
if "contacts_df" not in st.session_state:
    st.session_state.contacts_df = None

# Title
st.title("🔎 Customer Postcode Analysis")
st.markdown("Find and analyse customers within a specific radius of a UK postcode")

# Cache the postcode database to avoid reloading
@st.cache_data
def load_postcode_db():
    df = pd.read_csv('postcodes.csv', low_memory=False)[['Postcode', 'Latitude', 'Longitude']].copy()
    df['Postcode'] = df['Postcode'].astype(str).str.upper().str.strip()
    df['Postcode'] = df['Postcode'].str.replace(r'[^A-Z0-9]', '', regex=True)
    df['Postcode'] = df['Postcode'].apply(lambda x: x[:-3] + " " + x[-3:] if isinstance(x, str) and len(x) > 3 else x)
    return df

# UK postcode cleaner
def clean_uk_postcode(postcode):
    if pd.isna(postcode):
        return None
    postcode = str(postcode).upper().strip()
    postcode = re.sub(r'[^A-Z0-9]', '', postcode)
    if len(postcode) > 3:
        postcode = postcode[:-3] + " " + postcode[-3:]
    return postcode.strip()

# Haversine distance in miles
def haversine_distance(lat1, lon1, lat2, lon2):
    R = 3963.1  # miles
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c

# ---------- LLM helpers ----------
def _build_llm_payload(df: pd.DataFrame, centre_pc: str, radius_miles: float,
                       seg_col: str | None, spend_col: str | None) -> dict:
    """
    Create aggregate-only payload for AI. No raw personal data.
    - unique_customers uses all rows
    - spend stats use only positive, non-null spend values
    """
    payload: dict[str, object] = {
        "centre_postcode": centre_pc,
        "radius_miles": float(radius_miles),
        "unique_customers": int(len(df)),
    }

    # Area shares
    if "area" in df.columns:
        share = df["area"].value_counts(normalize=True).mul(100).round(1)
        payload["top_areas_pct"] = dict(share.head(12))
        payload["tail_areas_pct"] = dict(share.tail(12))

    # Segment shares
    if seg_col and seg_col in df.columns:
        seg_share = df[seg_col].value_counts(normalize=True).mul(100).round(1)
        payload["segment_share_pct"] = dict(seg_share.head(12))

    # Spend summaries and per-area means using only valid spenders
    if spend_col and spend_col in df.columns:
        spend_series = pd.to_numeric(df[spend_col], errors="coerce")

        # Valid positive spend only
        valid_spend = spend_series[spend_series > 0]

        if not valid_spend.empty:
            payload["spend_summary"] = {
                "mean": float(valid_spend.mean()),
                "median": float(valid_spend.median()),
                "p90": float(valid_spend.quantile(0.90)),
                "p10": float(valid_spend.quantile(0.10)),
                "valid_spend_customers": int(valid_spend.count()),
                "total_spend": float(valid_spend.sum()),
            }

            # Per-area averages over valid spenders only
            spend_by_area = (
                df.assign(_spend=pd.to_numeric(df[spend_col], errors="coerce"))
                  .query("_spend > 0")
                  .groupby("area", dropna=True)["_spend"]
                  .mean()
                  .round(2)
                  .sort_values(ascending=False)
            )
            if not spend_by_area.empty:
                payload["top_spend_areas"] = dict(spend_by_area.head(8))
                payload["low_spend_areas"] = dict(spend_by_area.tail(8))

    return payload


@st.cache_data(show_spinner=False)
def _summarise_with_llm(payload_json: str) -> str:
    """
    Cache by payload content so the LLM is only called when aggregates change.
    """
    _ = hashlib.sha256(payload_json.encode("utf-8")).hexdigest()  # cache key
    system = (
    "You are a UK retail insight storyteller creating concise, stakeholder friendly summaries for new site planning and post-launch marketing. "
    "Use only information provided in payload_json. Never assume or invent data. If something is missing, write 'Not available'. "
    "Explain what the data means for local demand, customer behaviour, and opportunities around a new site. "
    "Keep the language clear and narrative in tone, avoiding jargon. "
    "Use UK spelling and round money to the nearest pound. "
    "Keep output under 500 words. "
    "Always end with a section titled 'Recommendations' containing one clear action focused on site planning or launch marketing.")

    user = (
    "FORMAT:\n"
    "1. Start with a one-line headline that captures the key story about demand or opportunity for a new site.\n"
    "2. Then write short, stakeholder friendly bullet points under these headings:\n"
    "   - Hotspots: where customer activity or value is concentrated.\n"
    "   - Gaps: areas with low engagement, missing data, or limited coverage.\n"
    "   - Customer segments: list each segment with its share (segment_share_pct) and highlight what this mix means for a new site.\n"
    "   - Spend insights: explain spending patterns using the most relevant figures such as average spend, median, distribution ranges and total value.\n"
    "3. Use figures like '£123' and percentages like '45%'. Write clearly and avoid technical language.\n"
    "4. Finish with 'Recommendations' - one practical next step for new site planning or post-launch marketing.\n\n"
    "Important:\n"
    "- Use only data found in payload_json.\n"
    "- If data is missing, write 'Not available'.\n"
    "- Keep the story factual and avoid speculation.\n\n"
    f"payload_json:\n{payload_json}\n")

    
    resp = client.chat.completions.create(
        model="gpt-5",
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}],
        temperature=1,
    )
    return resp.choices[0].message.content.strip()
# ---------- end LLM helpers ----------

# Load postcode database
with st.spinner("Loading postcode database..."):
    postcode_db = load_postcode_db()
st.success(f"✓ Loaded {len(postcode_db):,} postcodes")

# Sidebar inputs in a form
st.sidebar.header("Search Parameters")
with st.sidebar.form("controls"):
    input_postcode_raw = st.text_input(
        "Enter UK Postcode:",
        placeholder="e.g., SW1A 1AA",
        help="Enter the centre postcode for your search",
    )
    radius_miles = st.number_input(
        "Radius (miles):", min_value=1.0, max_value=100.0, value=10.0, step=0.5
    )
    uploaded_file = st.file_uploader(
        "Upload Contacts File", type=['csv', 'xlsx'],
        help="Upload a CSV or Excel file with customer data",
    )
    postcode_column = st.text_input(
        "Postcode Column Name:", value="PostCode",
        help="Name of the column containing postcodes in your file",
    )
    customer_segment_column = st.text_input(
        "Customer Segment Column (optional):",
        value="",
        help="Name of the column containing customer segments (leave empty to skip)",
    )
    customer_spend_column = st.text_input(
        "Customer Spend Column (optional):",
        value="",
        help="Name of the column containing customer spend data (leave empty to skip)",
    )
    search_button = st.form_submit_button("🔍 Search", type="primary", use_container_width=True)

# Optional reset
if st.sidebar.button("🧹 Reset", use_container_width=True):
    st.session_state.search_done = False
    st.session_state.results = None
    st.session_state.merged_df = None
    st.session_state.center = None
    st.session_state.params = {}
    st.session_state.customer_segment_column = None
    st.session_state.customer_spend_column = None
    st.session_state.contacts_df = None
    st.success("State cleared")

# Main run on submit
if search_button:
    # Validate inputs
    input_postcode = clean_uk_postcode(input_postcode_raw) if input_postcode_raw else None
    if not input_postcode:
        st.error("⚠️ Please enter a valid postcode")
        st.stop()
    if not uploaded_file:
        st.error("⚠️ Please upload a contacts file")
        st.stop()

    # Load contacts
    try:
        if uploaded_file.name.lower().endswith(".csv"):
            Contact = pd.read_csv(uploaded_file, low_memory=False)
        else:
            Contact = pd.read_excel(uploaded_file)
        st.info(f"Loaded {len(Contact):,} contacts from file")

        if postcode_column not in Contact.columns:
            st.error(f"❌ Column '{postcode_column}' not found. Available columns: {', '.join(map(str, Contact.columns))}")
            st.stop()

        with st.spinner("Cleaning and standardising postcodes..."):
            Contact = Contact.copy()
            Contact[postcode_column] = Contact[postcode_column].apply(clean_uk_postcode)
            st.success(f"✓ Cleaned {Contact[postcode_column].notna().sum():,} postcodes")

        # store full contacts dataframe in session state for later runs
        st.session_state.contacts_df = Contact

    except Exception as e:
        st.error(f"Error loading contacts file: {e}")
        st.stop()

    # Locate centre postcode
    with st.spinner(f"Searching for postcodes within {radius_miles} miles of {input_postcode}..."):
        center_row = postcode_db[postcode_db['Postcode'] == input_postcode]
        if center_row.empty:
            st.error(f"❌ Could not find postcode '{input_postcode}' in the database.")
            st.stop()

        center_lat = float(center_row.iloc[0]['Latitude'])
        center_lon = float(center_row.iloc[0]['Longitude'])

        # Radius search
        distances = []
        progress_bar = st.progress(0)
        total_rows = len(postcode_db)

        for idx, (_, row) in enumerate(postcode_db.iterrows()):
            dist = haversine_distance(center_lat, center_lon, float(row['Latitude']), float(row['Longitude']))
            if dist <= radius_miles:
                distances.append({
                    'Postcode': row['Postcode'],
                    'distance_miles': round(dist, 2),
                    'area': str(row['Postcode']).split()[0],
                })
            if idx % 10000 == 0:
                progress_bar.progress(min((idx + 1) / total_rows, 1.0))
        progress_bar.progress(1.0)

        # Guard against empty search results
        if not distances:
            st.warning(f"No postcodes found within {radius_miles} miles of {input_postcode}. "
                       "Try a larger radius or check the centre postcode.")
            st.stop()

        # Proceed only if we have data
        results = pd.DataFrame(distances).sort_values('distance_miles').reset_index(drop=True)
        st.success(f"✓ Found {len(results):,} postcodes within {radius_miles} miles")
        
    # Match contacts
    with st.spinner("Matching customers..."):
        postcodes = results.rename(columns={'Postcode': 'PostCode'})
        merged_df = pd.merge(Contact, postcodes, left_on=postcode_column, right_on='PostCode', how='inner')

    # Persist so the UI does not disappear on rerun
    st.session_state.results = results
    st.session_state.merged_df = merged_df
    st.session_state.center = {"lat": center_lat, "lon": center_lon, "pc": input_postcode}
    st.session_state.params = {"radius_miles": radius_miles}
    st.session_state.customer_segment_column = customer_segment_column.strip() if customer_segment_column else None
    st.session_state.customer_spend_column = customer_spend_column.strip() if customer_spend_column else None
    st.session_state.search_done = True

# Always render if we have prior results
if st.session_state.search_done and st.session_state.results is not None:
    results = st.session_state.results
    merged_df = st.session_state.merged_df
    center_lat = st.session_state.center["lat"]
    center_lon = st.session_state.center["lon"]
    input_postcode = st.session_state.center["pc"]
    radius_miles = st.session_state.params["radius_miles"]
    contacts_df = st.session_state.contacts_df

    # ============================
    #           METRICS
    # ============================

    st.header("📊 Results")
    c1, c2, c3, c4 = st.columns(4)

    # Precompute global stats using full contacts dataframe
    total_customers = len(contacts_df) if contacts_df is not None else None
    total_avg_spend = None

    if (
        contacts_df is not None
        and st.session_state.customer_spend_column
        and st.session_state.customer_spend_column in contacts_df.columns
    ):
        all_spend = pd.to_numeric(
            contacts_df[st.session_state.customer_spend_column],
            errors="coerce",
        )
        all_spend = all_spend[all_spend > 0]
        total_avg_spend = all_spend.mean() if not all_spend.empty else None

    # Found within radius
    unique_found = len(merged_df)

    # -------------------------
    #      c1: TOTAL IN DB
    # -------------------------
    c1.markdown(
        "<div style='font-size: 14px; font-weight: 600;'>Total Customers in DB</div>",
        unsafe_allow_html=True,
    )

    if total_customers:
        c1_html = f"""
            <div style='font-size: 26px; font-weight: 600;'>
                {total_customers:,}
            </div>
        """
    else:
        c1_html = "<div style='font-size: 26px; font-weight: 600;'>N/A</div>"

    c1.markdown(c1_html, unsafe_allow_html=True)


    # ---------------------------------------------
    #   c2: CUSTOMERS FOUND IN POSTCODE RADIUS
    # ---------------------------------------------
    c2.markdown(
        "<div style='font-size: 14px; font-weight: 600;'>Customers Found in Postcode Radius</div>",
        unsafe_allow_html=True,
    )

    if total_customers and total_customers > 0:
        pct_unique = (unique_found / total_customers) * 100
        c2_html = f"""
            <div style='font-size: 26px; font-weight: 600;'>
                {unique_found:,}
                <span style='font-size: 16px; font-weight: 500;'> ({pct_unique:.1f}% of customer base)</span>
            </div>
        """
    else:
        c2_html = f"""
            <div style='font-size: 26px; font-weight: 600;'>
                {unique_found:,}
            </div>
        """

    c2.markdown(c2_html, unsafe_allow_html=True)


    # ---------------------------------------------
    #   Spend metrics INSIDE radius
    # ---------------------------------------------
    if (
        st.session_state.customer_spend_column
        and st.session_state.customer_spend_column in merged_df.columns
    ):
        valid_spend = pd.to_numeric(
            merged_df[st.session_state.customer_spend_column],
            errors="coerce",
        )
        valid_spend = valid_spend[valid_spend > 0]

        total_spend = valid_spend.sum() if not valid_spend.empty else 0
        avg_spend = valid_spend.mean() if not valid_spend.empty else 0

        # -------------------------
        #   c3: TOTAL SPEND IN AREA
        # -------------------------
        c3.markdown(
            "<div style='font-size: 14px; font-weight: 600;'>Postcode Area Customer Spend</div>",
            unsafe_allow_html=True,
        )

        c3_html = f"""
            <div style='font-size: 26px; font-weight: 600;'>
                £{total_spend:,.2f}
            </div>
        """
        c3.markdown(c3_html, unsafe_allow_html=True)


        # ------------------------------------------
        #   c4: AVERAGE SPEND + PERCENTAGE COLOURING
        # ------------------------------------------
        c4.markdown(
            "<div style='font-size: 14px; font-weight: 600;'>Postcode Area Average Spend</div>",
            unsafe_allow_html=True,
        )

        if total_avg_spend and total_avg_spend > 0:
            diff_pct = ((avg_spend - total_avg_spend) / total_avg_spend) * 100

            # Determine colour
            if abs(diff_pct) < 0.05:
                pct_colour = "black"
            elif diff_pct > 0:
                pct_colour = "green"
            else:
                pct_colour = "red"

            if abs(diff_pct) < 0.05:
                # In line with average
                bracket_text = "in line with overall avg spend"
                c4_html = f"""
                    <div style='font-size: 26px; font-weight: 600; white-space: nowrap;'>
                        £{avg_spend:,.2f}
                        <span style='font-size: 16px; font-weight: 500; color:black;'>({bracket_text})</span>
                    </div>
                """
            else:
                # With coloured % value
                comparison_symbol = ">" if diff_pct > 0 else "<"
                coloured_pct = f"<span style='color:{pct_colour};'>{diff_pct:+.1f}%</span>"

                bracket_text = (
                    f"{coloured_pct} {comparison_symbol} overall avg spend"
                )

                c4_html = f"""
                    <div style='font-size: 26px; font-weight: 600; white-space: nowrap;'>
                        £{avg_spend:,.2f}
                        <span style='font-size: 16px; font-weight: 500;'>({bracket_text})</span>
                    </div>
                """

        else:
            c4_html = f"""
                <div style='font-size: 26px; font-weight: 600;'>
                    £{avg_spend:,.2f}
                </div>
            """

        c4.markdown(c4_html, unsafe_allow_html=True)

    else:
        # If no spend column
        for col, title in [
            (c3, "Postcode Area Customer Spend"),
            (c4, "Postcode Area Average Spend"),
        ]:
            col.markdown(
                f"<div style='font-size: 14px; font-weight: 600;'>{title}</div>",
                unsafe_allow_html=True,
            )
            col.markdown(
                "<div style='font-size: 26px; font-weight: 600;'>N/A</div>",
                unsafe_allow_html=True,
            )


    # Map
    st.subheader("🗺️ Coverage Map")
    m = folium.Map(location=[center_lat, center_lon], zoom_start=10, tiles='OpenStreetMap')
    folium.Marker(
        [center_lat, center_lon],
        popup=f"<b>{input_postcode}</b><br>Centre Point",
        tooltip=input_postcode,
        icon=folium.Icon(color='red', icon='info-sign')
    ).add_to(m)
    folium.Circle(
        location=[center_lat, center_lon],
        radius=radius_miles * 1609.34,
        color='blue',
        fill=True,
        fill_color='lightblue',
        fill_opacity=0.2,
        popup=f"{radius_miles} mile radius",
        tooltip=f"{radius_miles} miles"
    ).add_to(m)
    st_folium(m, width=1400, height=500)

    # Table
    st.subheader("Customer Details")
    st.dataframe(merged_df, use_container_width=True, height=400)

    # Downloads
    st.subheader("📥 Download Results")
    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            "⬇️ Download Customers CSV",
            merged_df.to_csv(index=False).encode("utf-8"),
            file_name=f"{input_postcode.replace(' ', '_')}_customers.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with col2:
        st.download_button(
            "⬇️ Download Postcodes CSV",
            results.to_csv(index=False).encode("utf-8"),
            file_name=f"{input_postcode.replace(' ', '_')}_postcodes.csv",
            mime="text/csv",
            use_container_width=True,
        )

    # Customer Analytics (if segment or spend columns are available)
    customer_segment_column = st.session_state.customer_segment_column
    customer_spend_column = st.session_state.customer_spend_column

    if (customer_segment_column and customer_segment_column in merged_df.columns) or \
       (customer_spend_column and customer_spend_column in merged_df.columns):

        st.subheader("📊 Customer Analytics")

        # Case 1: Both segment and spend columns exist
        if customer_segment_column and customer_segment_column in merged_df.columns and \
           customer_spend_column and customer_spend_column in merged_df.columns:

            segment_counts = merged_df[customer_segment_column].value_counts(normalize=True) * 100
            avg_spend_seg = (merged_df[merged_df[customer_spend_column] > 0]
                             .dropna(subset=[customer_spend_column])
                             .groupby(customer_segment_column)[customer_spend_column].mean())

            fig, ax1 = plt.subplots(figsize=(10, 5), facecolor='#1E293B')
            ax1.set_facecolor('#1E293B')

            # Bar chart (percentages)
            bars = ax1.bar(segment_counts.index, segment_counts.values, alpha=0.7, color='skyblue', label="Customer %")
            ax1.set_ylabel("Percentage of Customers (%)", color='white')
            ax1.set_xlabel("Customer Segment", color='white')
            ax1.set_ylim(0, segment_counts.values.max() * 1.2)
            ax1.tick_params(colors='white')

            ax1.spines['top'].set_visible(False)
            ax1.spines['right'].set_visible(False)
            ax1.spines['left'].set_color('white')
            ax1.spines['bottom'].set_color('white')

            for bar in bars:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2, height + 1,
                         f"{height:.1f}%", ha='center', va='bottom', fontsize=9, color='white')

            # Line chart for average spend
            ax2 = ax1.twinx()
            from scipy.interpolate import make_interp_spline
            import numpy as np
            
            x_pos = np.arange(len(avg_spend_seg))
            y_vals = avg_spend_seg.values
            
            if len(x_pos) > 2:
                x_smooth = np.linspace(x_pos.min(), x_pos.max(), 300)
                spl = make_interp_spline(x_pos, y_vals, k=2)
                y_smooth = spl(x_smooth)
                ax2.plot(x_smooth, y_smooth, color='#10B981', linewidth=2, label="Average Spend")
                ax2.plot(x_pos, y_vals, 'o', color='#10B981', markersize=6)
            else:
                ax2.plot(x_pos, y_vals, color='#10B981', marker='o', linewidth=2, label="Average Spend")
            
            ax2.set_ylabel("Average Spend", color='white')
            ax2.tick_params(axis='y', labelcolor='white', colors='white')
            ax2.spines['top'].set_visible(False)
            
            for i, (idx, val) in enumerate(avg_spend_seg.items()):
                ax2.text(i, val, f"£{val:,.0f}", ha='center', va='bottom', fontsize=9, color='white')

            ax1.legend(loc="upper left", facecolor='#2D3B52', edgecolor='white', labelcolor='white')
            ax2.legend(loc="upper right", facecolor='#2D3B52', edgecolor='white', labelcolor='white')

            plt.title("Customer Segment Distribution and Average Spend", color='white')
            plt.xticks(rotation=0)
            plt.tight_layout()
            st.pyplot(fig)

        # Case 2: Only segment column exists
        elif customer_segment_column and customer_segment_column in merged_df.columns:
            segment_counts = merged_df[customer_segment_column].value_counts(normalize=True) * 100

            fig, ax = plt.subplots(figsize=(8, 4), facecolor='#1E293B')
            ax.set_facecolor('#1E293B')
            
            bars = ax.bar(segment_counts.index, segment_counts.values, color='skyblue')
            ax.set_ylabel("Percentage of Customers (%)", color='white')
            ax.set_xlabel("Customer Segment", color='white')
            ax.tick_params(colors='white')

            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_color('white')
            ax.spines['bottom'].set_color('white')

            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2, height + 1,
                        f"{height:.1f}%", ha='center', va='bottom', fontsize=9, color='white')

            plt.title("Customer Segment Distribution", color='white')
            plt.xticks(rotation=0)
            plt.tight_layout()
            st.pyplot(fig)

        # Case 3: Only spend column exists
        elif customer_spend_column and customer_spend_column in merged_df.columns:
            fig, ax = plt.subplots(figsize=(8, 4), facecolor='#1E293B')
            ax.set_facecolor('#1E293B')
            
            spend_data = merged_df[customer_spend_column].dropna()
            
            from scipy.interpolate import make_interp_spline
            import numpy as np
            
            x_pos = np.arange(len(spend_data))
            y_vals = spend_data.values
            
            if len(x_pos) > 2:
                x_smooth = np.linspace(x_pos.min(), x_pos.max(), 300)
                spl = make_interp_spline(x_pos, y_vals, k=2)
                y_smooth = spl(x_smooth)
                ax.plot(x_smooth, y_smooth, color='#10B981', linewidth=2)
                ax.plot(x_pos, y_vals, 'o', color='#10B981', markersize=6)
            else:
                ax.plot(x_pos, y_vals, color='#10B981', marker='o', linewidth=2)
            
            ax.set_ylabel("Spend (£)", color='white')
            ax.tick_params(colors='white')

            step = max(1, len(spend_data) // 10)
            for i in range(0, len(spend_data), step):
                val = spend_data.iloc[i]
                if pd.notna(val):
                    ax.text(i, val, f"£{val:,.0f}", ha='center', va='bottom', fontsize=8, color='white')

            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_color('white')
            ax.spines['bottom'].set_color('white')

            plt.title("Customer Spend Trend", color='white')
            plt.tight_layout()
            st.pyplot(fig)

    # ---------------- AI summary of results ----------------
    st.subheader("🧠 AI summary")
    st.caption("Only aggregated counts and statistics are sent to the AI. No raw personal data.")

    payload = _build_llm_payload(
        merged_df,
        centre_pc=input_postcode,
        radius_miles=radius_miles,
        seg_col=st.session_state.customer_segment_column,
        spend_col=st.session_state.customer_spend_column
    )

    payload_json = json.dumps(payload, sort_keys=True)

    placeholder = st.container()
    with placeholder:
        with st.status("Analysing results…", expanded=False):
            try:
                summary_text = _summarise_with_llm(payload_json)
                st.success("Summary ready")
            except Exception as e:
                summary_text = f"Sorry, the AI analysis failed: {e}"

    st.markdown(summary_text)

    st.download_button(
        "⬇️ Download summary (Markdown)",
        summary_text.encode("utf-8"),
        file_name=f"{input_postcode.replace(' ', '_')}_summary.md",
        mime="text/markdown",
        use_container_width=True,
    )

else:
    st.info("👈 Enter a postcode, radius, and upload a contacts file to begin")
    st.markdown("""
    ### How to use:
    1. Enter a UK postcode in the sidebar
    2. Set your search radius in miles
    3. Upload your contacts CSV or Excel file
    4. Specify the postcode column name in your file
    5. Specify the customer segment column name (optional)
    6. Specify the customer spend column name (optional)
    7. Click the Search button
    8. View and download the results

    ### Required file format:
    Your contacts file should contain at least these columns:
    - `PostCode`  UK postcode
    """)
