import streamlit as st
import pandas as pd
import plotly.express as px
import json
from models.forecasting_models import generate_forecasts, inventory_optimization_eoq
import os
# Page config

st.set_page_config(page_title="📊 Smart Stock Dashboard", layout="wide")
st.title("🧠 Smart Stock Inventory Optimization Dashboard")
CREDENTIALS_FILE = "credentials.json"

# -----------------------------
# Initialize credentials file
# -----------------------------
if not os.path.exists(CREDENTIALS_FILE):
    default_creds = {"username": "owner", "password": "admin123"}
    with open(CREDENTIALS_FILE, "w") as f:
        json.dump(default_creds, f, indent=4)

# -----------------------------
# Helper: Check Login
# -----------------------------
def check_login(username, password):
    with open(CREDENTIALS_FILE, "r") as f:
        creds = json.load(f)
    return username == creds["username"] and password == creds["password"]

# -----------------------------
# Login Page
# -----------------------------
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False

if not st.session_state["logged_in"]:
    st.title("🔐 Smart Stock Login Portal")

    with st.form("login_form"):
        username = st.text_input("👤 Username")
        password = st.text_input("🔑 Password", type="password")
        login_btn = st.form_submit_button("Login")

    if login_btn:
        if check_login(username, password):
            st.session_state["logged_in"] = True
            st.success("✅ Login successful!")
            st.rerun()
        else:
            st.error("❌ Invalid username or password. Try again.")

    st.stop()

# File uploader

uploaded_file = st.file_uploader("📂 Upload your sales CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("✅ File uploaded successfully!")
    
    # Normalize column names
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

  
    # Column mapping
   
    st.subheader("🧩 Map Columns (if auto-detection is incorrect)")
    date_col = st.selectbox("Select Date Column", df.columns, index=df.columns.get_loc('date') if 'date' in df.columns else 0)
    product_col = st.selectbox("Select Product Column", df.columns, index=df.columns.get_loc('product_id') if 'product_id' in df.columns else 0)
    sales_col = st.selectbox("Select Units Sold Column", df.columns, index=df.columns.get_loc('units_sold') if 'units_sold' in df.columns else 0)

    df.rename(columns={date_col:'date', product_col:'product_id', sales_col:'units_sold'}, inplace=True)
    
    
    # Backend processing
   
    with st.spinner("⏳ Generating forecasts and optimizing inventory..."):
        forecast_df = generate_forecasts(df, forecast_horizon=28)
        optimized_df = inventory_optimization_eoq(
            forecast_df,
            lead_time_days=7,
            Z=1.65,
            ordering_cost=50,
            holding_cost_per_unit=2
        )

    st.success("✅ Forecasting and optimization completed!")


    # Add Stock Alerts
   
    def stock_alert(row):
        if row['action'] == 'Restock':
            return "🔴 Stockout Risk"
        elif row['action'] == 'Reduce':
            return "🟡 Overstock Risk"
        else:
            return "🟢 Sufficient Stock"

    optimized_df['stock_alert'] = optimized_df.apply(stock_alert, axis=1)

  
    # Tabs
  
    tab1, tab2, tab3,tab4 = st.tabs(["📈 Demand Trends", "📦 Reorder Insights", "🌍 Regional Analysis","📊 Inventory KPIs"])

  
    # 1️. Demand Trends Tab
    
    with tab1:
        st.subheader("📅 Demand Forecast Trends")
        selected_pid = st.selectbox("Select Product to View Forecast", optimized_df['product_id'].unique())

        prod_df = optimized_df[optimized_df['product_id'] == selected_pid]
        fig = px.line(prod_df, x='date', y='forecast_units',
                      title=f"📈 Product {selected_pid} - 28-Day Demand Forecast",
                      markers=True, color_discrete_sequence=["#0078D7"])
        st.plotly_chart(fig, use_container_width=True)

        # Bar chart for top-selling products
        top_products = df.groupby('product_id')['units_sold'].sum().nlargest(10).reset_index()
        fig_top = px.bar(top_products, x='product_id', y='units_sold', color='units_sold',
                         title="🏆 Top 10 Selling Products", text='units_sold')
        st.plotly_chart(fig_top, use_container_width=True)

        # Pie chart for category distribution
        if 'product_type' in df.columns:
            cat_sales = df.groupby('product_type')['units_sold'].sum().reset_index()
            fig_pie = px.pie(cat_sales, names='product_type', values='units_sold',
                             title="🥧 Sales Share by Product Category")
            st.plotly_chart(fig_pie, use_container_width=True)


 
    # 2️. Reorder Insights Tab (Per Product Summary with Profit)
   
    with tab2:
        st.subheader("📦 Reorder Recommendations & EOQ Optimization (Per Product Summary)")

        # --- Aggregate per product ---
        product_summary = optimized_df.groupby('product_id').agg({
            'forecast_units': 'mean',
            'avg_daily_demand': 'mean',
            'safety_stock': 'mean',
            'reorder_point': 'mean',
            'EOQ': 'mean'
        }).reset_index()

        # --- Determine most frequent action per product ---
        dominant_action = (
            optimized_df.groupby('product_id')['action']
            .agg(lambda x: x.value_counts().index[0])
            .reset_index()
        )
        product_summary = product_summary.merge(dominant_action, on='product_id', how='left')

        # --- Add cost/selling price info if available ---
        if 'cost_per_unit' in df.columns and 'selling_price' in df.columns:
            price_info = df.groupby('product_id')[['cost_per_unit', 'selling_price']].mean().reset_index()
            product_summary = product_summary.merge(price_info, on='product_id', how='left')
        else:
            st.warning("⚠️ Columns `cost_per_unit` and `selling_price` not found — using defaults.")
            st.info("Enter default pricing values to apply to all products:")
            col_cost,col_price=st.columns(2)
            default_cost=col_cost.number_input("default cost per unit",min_value=1.0,value=70.0,step=1.0)
            default_price=col_price.number_input("default selling price per unit",min_value=1.0,value=100.0,step=1.0)
            product_summary['cost_per_unit'] = default_cost
            product_summary['selling_price'] = default_price
         # --- Discount Simulation ---
        st.markdown("### 💰 Apply Discount Simulation")
        discount = st.slider("Select Discount Percentage", 0, 50, 0, step=5)
        product_summary['discounted_price'] = product_summary['selling_price'] * (1 - discount / 100)
        
        # --- Calculate revenue, cost, profit & margin ---
        product_summary['total_revenue'] = product_summary['forecast_units'] * product_summary['selling_price']
        product_summary['total_cost'] = product_summary['forecast_units'] * product_summary['cost_per_unit']
        product_summary['estimated_profit'] = product_summary['total_revenue'] - product_summary['total_cost']
        product_summary['profit_margin_%'] = (
            (product_summary['estimated_profit'] / product_summary['total_revenue']) * 100
        ).round(2)
       
                # --- Visuals ---
        fig_profit = px.bar(
            product_summary.sort_values('estimated_profit', ascending=False).head(10),
            x='product_id', y='estimated_profit', color='profit_margin_%',
            title=f"💸 Top 10 Profitable Products after {discount}% Discount",
            text='profit_margin_%'
        )
        st.plotly_chart(fig_profit, use_container_width=True)
        # --- Add Stock Alert ---
        def stock_alert(row):
            if row['action'] == 'Restock':
                return "🔴 Stockout Risk"
            elif row['action'] == 'Reduce':
                return "🟡 Overstock Risk"
            else:
                return "🟢 Sufficient Stock"

        product_summary['stock_alert'] = product_summary.apply(stock_alert, axis=1)

        # --- Display Product Summary Table ---
        display_cols = [
            'product_id', 'forecast_units', 'avg_daily_demand', 'safety_stock', 'reorder_point',
            'EOQ', 'cost_per_unit', 'selling_price', 'total_revenue', 'estimated_profit',
            'profit_margin_%', 'action', 'stock_alert'
        ]
        st.dataframe(product_summary[display_cols], use_container_width=True)


        # --- Treemap of Product Actions ---
        action_summary = product_summary.groupby('action')['product_id'].count().reset_index()
        fig_treemap = px.treemap(
            action_summary,
            path=['action'], values='product_id',
            color='action',
            color_discrete_map={'Restock': 'red', 'Reduce': 'orange', 'Hold': 'green'},
            title="🧭 Inventory Action Distribution (Product Level)"
        )
        st.plotly_chart(fig_treemap, use_container_width=True)

        # --- Download CSV ---
        @st.cache_data
        def convert_df(df):
            return df.to_csv(index=False).encode('utf-8')

        csv = convert_df(product_summary[display_cols])
        st.download_button(
            "💾 Download Product-Level Recommendations CSV",
            csv,
            "product_level_recommendations.csv",
            "text/csv"
        )
    
    # 3. Regional analysis
    with tab3:
        st.subheader("🌍 Regional Sales Performance")

        region_col = None
        for col in ['region', 'city', 'location']:
            if col in df.columns:
                region_col = col
                break

        if region_col:
            region_sales = df.groupby(region_col)['units_sold'].sum().reset_index().sort_values('units_sold', ascending=False)
            fig_region = px.bar(region_sales, x=region_col, y='units_sold', color='units_sold',
                                title="🏙️ Total Sales by Region/City", text='units_sold')
            st.plotly_chart(fig_region, use_container_width=True)

            if 'latitude' in df.columns and 'longitude' in df.columns:
                st.map(df[['latitude', 'longitude']])
        else:
            st.info("📍 No region/city column found. Add a 'region' or 'city' column to view regional insights.")

  
    # 3️. Inventory KPIs Tab
    
    with tab4:
        st.subheader("📊 Inventory Performance Dashboard")

        avg_forecast = optimized_df['forecast_units'].mean()
        avg_eoq = optimized_df['EOQ'].mean()
        restock_count = (optimized_df['action'] == 'Restock').sum()
        hold_count = (optimized_df['action'] == 'Hold').sum()
        reduce_count = (optimized_df['action'] == 'Reduce').sum()

        stockout_count = (optimized_df['stock_alert'] == "🔴 Stockout Risk").sum()
        overstock_count = (optimized_df['stock_alert'] == "🟡 Overstock Risk").sum()

        col1, col2, col3, col4, col5, col6, col7 = st.columns(7)
        col1.metric("📦 Avg Forecast", f"{avg_forecast:.2f}")
        col2.metric("🧮 Avg EOQ", f"{avg_eoq:.2f}")
        col3.metric("🟥 Restock", restock_count)
        col4.metric("🟩 Hold", hold_count)
        col5.metric("🟨 Reduce", reduce_count)
        col6.metric("⚠️ Stockouts", stockout_count)
        col7.metric("📦 Overstocks", overstock_count)

        # ABC Classification Chart
        total_sales = df.groupby('product_id')['units_sold'].sum().sort_values(ascending=False).reset_index()
        total_sales['cum_percentage'] = 100 * total_sales['units_sold'].cumsum() / total_sales['units_sold'].sum()

        def classify(x):
            if x <= 70: return 'A'
            elif x <= 90: return 'B'
            else: return 'C'

        total_sales['ABC_Category'] = total_sales['cum_percentage'].apply(classify)

        fig_abc = px.bar(total_sales, x='product_id', y='units_sold', color='ABC_Category',
                         title="🧮 ABC Classification of Products",
                         color_discrete_map={'A': 'green', 'B': 'orange', 'C': 'red'})
        st.plotly_chart(fig_abc, use_container_width=True)

else:
    st.info("📂 Upload a retailer sales CSV to generate forecasts and inventory recommendations.")


# Dashboard Styling

st.markdown("""
<style>
    .stApp {
        background-color: #f8fafc;
    }
    h1, h2, h3 {
        color: #0078D7;
    }
    div[data-testid="stMetricValue"] {
        color: #0078D7;
    }
</style>
""", unsafe_allow_html=True)
import json

if st.button("🚪 Logout"):
    st.session_state["logged_in"] = False
    st.rerun()


