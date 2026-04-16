import streamlit as st
import pandas as pd
import pulp
import matplotlib.pyplot as plt

st.set_page_config(page_title="Optimization App", layout="wide")

st.title("📈 Business Optimization Dashboard")

# Upload CSV
file = st.file_uploader("Upload your dataset", type=["csv"])

if file:
    df = pd.read_csv(file)

    st.subheader("📊 Raw Dataset")
    st.dataframe(df)

    # ---------------- FIX COLUMN NAMES ---------------- #
    df.columns = df.columns.str.strip()

    # Auto mapping (handles any dataset)
    col_map = {}

    for col in df.columns:
        c = col.lower()

        if "product" in c or "item" in c:
            col_map['Product'] = col
        elif "profit" in c or "revenue" in c:
            col_map['Profit'] = col
        elif "material" in c or "raw" in c:
            col_map['Material'] = col
        elif "labour" in c or "labor" in c or "worker" in c:
            col_map['Labour'] = col
        elif "demand" in c:
            col_map['Max_Demand'] = col

    # Rename safely
    df = df.rename(columns={
        col_map.get('Product', ''): 'Product',
        col_map.get('Profit', ''): 'Profit',
        col_map.get('Material', ''): 'Material',
        col_map.get('Labour', ''): 'Labour',
        col_map.get('Max_Demand', ''): 'Max_Demand'
    })

    # Check required columns
    required = ['Product', 'Profit', 'Material', 'Labour', 'Max_Demand']
    missing = [col for col in required if col not in df.columns]

    if missing:
        st.error(f"❌ Missing required columns: {missing}")
        st.info("👉 Your CSV must contain something like: Product, Profit, Material, Labour, Demand")
        st.stop()

    st.subheader("✅ Cleaned Dataset")
    st.dataframe(df)

    # ---------------- SIDEBAR CONTROLS ---------------- #
    st.sidebar.header("⚙️ Constraints")

    material_limit = st.sidebar.slider("Material Limit", 50, 500, 200)
    labour_limit = st.sidebar.slider("Labour Limit", 50, 500, 150)

    # ---------------- OPTIMIZATION ---------------- #
    if st.button("🚀 Run Optimization"):

        model = pulp.LpProblem("MaxProfit", pulp.LpMaximize)

        # Decision Variables
        x = pulp.LpVariable.dicts("Prod", df['Product'],
                                 lowBound=0, cat='Integer')

        # Objective Function
        model += pulp.lpSum(df.loc[i, 'Profit'] * x[df.loc[i, 'Product']]
                            for i in df.index)

        # Constraints
        model += pulp.lpSum(df.loc[i, 'Material'] * x[df.loc[i, 'Product']]
                            for i in df.index) <= material_limit

        model += pulp.lpSum(df.loc[i, 'Labour'] * x[df.loc[i, 'Product']]
                            for i in df.index) <= labour_limit

        # Demand constraint
        for i in df.index:
            model += x[df.loc[i, 'Product']] <= df.loc[i, 'Max_Demand']

        # Solve
        model.solve()

        # Results
        df["Production"] = [x[p].value() for p in df['Product']]
        df["Total Profit"] = df["Production"] * df["Profit"]

        total_profit = pulp.value(model.objective)

        # ---------------- OUTPUT ---------------- #
        st.success(f"💰 Total Profit: {total_profit}")

        st.subheader("📋 Optimized Results")
        st.dataframe(df)

        # ---------------- VISUALIZATION ---------------- #
        st.subheader("📊 Production Plan")
        fig, ax = plt.subplots()
        ax.bar(df["Product"], df["Production"])
        ax.set_xlabel("Product")
        ax.set_ylabel("Production Quantity")
        st.pyplot(fig)

        # ---------------- DOWNLOAD ---------------- #
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("⬇️ Download Results", csv, "results.csv")

        # ---------------- WHAT-IF ANALYSIS ---------------- #
        st.subheader("🔍 What-if Analysis")

        new_material = st.slider("Try New Material Limit", 50, 500, 250)

        if st.button("Re-run Scenario"):
            st.info(f"👉 Increasing material to {new_material} may increase profit.")