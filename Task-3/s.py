import streamlit as st
import pandas as pd
import pickle
import plotly.express as px

# =========================
# LOAD MODEL + DATA
# =========================
@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        return pickle.load(f)

@st.cache_data
def load_data():
    return pd.read_csv("train.csv")

model = load_model()
df = load_data()

st.set_page_config(page_title="Titanic Survival App", layout="wide")

# =========================
# SIDEBAR NAVIGATION
# =========================
st.sidebar.title("🚢 Titanic App")
page = st.sidebar.radio("Go to", ["Prediction", "Dashboard"])

# =========================
# PAGE 1: PREDICTION
# =========================
if page == "Prediction":

    st.title("🚢 Titanic Survival Prediction App")
    st.write("Enter passenger details to predict survival probability")

    col1, col2 = st.columns(2)

    with col1:
        pclass = st.selectbox("Passenger Class", [1, 2, 3])
        sex = st.selectbox("Sex", ["male", "female"])
        age = st.slider("Age", 0, 80, 25)

    with col2:
        fare = st.number_input("Fare", 0.0, 500.0, 50.0)
        sibsp = st.number_input("Siblings/Spouses aboard", 0, 10, 0)
        parch = st.number_input("Parents/Children aboard", 0, 10, 0)
        embarked = st.selectbox("Port of Embarkation", ["C", "Q", "S"])

    if st.button("Predict"):
        input_data = pd.DataFrame({
            'Pclass': [pclass],
            'Sex': [sex],
            'Age': [age],
            'Fare': [fare],
            'SibSp': [sibsp],
            'Parch': [parch],
            'Embarked': [embarked]
        })

        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]

        st.subheader("Result:")
        if prediction == 1:
            st.success(f"🎉 Survived (Probability: {probability:.2f})")
        else:
            st.error(f"💀 Did Not Survive (Probability: {probability:.2f})")

        # Progress bar (advanced feature)
        st.progress(float(probability))

# =========================
# PAGE 2: DASHBOARD
# =========================
elif page == "Dashboard":

    st.title("📊 Titanic Data Dashboard")

    # Show dataset
    if st.checkbox("Show Raw Dataset"):
        st.dataframe(df)

    # =====================
    # FILTERS
    # =====================
    st.sidebar.subheader("Filters")
    selected_class = st.sidebar.multiselect(
        "Select Class", df["Pclass"].unique(), default=df["Pclass"].unique()
    )
    selected_gender = st.sidebar.multiselect(
        "Select Gender", df["Sex"].unique(), default=df["Sex"].unique()
    )

    filtered_df = df[
        (df["Pclass"].isin(selected_class)) &
        (df["Sex"].isin(selected_gender))
    ]

    st.write(f"Filtered Data Shape: {filtered_df.shape}")

    # =====================
    # VISUALIZATION 1
    # =====================
    st.subheader("Survival Count")
    fig1 = px.bar(filtered_df, x="Survived", color="Survived")
    st.plotly_chart(fig1)

    # =====================
    # VISUALIZATION 2
    # =====================
    st.subheader("Survival by Gender")
    fig2 = px.histogram(filtered_df, x="Sex", color="Survived", barmode="group")
    st.plotly_chart(fig2)

    # =====================
    # VISUALIZATION 3
    # =====================
    st.subheader("Age Distribution")
    fig3 = px.histogram(filtered_df, x="Age", nbins=30, color="Survived")
    st.plotly_chart(fig3)

    # =====================
    # VISUALIZATION 4
    # =====================
    st.subheader("Fare vs Age (Colored by Survival)")
    fig4 = px.scatter(filtered_df, x="Fare", y="Age", color="Survived")
    st.plotly_chart(fig4)
