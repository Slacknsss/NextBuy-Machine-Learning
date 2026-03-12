import joblib
import calendar
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.markdown(
    """
    <style>
        div[data-testid='stVerticalBlock'] {
            border-radius: 20px;
        }
        h1 {
            text-align: center;
            font-size: 3.5rem;
        }
    </style>
""",
    unsafe_allow_html=True,
)

df = pd.read_csv("./cleaned_dataset.csv")
df_gare = df["Gare de départ"].drop_duplicates()

internationalgare = (
    "BARCELONA",
    "MADRID",
    "STUTTGART",
    "LAUSANNE",
    "GENEVE",
    "ITALIE",
    "ZURICH",
    "FRANCFORT",
)


@st.cache_resource
def load_my_model():
    bundle = joblib.load("model.joblib")
    model = bundle["model"]
    year_min = bundle["year_min"]
    return (model, year_min)


model, year_min = load_my_model()


def predict_delay(
    model,
    depart: str,
    arrivee: str,
    duree: int | float,
    month: int,
    Year: int,
    national: bool = True,
    international: bool = False,
):
    """Return prediction of delay in minutes."""

    df_predict = pd.DataFrame(
        [
            {
                "Gare de départ": depart,
                "Gare d'arrivée": arrivee,
                "Durée moyenne du trajet": duree,
                "Is_Service_National": national,
                "Is_Service_International": international,
                "month": month,
                "Year": Year,
            }
        ]
    )

    df_predict["month_sin"] = np.sin(2 * np.pi * (df_predict["month"] - 1) / 12)
    df_predict["month_cos"] = np.cos(2 * np.pi * (df_predict["month"] - 1) / 12)
    df_predict["route"] = (
        df_predict["Gare de départ"] + " -> " + df_predict["Gare d'arrivée"]
    )
    df_predict["year_idx"] = df_predict["Year"] - year_min

    return float(model.predict(df_predict)[0])


st.set_page_config(page_title="Tardis Dashboard", page_icon="📊", layout="wide")
st.title("TARDIS")

with st.container(border=True):
    left, right = st.columns([1, 2])
    with left:
        st.markdown("### Analysis")
        st.markdown("---")
        st.markdown(
            "Average delayed trains by month based on the dataset given by the SCNF on the 2018-2025 period, peaking in the summer due to the amount of trains."
        )
    df_filtered = df
    df_filtre = (
        df_filtered.groupby("month")["Nombre de trains en retard au départ"]
        .agg(["mean", "max", "min", "median", "sum"])
        .reset_index()
    )
    df_filtre = df_filtre.sort_values(by="month")
    X_array = [calendar.month_name[month_number] for month_number in df_filtre["month"]]
    Y_array = df_filtre["mean"]
    with right:
        fig2, ax2 = plt.subplots(figsize=(10, 4))
        ax2.set_title("Numbers of trains delayed at departure per month")
        ax2.plot(X_array, Y_array, color="orange", alpha=0.25)
        ax2.scatter(X_array, Y_array, color="orange", s=50)
        ax2.set_xticklabels(X_array, rotation=45, ha="right")
        st.pyplot(fig2, width="stretch")

with st.container(border=True):
    left, right = st.columns([2, 1])
    df_filtered = df
    df_filtre = (
        df_filtered.groupby("Gare de départ")[
            "Retard moyen des trains en retard à l'arrivée"
        ]
        .agg(["mean", "max", "min", "median", "sum"])
        .reset_index()
    )
    df_filtre = df_filtre.sort_values(by="mean", ascending=False)
    df_filtre = df_filtre.head(20)
    X_array = df_filtre["Gare de départ"]
    Y_array = df_filtre["mean"]
    with left:
        fig2, ax2 = plt.subplots(figsize=(15, 6))
        ax2.bar(X_array, Y_array, color="blue")
        ax2.set_xlabel("Departure station")
        ax2.set_ylabel("Average delay (min)")
        ax2.set_title(
            "Average delay for the top 20 train stations based on the number of trains delayed at arrival"
        )
        ax2.set_xticks(range(len(X_array)))
        wrapped_labels = [
            (
                label[:bp] + "\n" + label[bp + 1 :]
                if (bp := label.rfind(" ", 0, 15)) != -1
                else label[:15] + "\n" + label[15:]
            )
            if len(label) > 15
            else label
            for label in X_array
        ]
        ax2.set_xticklabels(wrapped_labels, fontsize=8, ha="right", rotation=50)
        fig2.tight_layout()
        st.pyplot(fig2, width="stretch")
    with right:
        st.markdown("### Top Stations")
        st.markdown("---")
        st.markdown(
            "Top 20 stations by average arrival delay based on the dataset given by the SCNF on the 2018-2025 period."
        )

with st.container(border=True):
    left, right = st.columns([1, 2])
    with left:
        st.markdown("### Station Comparison")
        st.markdown("---")
        st.markdown("Compare two stations of your choice")
        st.markdown(
            "(the red line is the average delay at arrival of ll the train stations)"
        )
        st.markdown("")
        gare1 = st.selectbox(
            "Departure Station", df_gare, key="gare1_compare", width=250
        )
        if gare1 in list(df_gare):
            df_gare2 = list(df_gare)
            df_gare2.remove(gare1)
        gare2 = st.selectbox(
            "Arrival station", df_gare2, index=1, key="gare2_compare", width=250
        )
    df_filtered = df
    df_filtre = (
        df_filtered[df_filtered["Gare de départ"].isin([gare1, gare2])]
        .groupby("Gare de départ")["Retard moyen des trains en retard à l'arrivée"]
        .mean()
        .reset_index()
    )
    df_filtre = df_filtre.sort_values(
        by="Retard moyen des trains en retard à l'arrivée", ascending=False
    )
    X_array = df_filtre["Gare de départ"]
    Y_array = df_filtre["Retard moyen des trains en retard à l'arrivée"]
    with right:
        fig2, ax2 = plt.subplots(figsize=(12, 6))
        ax2.bar(range(len(X_array)), Y_array, color="green", width=0.8)
        ax2.set_xlabel("Departure station")
        ax2.set_ylabel("Average delay (min)")
        ax2.set_title("Average delay comparison")
        ax2.axhline(
            y=df["Retard moyen des trains en retard à l'arrivée"].mean(),
            color="red",
            linestyle="-",
            linewidth=2,
            label="Average",
        )
        ax2.set_xticks(range(len(X_array)))
        ax2.set_xticklabels(X_array, fontsize=10)
        fig2.tight_layout()
        st.pyplot(fig2, width="stretch")

with st.container(border=True):
    st.markdown("## Train Delay Predictor")
    st.markdown("**Selected Model:** Extra Trees Regressor")
    st.markdown("---")

    st.markdown("### Make a Prediction")
    col_form1, col_form2 = st.columns([1, 1])

    with col_form1:
        dateinput = st.date_input("Departure Date", width=300)
        garedeinput = st.selectbox("Departure Station", df_gare, width=300)

        if garedeinput in list(df_gare):
            df_gare2 = list(
                df[df["Gare de départ"] == garedeinput][
                    "Gare d'arrivée"
                ].drop_duplicates()
            )

        garearrinput = st.selectbox("Arrival Station", df_gare2, width=300)
        dureedetruc = st.number_input(
            "Journey Duration (minutes)", width=300, value=0, min_value=0
        )

        st.markdown("")
        submit_button = st.button("Predict Delay", width="stretch")

        if submit_button:
            if garearrinput in internationalgare or garedeinput in internationalgare:
                natio = False
                internatio = True
            else:
                natio = True
                internatio = False

            prediction = round(
                predict_delay(
                    model,
                    garedeinput,
                    garearrinput,
                    dureedetruc,
                    dateinput.month,
                    dateinput.year,
                    natio,
                    internatio,
                ),
                2,
            )
            prediction_m, prediction_s = int(prediction), int((prediction % 1) * 60)

            st.success(f"Predicted Delay: {prediction_m} min {prediction_s:02d} sec")

    st.markdown("---")
    st.markdown("### Model Benchmark Comparison")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.image("./assets/Model_MAE_SEC.png")

    with col2:
        st.image("./assets/Model_RMSE_SEC.png")

    with col3:
        st.image("./assets/Model_R²_SEC.png")

    st.markdown(
        "In this context the score MAE is the average error in minutes of the model compared to the reality"
    )
    st.markdown(
        "The score RMSE is derived from the MAE score but the mistakes have more influence on the score based on their size"
    )
    st.markdown(
        "the R² score is a statistical measure that represents the proportion of the variance in a dependent variable that is explained by one or more independent variables in a regression model."
    )