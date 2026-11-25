import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import utils  # noqa: E402

st.set_page_config(page_title="Irrigation Prediction", page_icon="🌱", layout="wide")


@st.cache_data
def get_data():
    return utils.load_data()


@st.cache_resource
def get_model():
    return utils.load_model()


df = get_data()
model = get_model()
features = list(model.feature_names_in_)

st.title("🌱 Irrigation Need Prediction")
st.write("Enter agricultural details, or start from a real sample record, to predict the irrigation requirement.")

choice = st.radio(
    "Start from",
    ["Median values", "Sample: Low need", "Sample: Medium need", "Sample: High need"],
    horizontal=True,
)
if choice == "Median values":
    defaults = utils.default_record(df, features)
else:
    defaults = utils.sample_record(df, features, choice.split(": ")[1].split()[0])

st.divider()

with st.form("irrigation_form"):
    st.subheader("Enter Agricultural Details")
    inputs = {}
    col1, col2 = st.columns(2)
    for i, column in enumerate(features):
        with (col1 if i % 2 == 0 else col2):
            key = "%s|%s" % (column, choice)  # new widget when the sample changes
            if pd.api.types.is_numeric_dtype(df[column]):
                inputs[column] = st.number_input(
                    utils.label_for(column), value=float(defaults[column]),
                    format="%.2f", help=utils.help_for(column), key=key)
            else:
                options = sorted(df[column].dropna().astype(str).unique().tolist())
                inputs[column] = st.selectbox(
                    utils.label_for(column), options,
                    index=options.index(str(defaults[column])), key=key)
    submitted = st.form_submit_button("Predict Irrigation Need", use_container_width=True)

if submitted:
    errors = utils.validate_inputs(inputs)
    if errors:
        st.session_state.pop("last", None)
        st.error("Invalid Input! Please correct the following:")
        for e in errors:
            st.write("❌", e)
        st.stop()
    label, probs = utils.predict(model, inputs)
    st.session_state["last"] = {"inputs": dict(inputs), "label": label, "probs": probs}
    history = st.session_state.setdefault("history", [])
    history.append({
        "Time": datetime.now().strftime("%H:%M:%S"),
        "Prediction": label,
        "P(Low) %": round(probs["Low"] * 100, 1),
        "P(Medium) %": round(probs["Medium"] * 100, 1),
        "P(High) %": round(probs["High"] * 100, 1),
        "Soil Moisture": round(inputs["Soil_Moisture"], 2),
        "Rainfall (mm)": round(inputs["Rainfall_mm"], 2),
        "Crop": inputs["Crop_Type"],
        "Growth Stage": inputs["Crop_Growth_Stage"],
    })

last = st.session_state.get("last")
if last:
    inputs, label, probs = last["inputs"], last["label"], last["probs"]

    st.divider()
    st.subheader("Prediction Result")
    st.metric("Irrigation Need", label)

    st.write("**Model confidence**")
    pc = st.columns(3)
    for col, cls in zip(pc, utils.CLASS_ORDER):
        col.metric(cls, "%.1f%%" % (probs[cls] * 100))
        col.progress(min(max(probs[cls], 0.0), 1.0))
    if label != "High" and probs["High"] >= utils.HIGH_WARN:
        st.warning(
            "Caution: the model also gives a %.0f%% chance of High irrigation need. "
            "High is the hardest class for the model, so check soil moisture and "
            "rainfall before deciding." % (probs["High"] * 100))
    st.info("This prediction is based on the details entered and the trained Random "
            "Forest model. It does not calculate an exact water quantity.")

    st.divider()
    st.subheader("💧 Irrigation Recommendation")
    kind, message = utils.CLASS_MESSAGES[label]
    getattr(st, kind)(message)
    tips = utils.recommendation_tips(df, inputs, label)
    if tips:
        st.write("**Notes computed from the dataset for your inputs:**")
        for t in tips:
            st.write("•", t)

    st.divider()
    st.subheader("🔍 Prediction Explanation")
    st.write("Overall, the model relies most on the features below. This shows what the "
             "model uses in general, not the contribution for this single prediction.")
    imp = utils.grouped_importance(model).head(8).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.barh([utils.label_for(i) for i in imp.index], imp.values, color="#2e7d32")
    ax.set_xlabel("Feature importance")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    top = [f for f in utils.grouped_importance(model).index
           if pd.api.types.is_numeric_dtype(df[f])][:5]
    st.write("**How your inputs compare with the dataset**")
    st.dataframe(utils.class_comparison(df, inputs, label, top), use_container_width=True)
    st.write("**Records with the same stage / mulching setting**")
    st.dataframe(utils.categorical_context(df, inputs), use_container_width=True)

    with st.expander("What-if: change one input and see the probabilities"):
        feat = st.selectbox("Feature to vary", utils.WHATIF_FEATURES,
                            format_func=utils.label_for)
        curve = utils.whatif_curve(model, inputs, feat, df)
        fig, ax = plt.subplots(figsize=(7, 3.6))
        for cls, colour in zip(utils.CLASS_ORDER, ["#2e7d32", "#ef6c00", "#c62828"]):
            ax.plot(curve.index, curve[cls] * 100, label=cls, color=colour)
        ax.axvline(inputs[feat], color="grey", linestyle="--", linewidth=1)
        ax.set_xlabel(utils.label_for(feat))
        ax.set_ylabel("Probability (%)")
        ax.legend()
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        st.caption("All other inputs stay as entered. The dashed line is your value.")

    with st.expander("View entered agricultural details"):
        st.dataframe(pd.DataFrame([inputs]), use_container_width=True)

hist = st.session_state.get("history")
if hist:
    st.divider()
    st.subheader("🕘 Prediction history (this session)")
    hdf = pd.DataFrame(hist)
    st.dataframe(hdf, use_container_width=True)
    st.download_button("Download history as CSV", hdf.to_csv(index=False).encode("utf-8"),
                       file_name="prediction_history.csv", mime="text/csv")
