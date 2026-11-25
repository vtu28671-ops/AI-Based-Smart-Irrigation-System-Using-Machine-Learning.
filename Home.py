import streamlit as st

import utils

st.set_page_config(page_title="Smart Irrigation System", page_icon="🌱", layout="wide")


@st.cache_data
def get_data():
    return utils.load_data()


@st.cache_resource
def get_model():
    return utils.load_model()


@st.cache_data
def get_scores():
    result = utils.evaluate_model(get_model(), get_data())
    return {k: result[k] for k in ("accuracy", "balanced_accuracy", "high_recall", "n_test")}


model = get_model()
scores = get_scores()

st.title("🌱 AI-Based Smart Irrigation System")
st.subheader("Machine Learning Based Irrigation Prediction")
st.write(
    "A web-based decision-support application that uses Machine Learning to "
    "classify the irrigation need of a field as Low, Medium or High from soil, "
    "crop and environmental information."
)

st.divider()
st.header("About the Project")
st.write(
    "The system uses a Random Forest Classifier trained on 19 agricultural "
    "and environmental features. For every prediction it shows the model's "
    "confidence, a recommendation, an explanation based on the dataset, and a "
    "what-if view that shows how the prediction changes when one input changes."
)

st.header("System Workflow")
st.info(
    "Farmer Input → Validation → Data Preprocessing → Machine Learning "
    "Prediction → Confidence → Recommendation → Explanation → Dashboard"
)

st.header("Key Features")
c1, c2 = st.columns(2)
with c1:
    st.subheader("🌾 Irrigation Prediction")
    st.write("Predicts Low, Medium or High irrigation need with class probabilities.")
    st.subheader("💧 Smart Recommendations")
    st.write("Recommendation text plus notes computed from the dataset for your inputs.")
    st.subheader("🔍 Prediction Explanation")
    st.write("Feature importance, comparison with the predicted class and a what-if chart.")
with c2:
    st.subheader("📊 Analytics Dashboard")
    st.write("Class distribution, crop-wise irrigation need and a feature explorer.")
    st.subheader("📈 Model Performance")
    st.write("Confusion matrix, class-wise metrics and a Decision Tree comparison.")
    st.subheader("🧪 Sample Records")
    st.write("Load a real Low, Medium or High record from the dataset in one click.")

st.divider()
st.header("Machine Learning Model")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Algorithm", "Random Forest")
m2.metric("Number of Trees", str(model.named_steps["classifier"].n_estimators))
m3.metric("Test Accuracy", "%.1f%%" % scores["accuracy"])
m4.metric("Balanced Accuracy", "%.1f%%" % scores["balanced_accuracy"])

st.warning(
    "The High class is rare (about 3%% of records) and the model finds %.0f%% of "
    "High cases on the test data. Treat a Low or Medium prediction with a "
    "noticeable High probability as a caution, not a guarantee." % scores["high_recall"]
)
st.caption(
    "Developed as a software-based academic project. Predictions are based on "
    "the available dataset and model. The application does not read live sensors "
    "and does not control irrigation equipment."
)
