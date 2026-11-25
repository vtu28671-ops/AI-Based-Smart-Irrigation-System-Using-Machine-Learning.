import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import utils  # noqa: E402

st.set_page_config(page_title="Model Performance", page_icon="📈", layout="wide")


@st.cache_data
def get_data():
    return utils.load_data()


@st.cache_resource
def get_model():
    return utils.load_model()


@st.cache_data
def get_rf(_n):
    return utils.evaluate_model(get_model(), get_data())


@st.cache_data
def get_tree(_n):
    return utils.evaluate_decision_tree(get_model(), get_data())


df, model = get_data(), get_model()
rf, dt = get_rf(len(df)), get_tree(len(df))

st.title("📈 Model Performance")
st.write("Results on the %d test records (20%% stratified split, random_state = 42)." % rf["n_test"])

a, b, c = st.columns(3)
a.metric("Accuracy", "%.2f%%" % rf["accuracy"])
b.metric("Balanced accuracy", "%.2f%%" % rf["balanced_accuracy"])
c.metric("High-class recall", "%.1f%%" % rf["high_recall"])

st.subheader("Class-wise metrics")
st.dataframe(rf["report"].round(3), use_container_width=True)

st.subheader("Confusion matrix")
cm = rf["confusion"]
fig, ax = plt.subplots(figsize=(5, 4))
im = ax.imshow(cm.values, cmap="Blues")
ax.set_xticks(range(3))
ax.set_yticks(range(3))
ax.set_xticklabels(utils.CLASS_ORDER)
ax.set_yticklabels(utils.CLASS_ORDER)
ax.set_xlabel("Predicted class")
ax.set_ylabel("Actual class")
for i in range(3):
    for j in range(3):
        ax.text(j, i, str(cm.values[i, j]), ha="center", va="center",
                color="white" if cm.values[i, j] > cm.values.max() / 2 else "black")
fig.colorbar(im, fraction=0.046)
fig.tight_layout()
st.pyplot(fig)
plt.close(fig)

st.subheader("Feature importance")
imp = utils.grouped_importance(model).iloc[::-1]
fig, ax = plt.subplots(figsize=(7, 5.5))
ax.barh([utils.label_for(i) for i in imp.index], imp.values, color="#2e7d32")
ax.set_xlabel("Importance (one-hot columns summed per feature)")
fig.tight_layout()
st.pyplot(fig)
plt.close(fig)

st.subheader("Random Forest vs Decision Tree")
comp = pd.DataFrame({
    "Model": ["Random Forest (300 trees, High weight 8)", "Decision Tree (baseline)"],
    "Accuracy %": [rf["accuracy"], dt["accuracy"]],
    "Balanced accuracy %": [rf["balanced_accuracy"], dt["balanced_accuracy"]],
    "High recall %": [rf["high_recall"], dt["high_recall"]],
}).round(2)
st.dataframe(comp, use_container_width=True, hide_index=True)
st.info(
    "On this dataset a single Decision Tree scores at least as well as the Random "
    "Forest. This suggests the irrigation classes are largely separable by a few "
    "features such as soil moisture, crop growth stage and rainfall. The Random "
    "Forest is kept because an ensemble is usually more stable on noisier "
    "real-world data, but this should be confirmed on field data."
)

st.subheader("Class-weight tuning (validation split)")
st.table(pd.DataFrame({
    "Class weight": ["balanced_subsample", "High = 3", "High = 5", "High = 8 (selected)"],
    "Accuracy %": [97.4, 98.45, 98.5, 98.5],
    "High recall": [0.40, 0.72, 0.70, 0.72],
}))
st.caption("Values come from running tune_model.py. Re-run it if you change the data or the model.")
