import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import utils  # noqa: E402

st.set_page_config(page_title="Analytics Dashboard", page_icon="📊", layout="wide")


@st.cache_data
def get_data():
    return utils.load_data()


df = get_data()
colours = {"Low": "#2e7d32", "Medium": "#ef6c00", "High": "#c62828"}

st.title("📊 Smart Irrigation Analytics Dashboard")
st.write("An overview of the agricultural dataset used to train the model.")

c1, c2, c3 = st.columns(3)
c1.metric("Total Records", f"{len(df):,}")
c2.metric("Input Features", str(df.shape[1] - 1))
c3.metric("Irrigation Classes", str(df[utils.TARGET].nunique()))

st.divider()
st.subheader("Irrigation Need Distribution")
counts = df[utils.TARGET].value_counts().reindex(utils.CLASS_ORDER)
fig, ax = plt.subplots(figsize=(6, 3.6))
bars = ax.bar(counts.index, counts.values, color=[colours[c] for c in counts.index])
for b, v in zip(bars, counts.values):
    ax.text(b.get_x() + b.get_width() / 2, v + 60,
            "%d (%.2f%%)" % (v, v / len(df) * 100), ha="center", fontsize=9)
ax.set_xlabel("Irrigation need class")
ax.set_ylabel("Number of records")
ax.set_ylim(0, counts.max() * 1.15)
fig.tight_layout()
st.pyplot(fig)
plt.close(fig)
st.caption("The High class is rare, which is why class-wise metrics matter more than accuracy alone.")

st.divider()
st.subheader("Irrigation Need by Category")
cat_cols = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c]) and c != utils.TARGET]
group = st.selectbox("Group by", cat_cols, format_func=utils.label_for)
share = pd.crosstab(df[group], df[utils.TARGET], normalize="index")[utils.CLASS_ORDER] * 100
fig, ax = plt.subplots(figsize=(7, 3.8))
share.plot(kind="bar", stacked=True, ax=ax, color=[colours[c] for c in utils.CLASS_ORDER])
ax.set_ylabel("Share of records (%)")
ax.set_xlabel(utils.label_for(group))
ax.legend(title="Need", bbox_to_anchor=(1.01, 1), loc="upper left")
plt.xticks(rotation=0)
fig.tight_layout()
st.pyplot(fig)
plt.close(fig)

st.divider()
st.subheader("Feature Explorer")
num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
feat = st.selectbox("Numeric feature", num_cols, format_func=utils.label_for)
fig, ax = plt.subplots(figsize=(7, 3.6))
for cls in utils.CLASS_ORDER:
    ax.hist(df.loc[df[utils.TARGET] == cls, feat], bins=25, alpha=0.55,
            density=True, label=cls, color=colours[cls])
ax.set_xlabel(utils.label_for(feat))
ax.set_ylabel("Density")
ax.legend()
fig.tight_layout()
st.pyplot(fig)
plt.close(fig)
avg = df.groupby(utils.TARGET)[feat].agg(["mean", "min", "max"]).reindex(utils.CLASS_ORDER).round(2)
st.dataframe(avg, use_container_width=True)

st.divider()
st.subheader("Dataset Preview")
st.dataframe(df.head(10), use_container_width=True)
st.download_button("Download cleaned dataset as CSV", df.to_csv(index=False).encode("utf-8"),
                   file_name="irrigation_dataset_clean.csv", mime="text/csv")
st.info("This dashboard displays dataset information. It does not show live farm sensor readings.")
