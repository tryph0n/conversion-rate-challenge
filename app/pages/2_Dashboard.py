"""Interactive dashboard for conversion rate analysis."""

import sys
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Dashboard", layout="wide")

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config import METADATA_FILE, TARGET, TRAIN_FILE  # noqa: E402


# ------------------------------------------------------------------
# Data loading
# ------------------------------------------------------------------
@st.cache_data
def load_data() -> pd.DataFrame:
    """Load training data from CSV."""
    return pd.read_csv(TRAIN_FILE)


@st.cache_data
def load_metadata() -> dict:
    """Load model metadata from joblib artifact."""
    return joblib.load(METADATA_FILE)


if not TRAIN_FILE.exists():
    st.error(
        "The Dashboard needs `data/input/conversion_data_train.csv`, which is not "
        "included in this repository. See the README, section Data."
    )
    st.stop()

data = load_data()
metadata = load_metadata()

# ------------------------------------------------------------------
# Sidebar filters
# ------------------------------------------------------------------
st.sidebar.header("Filters")

all_countries = sorted(data["country"].unique().tolist())
selected_countries = st.sidebar.multiselect(
    "Country",
    options=all_countries,
    default=all_countries,
)

all_sources = sorted(data["source"].unique().tolist())
selected_sources = st.sidebar.multiselect(
    "Source",
    options=all_sources,
    default=all_sources,
)

user_type_options = {"New user": 1, "Returning": 0}
selected_user_types = st.sidebar.multiselect(
    "User type",
    options=list(user_type_options.keys()),
    default=list(user_type_options.keys()),
)

# ------------------------------------------------------------------
# Apply filters
# ------------------------------------------------------------------
df = data.copy()
df = df[df["country"].isin(selected_countries)]
df = df[df["source"].isin(selected_sources)]

# Filter by user type
df = df[df["new_user"].isin([user_type_options[t] for t in selected_user_types])]

if df.empty:
    st.warning("No data matches the current filters. Please adjust the sidebar selections.")
    st.stop()

# ------------------------------------------------------------------
# Page title
# ------------------------------------------------------------------
st.title("Conversion Rate Dashboard")

# ------------------------------------------------------------------
# KPI row
# ------------------------------------------------------------------
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total visitors", f"{len(df):,}")

with col2:
    conv_rate = df[TARGET].mean() * 100
    st.metric("Conversion rate", f"{conv_rate:.2f}%")

with col3:
    best_f1 = metadata.get("val_f1_optimized", 0)
    st.metric("Best model F1", f"{best_f1:.4f}")

with col4:
    best_thresh = metadata.get("best_threshold", 0.5)
    st.metric("Optimal threshold", f"{best_thresh:.3f}")

# ------------------------------------------------------------------
# Tabs
# ------------------------------------------------------------------
tab_explore, tab_model = st.tabs(["Data Exploration", "Model Results"])

# ==================================================================
# Tab 1 -- Data Exploration
# ==================================================================
with tab_explore:
    # -- Conversion rate by country --
    st.subheader("Conversion rate by country")
    country_stats = (
        df.groupby("country")[TARGET]
        .mean()
        .reset_index()
        .rename(columns={TARGET: "conversion_rate"})
    )
    country_stats["conversion_rate"] *= 100
    fig_country = px.bar(
        country_stats,
        x="country",
        y="conversion_rate",
        text_auto=".2f",
        labels={
            "country": "Country",
            "conversion_rate": "Conversion rate (%)",
        },
    )
    fig_country.update_layout(yaxis_ticksuffix="%")
    st.plotly_chart(fig_country, width="stretch")

    # -- Conversion rate by source --
    st.subheader("Conversion rate by source")
    source_stats = (
        df.groupby("source")[TARGET]
        .mean()
        .reset_index()
        .rename(columns={TARGET: "conversion_rate"})
    )
    source_stats["conversion_rate"] *= 100
    fig_source = px.bar(
        source_stats,
        x="source",
        y="conversion_rate",
        text_auto=".2f",
        labels={
            "source": "Source",
            "conversion_rate": "Conversion rate (%)",
        },
    )
    fig_source.update_layout(yaxis_ticksuffix="%")
    st.plotly_chart(fig_source, width="stretch")

    # -- Pages visited vs conversion (dual-axis) --
    st.subheader("Pages visited vs conversion")
    pages_stats = (
        df.groupby("total_pages_visited")
        .agg(
            visitor_count=(TARGET, "count"),
            conversion_rate=(TARGET, "mean"),
        )
        .reset_index()
    )
    pages_stats["conversion_rate"] *= 100

    fig_pages = go.Figure()
    fig_pages.add_trace(
        go.Bar(
            x=pages_stats["total_pages_visited"],
            y=pages_stats["visitor_count"],
            name="Visitors",
            yaxis="y",
        )
    )
    fig_pages.add_trace(
        go.Scatter(
            x=pages_stats["total_pages_visited"],
            y=pages_stats["conversion_rate"],
            name="Conversion rate (%)",
            yaxis="y2",
            mode="lines+markers",
        )
    )
    fig_pages.update_layout(
        xaxis_title="Total pages visited",
        yaxis=dict(title="Visitor count", side="left"),
        yaxis2=dict(
            title="Conversion rate (%)",
            side="right",
            overlaying="y",
            ticksuffix="%",
        ),
        legend=dict(orientation="h", y=1.12),
    )
    st.plotly_chart(fig_pages, width="stretch")

    # -- Age distribution --
    st.subheader("Age distribution by conversion status")
    df_plot = df.copy()
    label_map = {0: "Not converted", 1: "Converted"}
    df_plot["converted_label"] = df_plot[TARGET].map(label_map)
    fig_age = px.histogram(
        df_plot,
        x="age",
        color="converted_label",
        barmode="overlay",
        labels={"age": "Age", "converted_label": "Status"},
        opacity=0.7,
    )
    st.plotly_chart(fig_age, width="stretch")

    # -- New user vs returning --
    st.subheader("New user vs returning -- conversion rate")
    user_stats = (
        df.groupby("new_user")[TARGET]
        .mean()
        .reset_index()
        .rename(columns={TARGET: "conversion_rate"})
    )
    user_stats["conversion_rate"] *= 100
    user_stats["label"] = user_stats["new_user"].map({0: "Returning", 1: "New user"})
    fig_user = px.bar(
        user_stats,
        x="label",
        y="conversion_rate",
        text_auto=".2f",
        labels={
            "label": "User type",
            "conversion_rate": "Conversion rate (%)",
        },
    )
    fig_user.update_layout(yaxis_ticksuffix="%")
    st.plotly_chart(fig_user, width="stretch")

# ==================================================================
# Tab 2 -- Model Results
# ==================================================================
with tab_model:
    # -- Model comparison --
    st.subheader("Model comparison")
    results = metadata.get("results_comparison", [])
    if results:
        df_results = pd.DataFrame(results)
        df_melted = df_results.melt(
            id_vars="Model",
            value_vars=["F1", "Precision", "Recall"],
            var_name="Metric",
            value_name="Score",
        )
        fig_comp = px.bar(
            df_melted,
            y="Model",
            x="Score",
            color="Metric",
            barmode="group",
            orientation="h",
            text_auto=".3f",
            labels={"Score": "Score", "Model": "Model"},
        )
        fig_comp.update_layout(
            xaxis=dict(range=[0, 1]),
            legend=dict(orientation="h", y=1.1),
        )
        st.plotly_chart(fig_comp, width="stretch")
    else:
        st.info("No model comparison data available.")

    # -- Feature importance --
    st.subheader("Feature importance (top 15)")
    feat_imp = metadata.get("feature_importance", [])
    if feat_imp:
        df_feat = (
            pd.DataFrame(feat_imp)
            .sort_values("Importance", ascending=False)
            .head(15)
            .sort_values("Importance", ascending=True)
        )
        fig_feat = px.bar(
            df_feat,
            y="Feature",
            x="Importance",
            orientation="h",
            text_auto=".4f",
            labels={
                "Feature": "Feature",
                "Importance": "Importance",
            },
        )
        st.plotly_chart(fig_feat, width="stretch")
    else:
        st.info("No feature importance data available.")

    # -- Threshold impact --
    st.subheader("Threshold impact")
    f1_default = metadata.get("val_f1_default", 0)
    f1_optimized = metadata.get("val_f1_optimized", 0)
    precision_opt = metadata.get("val_precision", 0)
    recall_opt = metadata.get("val_recall", 0)
    f1_delta = f1_optimized - f1_default

    tc1, tc2, tc3 = st.columns(3)
    with tc1:
        st.metric(
            "F1 (optimized vs default 0.5)",
            f"{f1_optimized:.4f}",
            delta=f"{f1_delta:+.4f}",
        )
    with tc2:
        st.metric(
            "Precision at optimal threshold",
            f"{precision_opt:.4f}",
        )
    with tc3:
        st.metric(
            "Recall at optimal threshold",
            f"{recall_opt:.4f}",
        )
