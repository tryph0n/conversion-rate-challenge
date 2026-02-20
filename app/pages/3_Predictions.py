"""Prediction page for conversion rate model."""

import sys
from pathlib import Path

import joblib
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Predictions", layout="wide")

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config import METADATA_FILE, MODEL_FILE  # noqa: E402
from src.features import add_features  # noqa: E402


# ------------------------------------------------------------------
# Model loading
# ------------------------------------------------------------------
@st.cache_resource
def load_model():
    """Load the trained model pipeline from disk."""
    return joblib.load(MODEL_FILE)


@st.cache_data
def load_metadata() -> dict:
    """Load model metadata from joblib artifact."""
    return joblib.load(METADATA_FILE)


model = load_model()
metadata = load_metadata()
threshold = metadata.get("best_threshold", 0.5)

# ------------------------------------------------------------------
# Page header
# ------------------------------------------------------------------
st.title("Predictions")
st.write("Generate conversion predictions for individual visitors with What-If analysis.")

# ------------------------------------------------------------------
# Section 1 -- Single prediction
# ------------------------------------------------------------------
st.header("Single Prediction")

prediction_made = False

with st.form("single_prediction"):
    col_left, col_right = st.columns(2)

    with col_left:
        country = st.selectbox("Country", options=["China", "UK", "Germany", "US"])
        age = st.slider("Age", min_value=17, max_value=80, value=30)
        user_type = st.radio("User type", options=["New user", "Returning user"])

    with col_right:
        source = st.selectbox("Source", options=["Direct", "Ads", "Seo"])
        total_pages_visited = st.slider("Pages visited", min_value=1, max_value=29, value=5)

    submitted = st.form_submit_button("Predict")

if submitted:
    new_user = 1 if user_type == "New user" else 0

    input_df = pd.DataFrame(
        [
            {
                "country": country,
                "age": age,
                "new_user": new_user,
                "source": source,
                "total_pages_visited": total_pages_visited,
            }
        ]
    )
    input_df = add_features(input_df)

    proba = model.predict_proba(input_df)[:, 1][0]
    predicted = proba >= threshold

    # Confidence based on distance from threshold
    distance = abs(proba - threshold)
    if distance > 0.15:
        confidence = "High"
    elif distance > 0.05:
        confidence = "Medium"
    else:
        confidence = "Low"

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            "Conversion probability",
            f"{proba * 100:.1f}%",
        )
    with col2:
        label = "Will convert" if predicted else "Will not convert"
        st.metric("Prediction", label)
    with col3:
        st.metric(
            "Confidence",
            confidence,
        )

    if predicted:
        st.success(
            f"This visitor is likely to convert "
            f"(probability: {proba * 100:.1f}%,"
            f" confidence: {confidence})."
        )
    else:
        st.info(
            f"This visitor is unlikely to convert "
            f"(probability: {proba * 100:.1f}%,"
            f" confidence: {confidence})."
        )

    with st.expander("How is confidence calculated?"):
        st.markdown(
            f"""
Confidence measures how far the predicted
probability ({proba * 100:.1f}%) is from the
decision threshold ({threshold:.3f}).

- **High**: probability is far from the threshold
  (distance > 15%). The classification is robust.
- **Medium**: moderate distance (5-15%).
  The prediction could change with small
  variations in visitor profile.
- **Low**: probability is very close to the
  threshold (< 5%). The prediction is uncertain.

Current distance: **{distance * 100:.1f}%**
            """.strip()
        )

    prediction_made = True

# ------------------------------------------------------------------
# Section 2 -- What-If Analysis
# ------------------------------------------------------------------
if prediction_made:
    st.header("What-If Analysis")

    what_if_data = []
    for pages in range(1, 30):
        what_if_data.append(
            {
                "country": country,
                "age": age,
                "new_user": new_user,
                "source": source,
                "total_pages_visited": pages,
            }
        )
    what_if_df = pd.DataFrame(what_if_data)
    what_if_df = add_features(what_if_df)
    probas = model.predict_proba(what_if_df)[:, 1]

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=list(range(1, 30)),
            y=probas,
            mode="lines+markers",
            name="Conversion probability",
        )
    )

    # Horizontal dashed line at threshold
    fig.add_hline(
        y=threshold,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Threshold ({threshold:.3f})",
        annotation_position="top right",
    )

    # Vertical dashed line at user's selected value
    fig.add_vline(
        x=total_pages_visited,
        line_dash="dash",
        line_color="gray",
        annotation_text=f"Selected ({total_pages_visited})",
        annotation_position="top right",
    )

    fig.update_layout(
        title="Impact of Pages Visited on Conversion Probability",
        xaxis_title="Pages visited",
        yaxis_title="Conversion probability",
        yaxis=dict(range=[0, 1], tickformat=".0%"),
        xaxis=dict(dtick=1),
        showlegend=False,
    )

    st.plotly_chart(fig, use_container_width=True)
