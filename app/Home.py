"""Streamlit landing page for the Conversion Rate Challenge project."""

import streamlit as st

st.set_page_config(
    page_title="Conversion Rate Challenge",
    layout="wide",
    page_icon="chart_with_upwards_trend",
)

st.title("Conversion Rate Challenge")

st.markdown(
    """
    Machine learning project predicting newsletter subscription conversions from user behavior data.
    Developed as a diploma defense piece and portfolio project.
    """
)

col1, col2, col3 = st.columns(3)

with col1:
    with st.container(border=True):
        st.subheader("Methodology")
        st.write("Approach and rationale behind the analysis.")
        st.page_link("pages/1_Methodology.py", label="Go to Methodology")

with col2:
    with st.container(border=True):
        st.subheader("Dashboard")
        st.write("Interactive data exploration and model results.")
        st.page_link("pages/2_Dashboard.py", label="Go to Dashboard")

with col3:
    with st.container(border=True):
        st.subheader("Predictions")
        st.write("Single conversion predictions with What-If analysis.")
        st.page_link("pages/3_Predictions.py", label="Go to Predictions")

st.divider()

st.header("Project Context")

ctx_left, ctx_right = st.columns(2)

with ctx_left:
    st.markdown(
        """
        **Binary classification challenge** from a real-world marketing dataset.

        - **Training samples:** 284,000 observations
        - **Positive class rate:** 3.2% (significant class imbalance)
        - **Features:** 6 input variables capturing user behavior and demographics
        - **Target:** whether a user subscribes to the newsletter

        The goal is to maximize predictive performance on the minority class
        while maintaining interpretability for business stakeholders.
        """
    )

with ctx_right:
    st.markdown(
        """
        **Key findings**

        - Best model: GradientBoosting with optimized threshold
        - Total pages visited is the strongest predictor of conversion
        - Germany shows the highest conversion rate among all countries
        - New users convert at a lower rate than returning users
        - Threshold optimization significantly improved the F1 score
          over the default 0.5 cutoff
        """
    )

st.caption("Built with Streamlit | Data Science Diploma Project")
