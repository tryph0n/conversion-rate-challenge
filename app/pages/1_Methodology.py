"""Methodology page -- conversion rate prediction project."""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Allow imports from project root.
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


st.set_page_config(page_title="Methodology", layout="wide")

# -------------------------------------------------------
# 1. Problem Statement
# -------------------------------------------------------
st.header("1. Problem Statement")

st.markdown(
    """
    The objective is to predict whether a website
    visitor will subscribe to a newsletter. This is
    a **binary classification** task on **highly
    imbalanced data** (3.2 % positive class).

    **Business context** -- False negatives (missing
    a potential subscriber) are more costly than
    false positives (targeting a visitor who would
    not convert). The model must therefore favor
    **recall** over precision when a tradeoff is
    necessary.
    """
)

# -------------------------------------------------------
# 2. Data Overview
# -------------------------------------------------------
st.header("2. Data Overview")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Training samples", "284,580")
col2.metric("Test samples", "31,620")
col3.metric("Original features", "6")
col4.metric("Conversion rate", "3.2 %")

st.markdown("**Imbalance ratio:** approximately 30:1 (negative to positive).")

st.subheader("Feature Descriptions")

feature_table = pd.DataFrame(
    {
        "Feature": [
            "country",
            "age",
            "new_user",
            "source",
            "total_pages_visited",
            "converted",
        ],
        "Type": [
            "Categorical",
            "Integer",
            "Binary",
            "Categorical",
            "Integer",
            "Binary (target)",
        ],
        "Values / Range": [
            "China, UK, Germany, US",
            "17 -- 123",
            "0 / 1",
            "Direct, Ads, Seo",
            "1 -- 29",
            "0 / 1",
        ],
    }
)

st.table(feature_table)

# -------------------------------------------------------
# 3. Data Cleaning & Outliers
# -------------------------------------------------------
st.header("3. Data Cleaning & Outliers")

st.markdown(
    """
    Rows with age > 80 were identified as data
    entry errors (ages 111 and 123, both with 100 %
    conversion rate). These **2 rows were removed**
    before training.
    """
)

# -------------------------------------------------------
# 4. Feature Engineering
# -------------------------------------------------------
st.header("4. Feature Engineering")

st.markdown(
    """
    Seven features were engineered from the original
    dataset to capture non-linear relationships and
    domain-specific signals.
    """
)

engineered_table = pd.DataFrame(
    {
        "Feature": [
            "age_group",
            "pages_group",
            "new_user_x_pages",
            "age_x_pages",
            "is_germany",
            "is_china",
            "is_seo",
        ],
        "Category": [
            "Binned",
            "Binned",
            "Interaction",
            "Interaction",
            "Binary flag",
            "Binary flag",
            "Binary flag",
        ],
        "Description": [
            "Age discretized into young / adult / middle / senior",
            "Pages visited discretized into low / medium / high / very high",
            "new_user * total_pages_visited",
            "age * total_pages_visited",
            "1 if country is Germany, else 0",
            "1 if country is China, else 0",
            "1 if source is Seo, else 0",
        ],
    }
)

st.table(engineered_table)

st.markdown("**Total features after preprocessing:** 6 original + 7 engineered = **13 features**.")

# -------------------------------------------------------
# 5. Feature Selection
# -------------------------------------------------------
st.header("5. Feature Selection")

st.markdown(
    """
    Feature relevance was assessed using **mutual
    information scores**. All features were retained
    as tree-based models handle irrelevant features
    natively via split selection. The analysis
    confirmed that `total_pages_visited` is the
    **dominant predictor**.
    """
)

# -------------------------------------------------------
# 6. Model Selection
# -------------------------------------------------------
st.header("6. Model Selection")

st.markdown(
    """
    Five models were compared using stratified
    cross-validation.

    To address the class imbalance (96.8% / 3.2%),
    we tested moderate upweighting (5x) for the
    minority class (`class_weight={0:1, 1:5}`), but
    **rejected it** as it degraded F1 by -0.002
    compared to default weights. The ~30x
    overcorrection of 'balanced' mode would destroy
    precision even more. Default weights were kept
    for all models.
    """
)

models_table = pd.DataFrame(
    {
        "Model": [
            "Logistic Regression",
            "Random Forest",
            "GradientBoosting",
            "GBT (tuned)",
            "LR (tuned)",
        ],
        "Type": [
            "Linear (baseline)",
            "Ensemble (bagging)",
            "Ensemble (boosting)",
            "Ensemble (boosting)",
            "Linear",
        ],
        "Tuning": [
            "Default (C=1.0, L2)",
            "n_estimators=200, max_depth=15",
            "Default (n_estimators=100, max_depth=3)",
            "GridSearchCV: max_depth, n_estimators, lr, min_samples_split",
            "GridSearchCV: C, penalty",
        ],
    }
)

st.table(models_table)

st.markdown(
    """
    **GBT tuning grid** -- `max_depth` [2, 3, 4],
    `n_estimators` [100, 200, 300],
    `learning_rate` [0.01, 0.05, 0.1],
    `min_samples_split` [5, 10].

    **LR tuning grid** -- `C` [0.01, 0.1, 1, 10],
    `penalty` [l1, l2].

    **Result** -- GradientBoosting (untuned baseline)
    was selected as the best performer based on
    cross-validated F1 score (F1 = **0.7593** at
    default threshold, improved to **0.7629** with
    threshold optimization).
    """
)

# -------------------------------------------------------
# 7. Threshold Optimization
# -------------------------------------------------------
st.header("7. Threshold Optimization")

st.markdown(
    """
    The default classification threshold of **0.5**
    is suboptimal for imbalanced datasets because
    the model's predicted probabilities are skewed
    toward the majority class. With a 30:1
    imbalance ratio, even a well-calibrated model
    outputs low probabilities for most positive
    cases, pushing the optimal decision boundary
    **below 0.5**.

    A **precision-recall tradeoff analysis** was
    performed to identify the threshold that
    maximizes the **F1 score**. The optimal
    threshold was found at **0.4459**, yielding
    F1 = 0.7629, precision = 0.8542,
    recall = 0.6892.
    """
)

# -------------------------------------------------------
# 8. Threshold Strategy
# -------------------------------------------------------
st.header("8. Threshold Strategy")

st.markdown(
    """
    Two threshold perspectives are presented:

    **Competition threshold** (F1-optimal at
    **0.4459**): Maximizes F1-score for the
    challenge leaderboard (F1 = 0.7629,
    precision = 0.8542, recall = 0.6892).

    **Break-even threshold** (**0.3025**): The
    point where precision equals recall (~0.75),
    yielding F1 = 0.7516. Useful when the
    business values precision and recall equally.

    In production, the threshold can be further
    adjusted based on the business cost ratio
    between missing a potential subscriber (false
    negative) and sending an unnecessary email
    (false positive).
    """
)

# -------------------------------------------------------
# 9. Prediction Confidence
# -------------------------------------------------------
st.header("9. Prediction Confidence")

st.markdown(
    """
    Each prediction includes a **confidence level**
    (High, Medium, or Low) based on how far the
    predicted probability is from the decision
    threshold.

    **How it works:**

    The model outputs a probability between 0% and
    100%. The decision threshold determines the
    cutoff for predicting conversion. Confidence
    measures the **distance** between the
    probability and this threshold:

    | Confidence | Distance | Meaning |
    |:----------:|:--------:|:--------|
    | **High** | > 15% | Classification is robust |
    | **Medium** | 5%-15% | Could change with small profile variations |
    | **Low** | < 5% | Near the boundary -- uncertain |

    **Why this approach:**

    This is a *decision margin* metric, widely used
    in classification. It directly answers the
    practical question: "How sure are we about this
    yes/no decision?" A visitor with 80% probability
    is clearly a converter (high confidence), while
    one at 47% (just above the threshold at 44.6%)
    could easily go either way (low confidence).

    This is more informative than model-internal
    metrics (e.g. ensemble variance) which tend to
    be near-zero for well-converged models.
    """
)

# -------------------------------------------------------
# 10. Validation Strategy
# -------------------------------------------------------
st.header("10. Validation Strategy")

col_a, col_b = st.columns(2)

with col_a:
    st.subheader("Splitting")
    st.markdown(
        """
        - **Stratified** train / validation split
          (80 / 20) to preserve the 3.2 % class
          ratio in both sets.
        - **3-fold stratified cross-validation**
          via GridSearchCV during hyperparameter
          tuning.
        """
    )

with col_b:
    st.subheader("Metrics")
    st.markdown(
        """
        - **F1 score** (primary) -- harmonic mean
          of precision and recall, suitable for
          imbalanced problems.
        - **Precision** -- proportion of predicted
          positives that are true positives.
        - **Recall** -- proportion of actual
          positives correctly identified.
        - **Threshold-optimized F1** -- F1
          re-evaluated at the optimal threshold
          found via precision-recall analysis.
        """
    )

st.markdown(
    """
    The final model was retrained on the **full
    training set** (284,580 samples) using the best
    hyperparameters found during cross-validation,
    then threshold optimization was applied before
    generating predictions on the test set.
    """
)
