import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide"
)


# =========================================================
# LOGIN SYSTEM
# =========================================================

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False


# =========================================================
# LOGIN PAGE
# =========================================================

if not st.session_state.logged_in:

    st.title("🔐 Credit Card Fraud Detection")

    st.subheader("Admin Login")

    st.write(
        "Please login to access the fraud detection dashboard."
    )

    username = st.text_input(
        "Username"
    )

    password = st.text_input(
        "Password",
        type="password"
    )

    if st.button(
        "🔑 Login",
        width="stretch"
    ):

        if username == "admin" and password == "admin123":

            st.session_state.logged_in = True

            st.success(
                "✅ Login successful!"
            )

            st.rerun()

        else:

            st.error(
                "❌ Invalid username or password"
            )

    st.stop()


# =========================================================
# LOAD MODEL AND DATA
# =========================================================

model = joblib.load("fraud_model.pkl")
data = pd.read_csv("creditcard.csv")


roc_data = joblib.load(
    "roc_data.pkl"
)

auc_score = joblib.load(
    "auc_score.pkl"
)

comparison = joblib.load(
    "model_comparison.pkl"
)


# =========================================================
# DATASET INFORMATION
# =========================================================

total_transactions = len(data)

normal_transactions = (
    data["Class"] == 0
).sum()

fraud_transactions = (
    data["Class"] == 1
).sum()


# =========================================================
# SIDEBAR NAVIGATION
# =========================================================

with st.sidebar:

    st.header("💳 Fraud Detection")

    st.success(
        "👤 Admin Logged In"
    )

    st.divider()

    page = st.radio(
        "📌 Navigation",
        [
            "🏠 Home",
            "📊 Dashboard",
            "🔍 Prediction",
            "📖 About Project"
        ]
    )

    st.divider()

    if st.button(
        "🚪 Logout",
        width="stretch"
    ):

        st.session_state.logged_in = False

        st.rerun()


# =========================================================
# HOME PAGE
# =========================================================

if page == "🏠 Home":

    st.title(
        "💳 Credit Card Fraud Detection System"
    )

    st.subheader(
        "🤖 Machine Learning Based Transaction Analysis"
    )

    st.info(
        "This system uses a Random Forest machine learning "
        "model with SMOTE to identify potentially fraudulent "
        "credit card transactions."
    )

    st.divider()

    st.header(
        "🏠 Welcome to the System"
    )

    st.write(
        """
        This application is designed to detect fraudulent
        credit card transactions using machine learning.

        The system analyzes transaction features and predicts
        whether a transaction is Normal or Fraudulent.
        """
    )

    st.divider()

    st.header(
        "📊 Project Overview"
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Total Transactions",
            f"{total_transactions:,}"
        )

    with col2:

        st.metric(
            "Normal Transactions",
            f"{normal_transactions:,}"
        )

    with col3:

        st.metric(
            "Fraud Transactions",
            f"{fraud_transactions:,}"
        )

    st.divider()

    st.header(
        "🚀 System Features"
    )

    col1, col2 = st.columns(2)

    with col1:

        st.success(
            """
            🔐 Admin Login

            📊 Dataset Dashboard

            📈 Model Performance

            🎯 ROC-AUC Analysis

            🔲 Confusion Matrix
            """
        )

    with col2:

        st.success(
            """
            🤖 Model Comparison

            🔍 Feature Importance

            💳 Transaction Prediction

            🧾 Manual Prediction

            📂 CSV Batch Prediction
            """
        )

    st.divider()

    st.header(
        "🧠 Machine Learning Model"
    )

    st.write(
        """
        The main machine learning algorithm used in this
        project is Random Forest.

        SMOTE (Synthetic Minority Over-sampling Technique)
        is used during training to handle the highly
        imbalanced fraud dataset.
        """
    )

    st.success(
        f"🏆 Random Forest ROC-AUC Score: {auc_score:.4f}"
    )


# =========================================================
# DASHBOARD PAGE
# =========================================================

elif page == "📊 Dashboard":

    st.title(
        "📊 Fraud Detection Dashboard"
    )

    st.subheader(
        "Dataset and Machine Learning Analysis"
    )

    st.divider()

    # =====================================================
    # DATASET DASHBOARD
    # =====================================================

    st.header(
        "📊 Dataset Dashboard"
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Total Transactions",
            f"{total_transactions:,}"
        )

    with col2:

        st.metric(
            "Normal Transactions",
            f"{normal_transactions:,}"
        )

    with col3:

        st.metric(
            "Fraud Transactions",
            f"{fraud_transactions:,}"
        )

    st.divider()

    # =====================================================
    # TRANSACTION DISTRIBUTION
    # =====================================================

    st.header(
        "📈 Transaction Distribution"
    )

    distribution = pd.DataFrame({
        "Transaction Type": [
            "Normal",
            "Fraud"
        ],

        "Count": [
            normal_transactions,
            fraud_transactions
        ]
    })

    fig, ax = plt.subplots(
        figsize=(10, 5)
    )

    ax.bar(
        distribution["Transaction Type"],
        distribution["Count"]
    )

    ax.set_xlabel(
        "Transaction Type"
    )

    ax.set_ylabel(
        "Number of Transactions"
    )

    ax.set_title(
        "Normal vs Fraud Transactions"
    )

    for i, value in enumerate(
        distribution["Count"]
    ):

        ax.text(
            i,
            value,
            f"{value:,}",
            ha="center",
            va="bottom"
        )

    st.pyplot(fig)

    plt.close(fig)

    st.divider()

    # =====================================================
    # MODEL PERFORMANCE
    # =====================================================

    st.header(
        "🎯 Random Forest Model Performance"
    )

    st.write(
        "Random Forest model evaluated on the test dataset "
        "after applying SMOTE to the training data."
    )

    metric1, metric2, metric3, metric4 = st.columns(4)

    with metric1:

        st.metric(
            "Accuracy",
            "99.94%"
        )

    with metric2:

        st.metric(
            "Precision",
            "82.00%"
        )

    with metric3:

        st.metric(
            "Recall",
            "83.67%"
        )

    with metric4:

        st.metric(
            "F1 Score",
            "82.83%"
        )

    st.divider()

    # =====================================================
    # ROC-AUC
    # =====================================================

    st.header(
        "📈 ROC-AUC Curve"
    )

    st.write(
        "ROC curve showing the performance of the "
        "Random Forest fraud detection model."
    )

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    ax.plot(
        roc_data["FPR"],
        roc_data["TPR"],
        label=f"Random Forest (AUC = {auc_score:.4f})"
    )

    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Random Classifier"
    )

    ax.set_xlabel(
        "False Positive Rate"
    )

    ax.set_ylabel(
        "True Positive Rate"
    )

    ax.set_title(
        "ROC Curve - Random Forest"
    )

    ax.legend()

    ax.grid(True)

    st.pyplot(fig)

    plt.close(fig)

    st.metric(
        "ROC-AUC Score",
        f"{auc_score:.4f}"
    )

    st.success(
        f"Random Forest achieved an excellent "
        f"ROC-AUC score of {auc_score:.4f}."
    )

    st.divider()

    # =====================================================
    # CONFUSION MATRIX
    # =====================================================

    st.header(
        "🔲 Confusion Matrix"
    )

    cm = [
        [56846, 18],
        [16, 82]
    ]

    cm_data = pd.DataFrame(
        cm,
        index=[
            "Actual Normal",
            "Actual Fraud"
        ],
        columns=[
            "Predicted Normal",
            "Predicted Fraud"
        ]
    )

    st.dataframe(
        cm_data,
        width="stretch"
    )

    st.write(
        "**Interpretation:**"
    )

    st.write(
        "✅ 56,846 normal transactions were correctly classified."
    )

    st.write(
        "✅ 82 fraudulent transactions were correctly detected."
    )

    st.write(
        "⚠️ 18 normal transactions were incorrectly classified as fraud."
    )

    st.write(
        "⚠️ 16 fraudulent transactions were incorrectly classified as normal."
    )

    st.divider()

    # =====================================================
    # MODEL COMPARISON
    # =====================================================

    st.header(
        "🤖 Model Comparison"
    )

    st.write(
        "Comparison between Logistic Regression and "
        "Random Forest."
    )

    display_comparison = comparison.copy()

    metrics = [
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score"
    ]

    display_comparison[metrics] = (
        display_comparison[metrics].round(2)
    )

    st.dataframe(
        display_comparison,
        width="stretch",
        hide_index=True
    )

    st.subheader(
        "📊 Performance Comparison"
    )

    models = comparison["Model"]

    x = range(
        len(models)
    )

    width = 0.18

    fig, ax = plt.subplots(
        figsize=(12, 6)
    )

    ax.bar(
        [i - 1.5 * width for i in x],
        comparison["Accuracy"],
        width,
        label="Accuracy"
    )

    ax.bar(
        [i - 0.5 * width for i in x],
        comparison["Precision"],
        width,
        label="Precision"
    )

    ax.bar(
        [i + 0.5 * width for i in x],
        comparison["Recall"],
        width,
        label="Recall"
    )

    ax.bar(
        [i + 1.5 * width for i in x],
        comparison["F1 Score"],
        width,
        label="F1 Score"
    )

    ax.set_xticks(
        list(x)
    )

    ax.set_xticklabels(
        models
    )

    ax.set_ylabel(
        "Score (%)"
    )

    ax.set_xlabel(
        "Machine Learning Model"
    )

    ax.set_title(
        "Machine Learning Model Performance Comparison"
    )

    ax.legend()

    ax.grid(
        axis="y",
        alpha=0.3
    )

    ax.set_ylim(
        0,
        105
    )

    st.pyplot(fig)

    plt.close(fig)

    st.success(
        "🏆 Random Forest performs better overall "
        "for this fraud detection project."
    )

    st.divider()

    # =====================================================
    # FEATURE IMPORTANCE
    # =====================================================

    st.header(
        "🔍 Feature Importance"
    )

    st.write(
        "Top features that contribute to the Random Forest "
        "fraud detection prediction."
    )

    feature_names = data.drop(
        "Class",
        axis=1
    ).columns

    importance = model.feature_importances_

    feature_importance = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importance
    })

    feature_importance = feature_importance.sort_values(
        by="Importance",
        ascending=False
    )

    top_features = feature_importance.head(
        10
    )

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    ax.barh(
        top_features["Feature"][::-1],
        top_features["Importance"][::-1]
    )

    ax.set_xlabel(
        "Importance"
    )

    ax.set_ylabel(
        "Feature"
    )

    ax.set_title(
        "Top 10 Important Features"
    )

    ax.grid(
        axis="x",
        alpha=0.3
    )

    st.pyplot(fig)

    plt.close(fig)

    st.dataframe(
        top_features,
        width="stretch",
        hide_index=True
    )


# =========================================================
# PREDICTION PAGE
# =========================================================

elif page == "🔍 Prediction":

    st.title(
        "🔍 Transaction Prediction"
    )

    st.subheader(
        "Use the Random Forest model to detect fraud"
    )

    st.divider()

    # =====================================================
    # DATASET TRANSACTION PREDICTION
    # =====================================================

    st.header(
        "💳 Dataset Transaction Prediction"
    )

    st.write(
        "Select a transaction from the dataset and "
        "check whether it is normal or fraudulent."
    )

    transaction_type = st.selectbox(
        "Choose transaction type",
        [
            "Normal Transaction",
            "Fraud Transaction"
        ]
    )

    if transaction_type == "Normal Transaction":

        indexes = data[
            data["Class"] == 0
        ].index

    else:

        indexes = data[
            data["Class"] == 1
        ].index

    selected_index = st.selectbox(
        "Select Transaction ID",
        indexes
    )

    transaction = data.loc[
        selected_index
    ]

    st.subheader(
        "💳 Transaction Details"
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Transaction ID",
            selected_index
        )

    with col2:

        st.metric(
            "Amount",
            f"${transaction['Amount']:.2f}"
        )

    with col3:

        actual_class = (
            "Fraud"
            if transaction["Class"] == 1
            else "Normal"
        )

        st.metric(
            "Actual Class",
            actual_class
        )

    st.subheader(
        "📋 Transaction Features"
    )

    feature_data = transaction.drop(
        "Class"
    )

    st.dataframe(
        feature_data.to_frame("Value"),
        width="stretch"
    )

    if st.button(
        "🔍 Check Transaction",
        width="stretch"
    ):

        input_data = transaction.drop(
            "Class"
        ).to_frame().T

        prediction = model.predict(
            input_data
        )[0]

        probability = model.predict_proba(
            input_data
        )[0]

        normal_probability = (
            probability[0] * 100
        )

        fraud_probability = (
            probability[1] * 100
        )

        st.subheader(
            "📊 Prediction Result"
        )

        if prediction == 1:

            st.error(
                "🚨 FRAUDULENT TRANSACTION"
            )

        else:

            st.success(
                "✅ NORMAL TRANSACTION"
            )

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "Normal Probability",
                f"{normal_probability:.2f}%"
            )

        with col2:

            st.metric(
                "Fraud Probability",
                f"{fraud_probability:.2f}%"
            )

        predicted_class = (
            "Fraud"
            if prediction == 1
            else "Normal"
        )

        st.subheader(
            "🔎 Prediction Verification"
        )

        if predicted_class == actual_class:

            st.success(
                f"Prediction Correct ✅ | "
                f"Actual: {actual_class} | "
                f"Predicted: {predicted_class}"
            )

        else:

            st.warning(
                f"Prediction Different ⚠️ | "
                f"Actual: {actual_class} | "
                f"Predicted: {predicted_class}"
            )

    st.divider()

    # =====================================================
    # MANUAL TRANSACTION PREDICTION
    # =====================================================

    st.header(
        "🧾 Manual Transaction Prediction"
    )

    st.write(
        "Enter transaction feature values manually."
    )

    model_features = data.drop(
        "Class",
        axis=1
    ).columns.tolist()

    manual_values = {}

    columns = st.columns(3)

    for i, feature in enumerate(
        model_features
    ):

        with columns[i % 3]:

            if feature == "Amount":

                manual_values[feature] = st.number_input(
                    f"{feature}",
                    min_value=0.0,
                    value=100.0,
                    step=1.0
                )

            elif feature == "Time":

                manual_values[feature] = st.number_input(
                    f"{feature}",
                    value=0.0,
                    step=1.0
                )

            else:

                manual_values[feature] = st.number_input(
                    f"{feature}",
                    value=0.0,
                    step=0.01,
                    format="%.4f"
                )

    if st.button(
        "🚀 Predict Manual Transaction",
        width="stretch"
    ):

        manual_input = pd.DataFrame(
            [manual_values],
            columns=model_features
        )

        manual_prediction = model.predict(
            manual_input
        )[0]

        manual_probability = model.predict_proba(
            manual_input
        )[0]

        normal_probability = (
            manual_probability[0] * 100
        )

        fraud_probability = (
            manual_probability[1] * 100
        )

        st.subheader(
            "📊 Manual Prediction Result"
        )

        if manual_prediction == 1:

            st.error(
                "🚨 FRAUDULENT TRANSACTION DETECTED"
            )

        else:

            st.success(
                "✅ TRANSACTION APPEARS NORMAL"
            )

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "Normal Probability",
                f"{normal_probability:.2f}%"
            )

        with col2:

            st.metric(
                "Fraud Probability",
                f"{fraud_probability:.2f}%"
            )

        probability_data = pd.DataFrame({
            "Probability": [
                normal_probability,
                fraud_probability
            ]
        }, index=[
            "Normal",
            "Fraud"
        ])

        st.subheader(
            "📈 Prediction Probability"
        )

        st.bar_chart(
            probability_data
        )

        predicted_label = (
            "Fraud"
            if manual_prediction == 1
            else "Normal"
        )

        st.subheader(
            "🔎 Prediction Summary"
        )

        st.info(
            f"Model Prediction: **{predicted_label}**"
        )

        st.write(
            f"💳 Transaction Amount: "
            f"${manual_values['Amount']:.2f}"
        )

        st.write(
            f"🧠 Fraud Probability: "
            f"{fraud_probability:.2f}%"
        )

    st.divider()

    # =====================================================
    # CSV BATCH PREDICTION
    # =====================================================

    st.header(
        "📂 CSV Batch Transaction Prediction"
    )

    st.write(
        "Upload a CSV file containing transaction features "
        "to predict multiple transactions at once."
    )

    uploaded_file = st.file_uploader(
        "Upload Transaction CSV",
        type=["csv"]
    )

    if uploaded_file is not None:

        uploaded_data = pd.read_csv(
            uploaded_file
        )

        st.subheader(
            "📋 Uploaded Data"
        )

        st.dataframe(
            uploaded_data.head(10),
            width="stretch"
        )

        required_features = data.drop(
            "Class",
            axis=1
        ).columns.tolist()

        missing_features = [
            feature
            for feature in required_features
            if feature not in uploaded_data.columns
        ]

        if missing_features:

            st.error(
                "❌ Missing required features:"
            )

            st.write(
                missing_features
            )

        else:

            if st.button(
                "🚀 Predict All Transactions",
                width="stretch"
            ):

                prediction_input = uploaded_data[
                    required_features
                ]

                predictions = model.predict(
                    prediction_input
                )

                probabilities = model.predict_proba(
                    prediction_input
                )

                result_data = uploaded_data.copy()

                result_data["Prediction"] = [
                    "Fraud"
                    if prediction == 1
                    else "Normal"
                    for prediction in predictions
                ]

                result_data[
                    "Fraud Probability (%)"
                ] = (
                    probabilities[:, 1] * 100
                ).round(2)

                result_data[
                    "Normal Probability (%)"
                ] = (
                    probabilities[:, 0] * 100
                ).round(2)

                total_predictions = len(
                    result_data
                )

                fraud_predictions = (
                    result_data["Prediction"] == "Fraud"
                ).sum()

                normal_predictions = (
                    result_data["Prediction"] == "Normal"
                ).sum()

                st.subheader(
                    "📊 Batch Prediction Summary"
                )

                col1, col2, col3 = st.columns(3)

                with col1:

                    st.metric(
                        "Total Transactions",
                        f"{total_predictions:,}"
                    )

                with col2:

                    st.metric(
                        "Normal Transactions",
                        f"{normal_predictions:,}"
                    )

                with col3:

                    st.metric(
                        "Fraud Transactions",
                        f"{fraud_predictions:,}"
                    )

                if fraud_predictions > 0:

                    st.warning(
                        f"⚠️ {fraud_predictions} potentially "
                        f"fraudulent transaction(s) detected."
                    )

                else:

                    st.success(
                        "✅ No potentially fraudulent "
                        "transactions detected."
                    )

                st.subheader(
                    "🔎 Prediction Results"
                )

                st.dataframe(
                    result_data,
                    width="stretch",
                    hide_index=True
                )

                csv_result = result_data.to_csv(
                    index=False
                )

                st.download_button(
                    label="⬇️ Download Prediction Results",
                    data=csv_result,
                    file_name="fraud_prediction_results.csv",
                    mime="text/csv",
                    width="stretch"
                )


# =========================================================
# ABOUT PROJECT PAGE
# =========================================================

elif page == "📖 About Project":

    st.title(
        "📖 About Project"
    )

    st.subheader(
        "💳 Credit Card Fraud Detection System"
    )

    st.divider()

    st.header(
        "🎯 Project Objective"
    )

    st.write(
        """
        The main objective of this project is to develop a
        machine learning based system that can identify
        potentially fraudulent credit card transactions.

        Since fraudulent transactions are much smaller in
        number compared to normal transactions, the dataset
        is highly imbalanced.
        """
    )

    st.divider()

    st.header(
        "🧠 Machine Learning Algorithm"
    )

    st.write(
        """
        Random Forest is used as the primary classification
        algorithm.

        Random Forest combines multiple decision trees to
        improve prediction performance and reduce overfitting.
        """
    )

    st.divider()

    st.header(
        "⚖️ SMOTE"
    )

    st.write(
        """
        SMOTE stands for Synthetic Minority Over-sampling
        Technique.

        It is used to handle class imbalance by generating
        synthetic samples for the minority fraud class during
        model training.
        """
    )

    st.divider()

    st.header(
        "📊 Dataset"
    )

    st.write(
        f"""
        Total Transactions: {total_transactions:,}

        Normal Transactions: {normal_transactions:,}

        Fraud Transactions: {fraud_transactions:,}
        """
    )

    st.divider()

    st.header(
        "🎯 Model Performance"
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Accuracy",
            "99.94%"
        )

    with col2:

        st.metric(
            "Precision",
            "82.00%"
        )

    with col3:

        st.metric(
            "Recall",
            "83.67%"
        )

    with col4:

        st.metric(
            "F1 Score",
            "82.83%"
        )

    st.divider()

    st.header(
        "📈 ROC-AUC"
    )

    st.metric(
        "ROC-AUC Score",
        f"{auc_score:.4f}"
    )

    st.divider()

    st.header(
        "🛠️ Technologies Used"
    )

    technologies = pd.DataFrame({
        "Technology": [
            "Python",
            "Pandas",
            "NumPy",
            "Scikit-learn",
            "Random Forest",
            "SMOTE",
            "Streamlit",
            "Matplotlib",
            "Joblib"
        ],

        "Purpose": [
            "Programming Language",
            "Data Processing",
            "Numerical Computation",
            "Machine Learning",
            "Fraud Classification",
            "Handling Imbalanced Data",
            "Web Application",
            "Data Visualization",
            "Model Saving and Loading"
        ]
    })

    st.dataframe(
        technologies,
        width="stretch",
        hide_index=True
    )

    st.divider()

    st.header(
        "🚀 Project Modules"
    )

    st.write(
        """
        🔐 Admin Authentication

        📊 Dataset Dashboard

        📈 Transaction Distribution

        🎯 Model Performance

        📈 ROC-AUC Analysis

        🔲 Confusion Matrix

        🤖 Model Comparison

        🔍 Feature Importance

        💳 Dataset Transaction Prediction

        🧾 Manual Transaction Prediction

        📂 CSV Batch Prediction
        """
    )

    st.divider()

    st.success(
        "🎓 Credit Card Fraud Detection | "
        "Final Year Data Science Project"
    )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "💳 Credit Card Fraud Detection | "
    "Final Year Data Science Project"
)