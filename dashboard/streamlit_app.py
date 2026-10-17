import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.detection.fraud_model import load_model, predict
from src.detection.rule_based import apply_fraud_rules
from src.utils.risk_scoring import combine_risk_scores, explain_risk


st.set_page_config(
    page_title="M-Pesa Fraud Detection System",
    page_icon="🔒",
    layout="wide",
)


@st.cache_resource
def get_model():
    try:
        return load_model()
    except Exception as e:
        return None


model_bundle = get_model()


st.title("M-Pesa Fraud Detection System")
st.markdown("Real-time fraud detection using ML, rule-based, and anomaly detection")

with st.sidebar:
    st.header("Navigation")
    page = st.selectbox(
        "Select Page",
        ["Dashboard", "Single Transaction", "Batch Analysis", "Model Performance"],
    )


if page == "Dashboard":
    st.header("System Overview")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Model Status", "Loaded" if model_bundle else "Not Found")
    col2.metric("Detection Methods", "3")
    col3.metric("ML + Rules + Anomaly", "Combined")
    col4.metric("Risk Scoring", "Active")

    st.info(
        "Upload transaction data or test single transactions to detect potential fraud."
    )


elif page == "Single Transaction":
    st.header("Analyze Single Transaction")
    with st.form("single_txn"):
        col1, col2 = st.columns(2)
        with col1:
            step = st.number_input("Step (hour)", min_value=1, value=1)
            txn_type = st.selectbox("Type", ["TRANSFER", "CASH_OUT", "CASH_IN", "PAYMENT", "DEBIT"])
            amount = st.number_input("Amount", min_value=0.0, value=1000.0)
            oldbal = st.number_input("Old Balance Orig", min_value=0.0, value=10000.0)
            newbal = st.number_input("New Balance Orig", min_value=0.0, value=9000.0)
        with col2:
            nameorig = st.text_input("Name Orig", value="C123")
            namedest = st.text_input("Name Dest", value="M456")
            oldbal_dest = st.number_input("Old Balance Dest", min_value=0.0, value=0.0)
            newbal_dest = st.number_input("New Balance Dest", min_value=0.0, value=1000.0)

        submitted = st.form_submit_button("Analyze")
        if submitted:
            data = {
                "step": [step],
                "type": [txn_type],
                "amount": [amount],
                "nameOrig": [nameorig],
                "oldbalanceOrg": [oldbal],
                "newbalanceOrig": [newbal],
                "nameDest": [namedest],
                "oldbalanceDest": [oldbal_dest],
                "newbalanceDest": [newbal_dest],
            }
            df = pd.DataFrame(data)
            try:
                res = combine_risk_scores(df)
                st.success("Analysis Complete")
                col1, col2, col3 = st.columns(3)
                col1.metric("Risk Score", f"{res['risk_score'].iloc[0]:.3f}")
                col2.metric("Risk Level", res['risk_level'].iloc[0].upper())
                try:
                    rule_res = apply_fraud_rules(df)
                    st.subheader("Rule-Based Analysis")
                    st.dataframe(rule_res[["rule_large_amount", "rule_balance_depletion", "rule_balance_mismatch", "rule_sensitive_transaction_type", "rules_triggered", "rule_based_flag", "rule_reasons"]])
                except Exception as e:
                    st.warning(f"Rule analysis failed: {e}")
                if model_bundle:
                    ml_res = predict(df, model_bundle)
                    st.subheader("ML Prediction")
                    st.dataframe(ml_res[["fraud_prediction", "fraud_probability"]])
                st.subheader("Risk Explanation")
                st.json(explain_risk(res.iloc[0]))
            except Exception as e:
                st.error(f"Analysis failed: {e}")


elif page == "Batch Analysis":
    st.header("Batch Transaction Analysis")
    uploaded = st.file_uploader("Upload CSV file", type=["csv"])
    if uploaded is not None:
        df = pd.read_csv(uploaded)
        st.write(f"Loaded {len(df)} transactions")
        with st.spinner("Analyzing..."):
            try:
                res = combine_risk_scores(df)
                st.success("Analysis Complete")
                st.dataframe(
                    res[["amount", "type", "risk_score", "risk_level"]]
                    .head(100)
                )
                high_risk = res[res["risk_level"] == "high"]
                st.metric("High Risk Transactions", len(high_risk))
                st.download_button(
                    label="Download Results (CSV)",
                    data=res.to_csv(index=False),
                    file_name="fraud_analysis_results.csv",
                    mime="text/csv",
                )
            except Exception as e:
                st.error(f"Batch analysis failed: {e}")


elif page == "Model Performance":
    st.header("Model Performance")
    st.info("Model performance metrics from evaluation")
    from src.evaluation.model_evaluation import evaluate_on_test_set

    if st.button("Run Evaluation"):
        with st.spinner("Evaluating..."):
            try:
                res = evaluate_on_test_set()
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Accuracy", f"{res['accuracy']:.3f}")
                col2.metric("Precision", f"{res['precision']:.3f}")
                col3.metric("Recall", f"{res['recall']:.3f}")
                col4.metric("F1", f"{res['f1']:.3f}")
                st.metric("ROC-AUC", f"{res['roc_auc']:.3f}")
                st.subheader("Classification Report")
                st.text(res["classification_report"])
                st.subheader("Confusion Matrix")
                cm = res["confusion_matrix"]
                st.write(pd.DataFrame(cm, columns=["Pred 0", "Pred 1"], index=["True 0", "True 1"]))
            except Exception as e:
                st.error(f"Evaluation failed: {e}")
