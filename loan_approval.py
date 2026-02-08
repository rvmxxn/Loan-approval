
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

@st.cache_data
def load_data(csv_path: str) -> pd.DataFrame:
  df = pd.read_csv(csv_path)
  return df

@st.cache_resource
def train_model(df: pd.DataFrame):
    target = "approved"
    drop_cols = [target]

    if "applicant_name" in df.columns:
        drop_cols.append("applicant_name")

    X = df.drop(columns=drop_cols)
    y = df[target]

    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, num_cols),
            ("cat", categorical_transformer, cat_cols)
        ]
    )

    model = LogisticRegression(max_iter=2000)

    clf = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("model", model)
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist()
    }

    feature_order = X.columns.tolist()

    return clf, metrics, feature_order


st.set_page_config(page_title="Loan Approval Prediction", page_icon=":money_with_wings:", layout="wide")
st.title("Loan Approval Prediction")
st.caption("Machine Learning Project")

st.sidebar.header("I. Load Dataset")
csv_path = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])

#loading dataset
try:
    df = load_data(csv_path)
except Exception as e:
    st.error(f"Error loading CSV file: {e}")
    st.stop()

st.sidebar.success(f"Loaded {len(df) :,} rows of data")

clf = None
metrics = None
feature_order = None



st. sidebar.header("II. Train Model")
train_now = st.sidebar.button("Train Model")

if train_now:
    with st.spinner("Training model..."):
        clf, metrics, feature_order = train_model(df)
        st.session_state["model"] = clf
        st.session_state["metrics"] = metrics
        st.session_state["feature_order"] = feature_order
colA, colB = st.columns([1,1])

with colA:
  st.subheader("Data Preview")
  st.dataframe(df.head(10), use_container_width=True)

if clf is not None and metrics is not None:
  with colB:
    st.subheader("Metrics (holdout test set)")
    st.write(f"Accuracy: {metrics['accuracy']}")
    st.write(f"Precision: {metrics['precision']}")
    st.write(f"Recall: {metrics['recall']}")
    st.write(f"F1 Score: {metrics['f1']}")

    cm = np.array(metrics["confusion_matrix"])
    st.write("Confusion Matrix (row: actual[0,1], cols: predicted[0,1])")
    st.dataframe(pd.DataFrame(cm, columns=["Pred 0", "Pred 1"], index=["Actual 0", "Actual 1"]), use_container_width=True)
    st.divider()

    st.subheader("Run a Prediction")
    c1, c2, c3, c4 = st.columns(4)

    with c1:
      applicant_name = st.text_input("Applicant Name", value="Muhammad Ali")
      gender = st.selectbox("Gender", ["M", "F"], index=0)
      age = st.slider("Age", 21, 60, 30)

    with c2:
      city = st.selectbox("City", sorted(df["city"].unique().tolist()))
      employement_type = st.selectbox("Employment Type", sorted(df["employment_type"].unique().tolist()))
      bank = st.selectbox("Bank", sorted(df["bank"].unique().tolist()))

    with c3:
      monthly_income_pkr = st.number_input("Monthly Income (PKR)", min_value=10000, max_value=1000000, value=50000)
      credit_score = st.slider("Credit Score", 300, 900, 600)

    with c4:
      loan_amount_pkr = st.number_input("Loan Amount (PKR)", min_value=50000, max_value=3500000, value=800000)
      loan_tenure_months = st.selectbox("Tenure (months)", [6, 12, 18, 24, 36, 48, 60])
      existing_loans = st.selectbox("Existing Loans", [0, 1, 2, 3], index=0)
      default_history = st.selectbox("Default History", [0, 1, 2], format_func=lambda x: "No ()" if x == 0 else "Yes (1)", index=0)
      has_credit_card = st.selectbox("Credit Card History", [0, 1], format_func=lambda x: "No ()" if x == 0 else "Yes (1)", index=0)


  
    input_row = pd.DataFrame([{
        "gender" : gender,
        "age" : age,
        "city" : city,
        "employment_type" : employement_type,
        "bank" : bank,
        "monthly_income_pkr" : monthly_income_pkr,
        "credit_score" : credit_score,
        "loan_amount_pkr" : loan_amount_pkr,
        "loan_tenure_months" : loan_tenure_months,
        "existing_loans" : existing_loans,
        "default_history" : default_history,
        "has_credit_card" : has_credit_card

    }])
  #input_row = input_row[feature_order]

if st.session_state.feature_order is not None:
  input_row = input_row[st.session_state.feature_order]

if st.button("Predict Loan Approva"):
  if st.session_state.model is not None:
    prob = float(st.session_state.model.predict_proba(input_row)[:,1][0])
    pred = int (prob >= 0.5)
    if pred == 1:
      st.success(f"{applicant_name} : APPROVED (Probability: {prob:.2%})")
    else:
      st.error(f"{applicant_name} : DENIED (Probability: {prob:.2%})")
  else:
    st.warning("Please train the model first by clicking 'Train Model' in the sidebar.")
