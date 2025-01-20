import streamlit as st
from streamlit_option_menu import option_menu
import pickle
import numpy as np
import pandas as pd 

# Load the Model
model = pickle.load(open("Model_RF.pkl", "rb"))

# -----------Header----------------
st.markdown("""
    <div style='text-align: center;'>
        <h1 style='color: #2222b2; white-space: nowrap;'>Online Payments Fraud Detection</h1>
    </div>
    """, unsafe_allow_html=True)

# -----------Background-img----------------
page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] > .main {{
background-image: url("https://i.ibb.co/68MXp2t/background-img.jpgg");
background-size: 180%;
background-position: top left;
background-repeat: no-repeat;
background-attachment: local;
}}
"""
st.markdown(page_bg_img, unsafe_allow_html=True)

# -----------Menu Bar----------------
def streamlit_menu():
    selected = option_menu(
        menu_title=None,  # required
        options=["Home", "File-Predict" ,"About App", "Help"],  # required
        icons=["house", "upload", "book", "envelope"],  # optional
        menu_icon="cast",  # optional
        default_index=1,  # optional
        orientation="horizontal",
        styles={
            "nav-link-selected": {"background-color": "blue"},
        }
    )
    return selected

selected = streamlit_menu()

# -----------Home Page----------------
if selected == "Home":

    def find_type(text):
        if text == "CASH_IN":
            return 0
        elif text == "CASH_OUT":
            return 1
        elif text == "DEBIT":
            return 2
        elif text == "PAYMENT":
            return 3
        else:
            return 4

    # Columns
    col1, col2 = st.columns(2)

    with col1:
        types = st.selectbox("Transaction Type", ("CASH_IN(0)", "CASH_OUT(1)", "DEBIT(2)", "PAYMENT(3)", "TRANSFER(4)"))
        oldbalanceOrg = st.number_input("Old Balance Original")

    with col2:
        amount = st.number_input("Amount")
        newbalanceOrg = st.number_input("New Balance Original")

    # custom box for result
    def apply_custom_css_fraud():
        st.markdown(
            """
            <style>
            .stAlert {
                background-color: #ffcccc;  /* Light red background for fraudulent transactions */
                color: black;  /* Text color */
            }
            </style>
            """,
            unsafe_allow_html=True
        )
    def apply_custom_css_notfraud():
        st.markdown(
            """
            <style>
            .stAlert {
                background-color: #ccffcc;  /* Light green background for non-fraudulent transactions */
                color: black;  /* Text color */
            }
            </style>
            """,
            unsafe_allow_html=True
        )

    if st.button("Predict"):
        types = find_type(types)
        test = np.array([[types, amount, oldbalanceOrg, newbalanceOrg]])
        res = model.predict(test)
        if res == 'Fraud':
            apply_custom_css_fraud()
            st.success("Prediction: " + "⚠️ The transaction is predicted to be fraudulent.")
        else:
            apply_custom_css_notfraud()
            st.success("Prediction: " + "✅ The transaction is predicted to be non-fraudulent.")

    # Create a button to clear the input fields
    if st.button("Clear"):
        st.experimental_rerun()

# -----------File-Predict Page----------------
if selected == "File-Predict":
    header = st.container()

    # Define columns and type dictionary
    columns = ['type', 'amount', 'oldbalanceOrg', 'newbalanceOrig']
    type_dict = {
        'PAYMENT': 0,
        'TRANSFER': 1,
        'CASH_OUT': 2, 
        'DEBIT': 3, 
        'CASH_IN': 4
    }

    def predict_batch(data):
        return model.predict(data)

    output = {0: 'Not Fraud', 1: 'Fraud'}

    with header:
        st.header("📂 Predict Multiple Transactions")
        st.write("Upload CSV file to check multiple transactions.")
        
    with st.expander("Upload .csv file to predict your transactions", expanded=True):
        uploaded_file = st.file_uploader("Choose file:", type="csv")

        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            st.write(df.head())
            st.write(f"Uploaded file contains {df.shape[0]} records and {df.shape[1]} attributes.")
        
            if st.button("Predict"):
                try:
                    # Prepare the data for prediction
                    new_df = df[columns].copy()
                    new_df['type'] = new_df['type'].replace(type_dict)

                    # Ensure numeric conversion
                    new_df[columns] = new_df[columns].apply(pd.to_numeric, errors='coerce')

                    # Predict
                    predictions = predict_batch(new_df)
                    df['isFraud'] = predictions
                    df['isFraud'] = df['isFraud'].replace(output)

                    # Display results
                    st.write("Predictions are succesfully stored in the file..")
                    
                    # Download the updated CSV file
                    st.download_button(label="Download updated CSV", data=df.to_csv(index=False), file_name='predicted_transactions.csv', mime='text/csv')
                except KeyError as e:
                    st.error(f"Key error: {e}. Please ensure the uploaded CSV contains the required columns.")
                except Exception as e:
                    st.error(f"An error occurred: {e}")

# -----------Project Page----------------
if selected == "About App":

    st.markdown(
        """
        <style>
        .project-image {
            width: 50%;  /* Adjust the width as needed */
            height: auto;  /* Maintain the aspect ratio */
            border: 2px solid #555;  /* Add border with specified color */
            display: block;
            margin-left: auto;
            margin-right: auto;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div style="text-align: center;">
            <img src="https://i.ibb.co/WnsXLs9/Untitled-design.jpg" class="project-image" alt="Project Image">
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("""
    ### App Introduction
    This application detects fraudulent online payment transactions using a Random Forest Classifier. 
    Users can input transaction details and instantly receive a prediction on whether the transaction is fraudulent or not, 
    helping to prevent financial losses and enhance security.
    """)

    st.markdown("""
    ### About Random Forest
    Random Forest is an ensemble learning method that operates by constructing a multitude of decision trees during training. 
    The model outputs the class that is the mode of the classes (classification) or mean prediction (regression) of the individual trees. 
    Random Forest corrects for decision trees' habit of overfitting to their training set. By averaging multiple trees trained on different parts of the same training set, 
    the Random Forest model reduces the variance and improves prediction accuracy.
    """)

    st.subheader("Model Performance Metrics")
    st.write(f"**Accuracy:** 99.21%")
    st.write(f"**Precision:** 99.06%")
    st.write(f"**Recall:** 99.41%")
    st.write(f"**F1 Score:** 99.23%")

    st.markdown("""
    ### Confusion Matrix & ROC Curve
    """)
    col3, col4 = st.columns(2)
    with col3:
        st.markdown(
            """
            <div style="align: left; width: 150%; max-width: 600px;">
                <img src="https://i.ibb.co/7XMVmDL/Confusion-matrix.png" class="project-image" alt="Project Image">
            </div>
            """,
            unsafe_allow_html=True
        )

    with col4:
        st.markdown(
            """
            <div style="align: left; width: 150%; max-width: 600px;">
                <img src="https://i.ibb.co/Cmw1JKK/ROC.png" class="project-image" alt="Project Image">
            </div>
            """,
            unsafe_allow_html=True
        )    
    
# -----------Help Page----------------
if selected == "Help":
    st.header("How to Use?")

    st.markdown("""
    1. **Select Transaction Type**: Choose the type of transaction (e.g., CASH_IN, CASH_OUT).
    2. **Enter Amount**: Input the transaction amount.
    3. **Enter Old Balance Original**: Provide the original balance before the transaction.
    4. **Enter New Balance Original**: Provide the new balance after the transaction.
    5. **Click 'Predict'**: Click the 'Predict' button to determine if the transaction is fraudulent.
    6. **View Results**: The prediction result will be displayed indicating whether the transaction is fraudulent or non-fraudulent.
    7. **Clear Fields**: Click 'Clear' to reset all input fields and start a new prediction.
    """)

# Hide Header Footer
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)