import streamlit as st
import pandas as pd
import numpy as np
import os
import sys

# Add src to path
sys.path.append('src')

from src.pipeline.predict_pipeline import CustomData, PredictPipeline
from src.exception import CustomException
from src.utils import load_object

def main():
    st.set_page_config(
        page_title="APS Failure Prediction",
        page_icon="🚛",
        layout="wide"
    )

    st.title("🚛 APS Failure Prediction System")
    st.markdown("### Predict Air Pressure System (APS) failure in Scania trucks")
    
    st.sidebar.header("📋 Model Information")
    st.sidebar.info("""
    This system predicts whether a truck failure is related to the Air Pressure System (APS) or other components.
    
    **Classes:**
    - **Positive (pos)**: APS-related failure
    - **Negative (neg)**: Non-APS related failure
    
    **Model Features:**
    - Handles severe class imbalance (1:59 ratio)
    - Uses advanced sampling techniques (SMOTE)
    - Multiple algorithms with class balancing
    """)

    # Main input section
    st.header("🔧 Enter Vehicle Sensor Data")
    
    # Create two columns for input
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Primary Sensors")
        aa_000 = st.number_input("aa_000", value=0.0, help="Primary sensor reading")
        ac_000 = st.number_input("ac_000", value=0.0, help="Pressure sensor")
        ad_000 = st.number_input("ad_000", value=0.0, help="Temperature sensor")
        ae_000 = st.number_input("ae_000", value=0.0, help="Flow sensor")
        af_000 = st.number_input("af_000", value=0.0, help="Auxiliary sensor")

    with col2:
        st.subheader("Secondary Sensors") 
        ag_000 = st.number_input("ag_000", value=0.0, help="Secondary pressure")
        ag_001 = st.number_input("ag_001", value=0.0, help="Backup sensor")
        ag_002 = st.number_input("ag_002", value=0.0, help="Safety sensor")
        ag_003 = st.number_input("ag_003", value=0.0, help="Control sensor")
        ag_004 = st.number_input("ag_004", value=0.0, help="Monitor sensor")

    # Add some key sensors that are likely important
    st.subheader("📊 Additional Key Metrics")
    col3, col4, col5 = st.columns(3)
    
    with col3:
        ee_002 = st.number_input("ee_002", value=0.0, help="End effector sensor")
        ee_003 = st.number_input("ee_003", value=0.0, help="Performance metric")
        
    with col4:
        ee_004 = st.number_input("ee_004", value=0.0, help="Efficiency sensor")
        ee_005 = st.number_input("ee_005", value=0.0, help="Quality metric")
        
    with col5:
        ee_006 = st.number_input("ee_006", value=0.0, help="Reliability sensor")
        ee_007 = st.number_input("ee_007", value=0.0, help="Safety metric")

    # File upload option
    st.header("📁 Or Upload Sensor Data File")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"File uploaded successfully! Shape: {df.shape}")
            st.dataframe(df.head())
            
            if st.button("🔍 Predict from File"):
                with st.spinner("Making predictions..."):
                    try:
                        predict_pipeline = PredictPipeline()
                        predictions = predict_pipeline.predict(df)
                        
                        # Create results dataframe
                        results_df = df.copy()
                        results_df['Prediction'] = predictions
                        results_df['Prediction_Label'] = results_df['Prediction'].map({0: 'neg (Non-APS)', 1: 'pos (APS-related)'})
                        
                        st.success("✅ Predictions completed!")
                        st.dataframe(results_df[['Prediction', 'Prediction_Label']])
                        
                        # Summary statistics
                        st.subheader("📊 Prediction Summary")
                        col_sum1, col_sum2 = st.columns(2)
                        with col_sum1:
                            aps_failures = (predictions == 1).sum()
                            st.metric("APS-related failures", aps_failures)
                        with col_sum2:
                            non_aps_failures = (predictions == 0).sum()
                            st.metric("Non-APS failures", non_aps_failures)
                        
                    except Exception as e:
                        st.error(f"Error making predictions: {str(e)}")
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")

    # Manual prediction
    if st.button("🎯 Predict APS Failure", type="primary"):
        with st.spinner("Analyzing sensor data..."):
            try:
                # Create a feature dictionary with all sensors
                # For simplicity, we'll use the input sensors and fill others with 0
                feature_dict = {
                    'aa_000': aa_000,
                    'ac_000': ac_000, 
                    'ad_000': ad_000,
                    'ae_000': ae_000,
                    'af_000': af_000,
                    'ag_000': ag_000,
                    'ag_001': ag_001,
                    'ag_002': ag_002,
                    'ag_003': ag_003,
                    'ag_004': ag_004,
                    'ee_002': ee_002,
                    'ee_003': ee_003,
                    'ee_004': ee_004,
                    'ee_005': ee_005,
                    'ee_006': ee_006,
                    'ee_007': ee_007
                }
                
                # Load a sample of the training data to get all feature names
                if os.path.exists('artifacts/train.csv'):
                    sample_df = pd.read_csv('artifacts/train.csv', nrows=1)
                    feature_columns = [col for col in sample_df.columns if col != 'class']
                    
                    # Create full feature vector
                    full_features = {}
                    for col in feature_columns:
                        full_features[col] = feature_dict.get(col, 0.0)
                    
                    # Create prediction
                    data = CustomData(**full_features)
                    pred_df = data.get_data_as_data_frame()
                    
                    predict_pipeline = PredictPipeline()
                    results = predict_pipeline.predict(pred_df)
                    
                    # Display results
                    prediction = results[0]
                    
                    if prediction == 1:
                        st.error("🚨 **APS-RELATED FAILURE PREDICTED**")
                        st.markdown("""
                        **Recommendation:** 
                        - Immediate inspection of Air Pressure System required
                        - Check air compressor, valves, and pressure lines  
                        - Schedule maintenance for APS components
                        """)
                    else:
                        st.success("✅ **NON-APS FAILURE PREDICTED**") 
                        st.markdown("""
                        **Recommendation:**
                        - Issue likely related to other vehicle systems
                        - Perform general diagnostic check
                        - APS system appears to be functioning normally
                        """)
                        
                    # Add confidence information
                    st.info("💡 This prediction is based on sensor patterns from historical truck data.")
                        
                else:
                    st.error("Model artifacts not found. Please train the model first.")
                    
            except Exception as e:
                st.error(f"Prediction failed: {str(e)}")

    # Model performance section
    if st.checkbox("📈 Show Model Performance"):
        st.header("🎯 Model Performance Metrics")
        
        if os.path.exists('artifacts/model_report.txt'):
            with open('artifacts/model_report.txt', 'r') as f:
                report_content = f.read()
            st.text(report_content)
        else:
            st.warning("Model report not available. Train the model to generate performance metrics.")

    # Training section  
    st.header("🚀 Model Training")
    if st.button("🔄 Retrain Model"):
        with st.spinner("Training model... This may take several minutes."):
            try:
                from src.pipeline.train_pipeline import TrainingPipeline
                
                training_pipeline = TrainingPipeline()
                score = training_pipeline.start_training_pipeline()
                
                st.success(f"✅ Model training completed! F1 Score: {score:.4f}")
                st.balloons()
                
            except Exception as e:
                st.error(f"Training failed: {str(e)}")

    # Footer
    st.markdown("---")
    st.markdown("🔧 **Built with Streamlit** | 📊 **APS Failure Prediction System**")

if __name__ == "__main__":
    main()
