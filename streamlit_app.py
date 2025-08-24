import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import json
from datetime import datetime

# Add src to path
sys.path.append('src')

from src.pipeline.predict_pipeline import CustomData, PredictPipeline
from src.exception import CustomException
from src.utils import load_object

def load_model_metadata():
    """Load model metadata if available"""
    try:
        metadata_path = os.path.join('artifacts', 'models', 'model_metadata.json')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                return json.load(f)
    except:
        pass
    return None

# def load_feature_info():
    # """Load feature information from data transformation"""
    # try:
def fix_features_df(df, feature_info_path='artifacts/data_transformation/feature_info.json'):
        try:
            with open(feature_info_path, 'r') as f:
                feature_info = json.load(f)
            final_features = feature_info["final_features"]
            
            for col in final_features:
                if col not in df.columns:
                    df[col] = 0.0
            
            df = df.loc[:, df.columns.isin(final_features)]
            df = df[final_features]
            return df

        # feature_info_path = os.path.join('artifacts', 'data_transformation', 'feature_info.json')
        # if os.path.exists(feature_info_path):
        #     with open(feature_info_path, 'r') as f:
        #         return json.load(f)
        except:
            pass
        return None

def load_model_comparison():
    """Load model comparison results"""
    try:
        comparison_path = os.path.join('artifacts', 'evaluations', 'model_comparison.csv')
        if os.path.exists(comparison_path):
            return pd.read_csv(comparison_path)
    except:
        pass
    return None

def main():
    st.set_page_config(
        page_title="APS Failure Prediction",
        page_icon="🚛",
        layout="wide"
    )

    st.title("🚛 APS Failure Prediction System")
    st.markdown("### Predict Air Pressure System (APS) failure in Scania trucks")

    # Load model metadata
    model_metadata = load_model_metadata()
    # feature_info = load_feature_info()
    feature_info = fix_features_df()
    
    # Sidebar with enhanced model information
    st.sidebar.header("📋 Model Information")
    
    if model_metadata:
        st.sidebar.success("✅ Model Loaded Successfully!")
        st.sidebar.info(f"""
        **Model Details:**
        - **Algorithm**: {model_metadata.get('model_name', 'Unknown')}
        - **Type**: {model_metadata.get('model_type', 'Unknown').title()}
        - **F1-Score**: {model_metadata.get('f1_score', 0):.4f}
        - **Precision**: {model_metadata.get('precision', 0):.4f}
        - **Recall**: {model_metadata.get('recall', 0):.4f}
        - **Trained**: {model_metadata.get('training_date', 'Unknown')[:10]}
        """)
    else:
        st.sidebar.warning("⚠️ Model metadata not found")
    
    if feature_info:
        st.sidebar.info(f"""
        **Data Processing:**
        - **Original Features**: {feature_info.get('n_original_features', 0)}
        - **Selected Features**: {feature_info.get('n_selected_features', 0)}
        - **Removed Features**: {feature_info.get('n_removed_features', 0)}
        """)

    st.sidebar.info("""
    **Problem Overview:**
    This system predicts whether a truck failure is related to the Air Pressure System (APS) or other components.

    **Classes:**
    - **Positive (pos)**: APS-related failure
    - **Negative (neg)**: Non-APS related failure

    **Key Features:**
    - Handles severe class imbalance (59:1 ratio)
    - Uses advanced sampling techniques (SMOTE)
    - Bayesian hyperparameter optimization
    - Multiple class balancing strategies
    """)

    # Check if model exists
    model_path = os.path.join('artifacts', 'models', 'best_model.pkl')
    model_exists = os.path.exists(model_path)
    
    if not model_exists:
        st.error("🚨 **Model not found!** Please train the model first.")
        st.info("👇 Use the 'Model Training' section below to train a new model.")
    
    # Main input section
    st.header("🔧 Enter Vehicle Sensor Data")
    
    # Load sample data for reference if available
    sample_data = None
    try:
        sample_data = CustomData.create_sample_input()
    except:
        pass

    # Create input method selection
    input_method = st.radio(
        "Choose input method:",
        ["Manual Input", "Upload CSV File", "Use Sample Data"],
        horizontal=True
    )

    if input_method == "Manual Input":
        # Create input fields for key features
        st.subheader("📊 Key Sensor Inputs")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("**Primary Sensors**")
            aa_000 = st.number_input("aa_000", value=0.0, help="Primary sensor reading", format="%.2f")
            ac_000 = st.number_input("ac_000", value=0.0, help="Pressure sensor", format="%.2f")
            ad_000 = st.number_input("ad_000", value=0.0, help="Temperature sensor", format="%.2f")
            ae_000 = st.number_input("ae_000", value=0.0, help="Flow sensor", format="%.2f")
            af_000 = st.number_input("af_000", value=0.0, help="Auxiliary sensor", format="%.2f")

        with col2:
            st.markdown("**Secondary Sensors**")
            ag_000 = st.number_input("ag_000", value=0.0, help="Secondary pressure", format="%.2f")
            ag_001 = st.number_input("ag_001", value=0.0, help="Backup sensor", format="%.2f")
            ag_002 = st.number_input("ag_002", value=0.0, help="Safety sensor", format="%.2f")
            ag_003 = st.number_input("ag_003", value=0.0, help="Control sensor", format="%.2f")
            ag_004 = st.number_input("ag_004", value=0.0, help="Monitor sensor", format="%.2f")

        with col3:
            st.markdown("**Performance Metrics**")
            ee_002 = st.number_input("ee_002", value=0.0, help="End effector sensor", format="%.2f")
            ee_003 = st.number_input("ee_003", value=0.0, help="Performance metric", format="%.2f")
            ee_004 = st.number_input("ee_004", value=0.0, help="Efficiency sensor", format="%.2f")
            ee_005 = st.number_input("ee_005", value=0.0, help="Quality metric", format="%.2f")
            ee_006 = st.number_input("ee_006", value=0.0, help="Reliability sensor", format="%.2f")

        with col4:
            st.markdown("**Additional Sensors**")
            ee_007 = st.number_input("ee_007", value=0.0, help="Safety metric", format="%.2f")
            ee_008 = st.number_input("ee_008", value=0.0, help="Diagnostic sensor", format="%.2f")
            ee_009 = st.number_input("ee_009", value=0.0, help="Status indicator", format="%.2f")
            ef_000 = st.number_input("ef_000", value=0.0, help="External sensor", format="%.2f")
            eg_000 = st.number_input("eg_000", value=0.0, help="General sensor", format="%.2f")

        # Create feature dictionary
        manual_features = {
            'aa_000': aa_000, 'ac_000': ac_000, 'ad_000': ad_000, 'ae_000': ae_000, 'af_000': af_000,
            'ag_000': ag_000, 'ag_001': ag_001, 'ag_002': ag_002, 'ag_003': ag_003, 'ag_004': ag_004,
            'ee_002': ee_002, 'ee_003': ee_003, 'ee_004': ee_004, 'ee_005': ee_005, 'ee_006': ee_006,
            'ee_007': ee_007, 'ee_008': ee_008, 'ee_009': ee_009, 'ef_000': ef_000, 'eg_000': eg_000
        }

    elif input_method == "Upload CSV File":
        st.subheader("📁 Upload Sensor Data File")
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        




        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                df = fix_features_df(df)
                st.success(f"✅ File uploaded successfully! Shape: {df.shape}")
                st.dataframe(df.head())

                if st.button("🔍 Predict from File", type="primary"):
                    if model_exists:
                        with st.spinner("Making predictions..."):
                            try:
                                predict_pipeline = PredictPipeline()
                                predictions = predict_pipeline.predict(df)

                                # Create results dataframe
                                results_df = df.copy()
                                results_df['Prediction'] = predictions
                                results_df['Prediction_Label'] = results_df['Prediction'].map({
                                    0: 'neg (Non-APS)', 
                                    1: 'pos (APS-related)',
                                    'neg': 'neg (Non-APS)',
                                    'pos': 'pos (APS-related)'
                                })

                                st.success("✅ Predictions completed!")
                                st.dataframe(results_df[['Prediction', 'Prediction_Label']])

                                # Summary statistics
                                st.subheader("📊 Prediction Summary")
                                col_sum1, col_sum2, col_sum3 = st.columns(3)
                                
                                with col_sum1:
                                    if isinstance(predictions[0], str):
                                        aps_failures = (predictions == 'pos').sum()
                                        non_aps_failures = (predictions == 'neg').sum()
                                    else:
                                        aps_failures = (predictions == 1).sum()
                                        non_aps_failures = (predictions == 0).sum()
                                    
                                    st.metric("🚨 APS-related failures", aps_failures)
                                
                                with col_sum2:
                                    st.metric("✅ Non-APS failures", non_aps_failures)
                                
                                with col_sum3:
                                    total_samples = len(predictions)
                                    aps_percentage = (aps_failures / total_samples) * 100
                                    st.metric("📊 APS failure rate", f"{aps_percentage:.1f}%")

                                # Download results
                                csv = results_df.to_csv(index=False)
                                st.download_button(
                                    label="💾 Download Results",
                                    data=csv,
                                    file_name=f"aps_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                    mime="text/csv"
                                )

                            except Exception as e:
                                st.error(f"❌ Error making predictions: {str(e)}")
                    else:
                        st.error("🚨 Model not available. Please train the model first.")

            except Exception as e:
                st.error(f"❌ Error reading file: {str(e)}")
        return

    elif input_method == "Use Sample Data":
        st.subheader("📊 Using Sample Data")
        if sample_data:
            st.success("✅ Sample data loaded successfully!")
            
            # Display sample data in a nice format
            sample_df = pd.DataFrame([sample_data])
            st.dataframe(sample_df.head())
            
            st.info("💡 This is a realistic sample from the APS dataset. You can modify individual values if needed.")
            
            # Option to modify sample data
            if st.checkbox("🔧 Modify Sample Data"):
                modified_data = {}
                cols = st.columns(4)
                col_names = list(sample_data.keys())[:20]  # Show first 20 features for modification
                
                for i, feature in enumerate(col_names):
                    with cols[i % 4]:
                        modified_data[feature] = st.number_input(
                            feature, 
                            value=float(sample_data[feature]), 
                            format="%.2f",
                            key=f"sample_{feature}"
                        )
                
                # Update sample_data with modifications
                sample_data.update(modified_data)
            
            manual_features = sample_data
        else:
            st.warning("⚠️ Could not load sample data. Please use manual input or file upload.")
            return

    # Prediction button for manual input and sample data
    if input_method in ["Manual Input", "Use Sample Data"]:
        if st.button("🎯 Predict APS Failure", type="primary"):
            if model_exists:
                with st.spinner("🔍 Analyzing sensor data..."):
                    try:
                        # Create prediction using CustomData
                        if input_method == "Manual Input":
                            # For manual input, create full feature set
                            data = CustomData(**manual_features)
                        else:
                            # For sample data, use all features
                            data = CustomData(**manual_features)
                        
                        pred_df = data.get_data_as_data_frame()
                        predict_pipeline = PredictPipeline()
                        results = predict_pipeline.predict(pred_df)

                        # Display results with enhanced UI
                        prediction = results[0]
                        
                        col_res1, col_res2 = st.columns(2)
                        
                        with col_res1:
                            if prediction == 1 or prediction == 'pos':
                                st.error("🚨 **APS-RELATED FAILURE PREDICTED**")
                                st.markdown("""
                                **⚠️ Immediate Action Required:**
                                - 🔧 Inspect Air Pressure System components
                                - 🔍 Check air compressor functionality
                                - 🛠️ Examine valves and pressure lines
                                - 📅 Schedule APS maintenance immediately
                                """)
                            else:
                                st.success("✅ **NON-APS FAILURE PREDICTED**")
                                st.markdown("""
                                **💡 Recommended Actions:**
                                - 🔍 Perform general diagnostic check
                                - 🛠️ Inspect other vehicle systems
                                - ✅ APS system appears normal
                                - 📊 Monitor for other failure indicators
                                """)
                        
                        with col_res2:
                            st.info("""
                            **📊 Prediction Details:**
                            
                            This prediction is based on:
                            - 🤖 Advanced machine learning algorithms
                            - 📈 Historical truck failure patterns  
                            - ⚖️ Class-balanced training approach
                            - 🎯 Optimized for imbalanced data
                            
                            **🔍 Model Performance:**
                            - Trained on 60,000 samples
                            - Handles 59:1 class imbalance
                            - Uses SMOTE for minority class synthesis
                            """)

                    except Exception as e:
                        st.error(f"❌ Prediction failed: {str(e)}")
                        st.error("Please check that all required preprocessors are available.")
            else:
                st.error("🚨 Model not available. Please train the model first.")

    # Model Performance Section
    st.header("📊 Model Performance & Analytics")
    
    performance_tab1, performance_tab2, performance_tab3 = st.tabs(["📈 Training Report", "🏆 Model Comparison", "📋 Technical Details"])
    
    with performance_tab1:
        st.subheader("📈 Model Training Report")
        report_path = os.path.join('artifacts', 'model_trainer', 'training_report.txt')
        if os.path.exists(report_path):
            with open(report_path, 'r',encoding='utf-8', errors='replace') as f:
                report_content = f.read()
            st.text(report_content)
        else:
            st.warning("⚠️ Training report not available. Train the model to generate performance metrics.")
    
    with performance_tab2:
        st.subheader("🏆 Model Comparison Results")
        comparison_df = load_model_comparison()
        if comparison_df is not None:
            st.dataframe(comparison_df.round(4))
            
            # Create performance visualization
            if len(comparison_df) > 0:
                st.subheader("📊 Performance Visualization")
                col_viz1, col_viz2 = st.columns(2)
                
                with col_viz1:
                    st.bar_chart(comparison_df.set_index('Model')['Test_F1'])
                    st.caption("F1-Score Comparison")
                
                with col_viz2:
                    metrics_df = comparison_df[['Model', 'Test_Precision', 'Test_Recall']].set_index('Model')
                    st.bar_chart(metrics_df)
                    st.caption("Precision vs Recall")
        else:
            st.warning("⚠️ Model comparison data not available.")
    
    with performance_tab3:
        st.subheader("📋 Technical Implementation Details")
        st.markdown("""
        **🔬 Machine Learning Pipeline:**
        
        **1. Data Preprocessing:**
        - 📊 Handles 170 features with extensive missing values
        - 🔄 Median imputation for numerical features
        - 📏 Standard scaling for feature normalization
        - 🎯 Variance threshold for feature selection
        
        **2. Class Imbalance Handling:**
        - ⚖️ Class weighting (`class_weight='balanced'`)
        - 🔄 SMOTE (Synthetic Minority Oversampling Technique)
        - 📊 Random over/under sampling strategies
        - 🎯 Macro F1-score for evaluation
        
        **3. Model Training:**
        - 🤖 Multiple algorithms: Logistic Regression, SVM, Decision Trees, Random Forest
        - 🔍 Bayesian hyperparameter optimization (10 iterations, 3-fold CV)
        - 📈 Stratified cross-validation
        - 🏆 Best model selection based on F1-macro score
        
        **4. Model Evaluation:**
        - 📊 80/20 stratified train-test split
        - 🎯 F1-score, Precision, Recall metrics
        - 📋 Confusion matrix analysis
        - 📈 Cross-validation vs test performance comparison
        """)

    # Model Training Section
    st.header("🚀 Model Training & Management")
    
    training_col1, training_col2 = st.columns(2)
    
    with training_col1:
        st.subheader("🔄 Retrain Model")
        if st.button("🚀 Start Training", type="secondary"):
            with st.spinner("🔥 Training model... This may take 20-30 minutes."):
                try:
                    from src.pipeline.train_pipeline import TrainingPipeline
                    
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Show training progress
                    status_text.text("📊 Initializing training pipeline...")
                    progress_bar.progress(10)
                    
                    training_pipeline = TrainingPipeline()
                    
                    status_text.text("📁 Loading and preprocessing data...")
                    progress_bar.progress(30)
                    
                    status_text.text("🤖 Training multiple models with Bayesian optimization...")
                    progress_bar.progress(70)
                    
                    score = training_pipeline.start_training_pipeline()
                    
                    status_text.text("✅ Training completed successfully!")
                    progress_bar.progress(100)
                    
                    st.success(f"🎉 Model training completed! Final F1 Score: {score:.4f}")
                    st.balloons()
                    
                    # Refresh page to show new model info
                    st.experimental_rerun()
                    
                except Exception as e:
                    st.error(f"❌ Training failed: {str(e)}")
    
    with training_col2:
        st.subheader("📋 Training Information")
        st.info("""
        **⏱️ Training Process:**
        - Data ingestion and preprocessing
        - Feature engineering and selection  
        - Multiple model training (4-12 models)
        - Hyperparameter optimization
        - Model evaluation and selection
        
        **⚡ Performance:**
        - Estimated time: 20-30 minutes
        - Uses Bayesian optimization for speed
        - 3-fold cross-validation
        - Parallel processing enabled
        """)

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center'>
    <p>🔧 <strong>Built with Streamlit</strong> | 📊 <strong>APS Failure Prediction System</strong></p>
    <p>🚛 Helping predict Air Pressure System failures in Scania trucks</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
