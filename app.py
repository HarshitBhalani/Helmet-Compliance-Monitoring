import streamlit as st
import tensorflow as tf
import cv2
import numpy as np
from PIL import Image
import os
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Helmet Compliance Monitor",
    page_icon="⛑️",
    layout="wide"
)

# Load the H5 model
@st.cache_resource
def load_model():
    try:
        # Load the .h5 model file
        model = tf.keras.models.load_model('model.h5')
        return model
    except FileNotFoundError:
        st.error("❌ model.h5 file not found! Please ensure model.h5 is in the same folder as app.py")
        return None
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None

# Prediction function
def predict_helmet(image, model):
    try:
        # Resize image to 224x224 (Teachable Machine standard)
        img = cv2.resize(image, (224, 224))
        
        # Convert to RGB if needed
        if len(img.shape) == 3 and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Normalize pixel values to 0-1
        img = img.astype(np.float32) / 255.0
        
        # Add batch dimension
        img = np.expand_dims(img, axis=0)
        
        # Make prediction
        prediction = model.predict(img, verbose=0)
        
        # Get prediction results
        confidence_scores = prediction[0]
        predicted_class = np.argmax(confidence_scores)
        max_confidence = float(np.max(confidence_scores))
        
        return predicted_class, max_confidence, confidence_scores
        
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None, None, None

# Initialize session state for violation logs
if 'violation_logs' not in st.session_state:
    st.session_state.violation_logs = []

def log_violation(image_name, confidence, violation_type):
    """Log safety violations"""
    log_entry = {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'image': image_name,
        'violation': violation_type,
        'confidence': f"{confidence:.2%}"
    }
    st.session_state.violation_logs.append(log_entry)

def main():
    st.title("⛑️ Helmet Compliance Monitoring System")
    st.markdown("### Ensuring Workplace Safety Through AI Detection")
    st.markdown("---")
    
    # Load model
    model = load_model()
    
    if model is None:
        st.error("🚨 **Model Loading Failed!**")
        
        # Check what files exist
        st.subheader("📁 Files in Current Directory:")
        current_files = [f for f in os.listdir('.') if os.path.isfile(f)]
        
        if current_files:
            for file in current_files:
                if file.endswith('.h5'):
                    st.success(f"✅ Found: {file}")
                else:
                    st.write(f"📄 {file}")
        else:
            st.write("No files found")
        
        st.markdown("""
        **🔧 Fix Steps:**
        1. Ensure your `model.h5` file is in the same folder as `app.py`
        2. File should be named exactly `model.h5`
        3. Restart the Streamlit app after placing the file
        """)
        return
    
    # Success message
    st.success("✅ Model loaded successfully!")
    
    # Model info
    with st.expander("📊 Model Information"):
        try:
            st.write(f"**Input Shape:** {model.input_shape}")
            st.write(f"**Output Shape:** {model.output_shape}")
            st.write(f"**Total Parameters:** {model.count_params():,}")
        except:
            st.write("Model information not available")
    
    # Sidebar
    st.sidebar.title("🎛️ Controls")
    st.sidebar.markdown("---")
    
    # Detection threshold
    threshold = st.sidebar.slider("Detection Threshold", 0.1, 0.95, 0.7, 0.05)
    st.sidebar.info(f"Current threshold: {threshold}")
    
    # Mode selection
    mode = st.sidebar.selectbox(
        "Select Detection Mode", 
        ["📷 Image Upload", "📹 Live Camera", "📁 Batch Processing", "📊 Violation Logs"]
    )
    
    st.sidebar.markdown("---")
    
    # Statistics
    if st.session_state.violation_logs:
        total_checks = len(st.session_state.violation_logs)
        violations = len([log for log in st.session_state.violation_logs if log['violation'] != 'Compliant'])
        compliance_rate = ((total_checks - violations) / total_checks) * 100
        
        st.sidebar.metric("Total Checks", total_checks)
        st.sidebar.metric("Compliance Rate", f"{compliance_rate:.1f}%")
        st.sidebar.metric("Violations", violations)
    
    # Main content based on mode
    if mode == "📷 Image Upload":
        st.header("📷 Single Image Detection")
        
        uploaded_file = st.file_uploader(
            "Upload an image for helmet detection", 
            type=['jpg', 'jpeg', 'png', 'bmp'],
            help="Supported formats: JPG, JPEG, PNG, BMP"
        )
        
        if uploaded_file is not None:
            # Display image
            image = Image.open(uploaded_file)
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.subheader("📸 Original Image")
                st.image(image, use_column_width=True)
                st.caption(f"File: {uploaded_file.name}")
            
            # Convert to array for prediction
            img_array = np.array(image)
            
            # Make prediction
            with st.spinner("🔍 Analyzing image..."):
                predicted_class, confidence, all_scores = predict_helmet(img_array, model)
            
            if predicted_class is not None:
                with col2:
                    st.subheader("🎯 Detection Results")
                    
                    # Fixed class mapping: Class 0 = With Helmet, Class 1 = Without Helmet
                    class_names = ["With Helmet", "Without Helmet"]
                    is_helmet_detected = predicted_class == 0  # Class 0 means helmet detected
                    
                    # Show all confidence scores
                    st.write("**Confidence Scores:**")
                    for i, score in enumerate(all_scores):
                        if i == predicted_class:
                            st.write(f"**{class_names[i]}: {score:.3f} ({score*100:.1f}%)**")
                        else:
                            st.write(f"{class_names[i]}: {score:.3f} ({score*100:.1f}%)")
                    
                    st.markdown("---")
                    
                    # Decision based on threshold and corrected logic
                    is_compliant = is_helmet_detected and confidence >= threshold
                    
                    if is_compliant:
                        st.success("✅ **HELMET DETECTED - COMPLIANT**")
                        st.metric("Status", "SAFE", "✓")
                        violation_type = "Compliant"
                    else:
                        st.error("❌ **NO HELMET DETECTED - VIOLATION**")
                        st.metric("Status", "UNSAFE", "⚠️")
                        violation_type = "No Helmet Detected"
                        
                        # Safety alert
                        st.warning("🚨 **SAFETY VIOLATION ALERT**")
                        st.markdown("""
                        **Immediate Actions Required:**
                        - 🛑 Stop work immediately
                        - 🪖 Provide safety helmet
                        - 📋 Brief worker on safety protocols
                        - 📝 Document the incident
                        """)
                    
                    st.metric("Confidence Level", f"{confidence:.1%}")
                    
                    # Debug information
                    with st.expander("🔍 Debug Information"):
                        st.write(f"Raw predicted class: {predicted_class}")
                        st.write(f"Is helmet detected: {is_helmet_detected}")
                        st.write(f"Confidence: {confidence:.3f}")
                        st.write(f"Threshold: {threshold}")
                        st.write(f"Raw confidence scores: {all_scores}")
                    
                    # Log the result
                    log_violation(uploaded_file.name, confidence, violation_type)
                    
                    # Additional info
                    if confidence < threshold:
                        st.info(f"ℹ️ Detection confidence ({confidence:.1%}) is below threshold ({threshold:.1%})")
    
    elif mode == "📹 Live Camera":
        st.header("📹 Live Camera Detection")
        st.info("📱 Use your device camera for real-time helmet detection")
        
        # Camera input
        camera_input = st.camera_input("📷 Take a picture for helmet detection")
        
        if camera_input is not None:
            image = Image.open(camera_input)
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.subheader("📸 Captured Image")
                st.image(image, use_column_width=True)
            
            # Convert to array and predict
            img_array = np.array(image)
            
            with st.spinner("🔍 Processing..."):
                predicted_class, confidence, all_scores = predict_helmet(img_array, model)
            
            if predicted_class is not None:
                with col2:
                    st.subheader("⚡ Real-time Results")
                    
                    # FIXED: Apply the same class logic as image upload
                    # Class 0 = With Helmet, Class 1 = Without Helmet
                    is_helmet_detected = predicted_class == 0  # Class 0 means helmet detected
                    is_compliant = is_helmet_detected and confidence >= threshold
                    
                    if is_compliant:
                        st.success("✅ **HELMET DETECTED**")
                        # REMOVED: st.balloons() - No more balloon animation
                        violation_type = "Compliant"
                    else:
                        st.error("❌ **NO HELMET DETECTED**")
                        st.warning("🚨 Safety violation!")
                        violation_type = "No Helmet Detected"
                    
                    # Show confidence
                    st.metric("Confidence", f"{confidence:.1%}")
                    
                    # Progress bar for confidence
                    st.progress(confidence)
                    
                    # Debug information for camera too
                    with st.expander("🔍 Debug Information"):
                        st.write(f"Raw predicted class: {predicted_class}")
                        st.write(f"Is helmet detected: {is_helmet_detected}")
                        st.write(f"Confidence: {confidence:.3f}")
                        st.write(f"Threshold: {threshold}")
                        st.write(f"Raw confidence scores: {all_scores}")
                    
                    # Log result
                    timestamp = datetime.now().strftime("%H:%M:%S")
                    log_violation(f"Camera_{timestamp}", confidence, violation_type)
    
    elif mode == "📁 Batch Processing":
        st.header("📁 Batch Image Processing")
        st.info("📤 Upload multiple images for bulk helmet compliance checking")
        
        uploaded_files = st.file_uploader(
            "Upload multiple images", 
            type=['jpg', 'jpeg', 'png', 'bmp'], 
            accept_multiple_files=True,
            help="Select multiple images for batch processing"
        )
        
        if uploaded_files:
            st.write(f"📊 Processing {len(uploaded_files)} images...")
            
            # Process all images
            results = []
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, uploaded_file in enumerate(uploaded_files):
                status_text.text(f"Processing {uploaded_file.name}...")
                
                image = Image.open(uploaded_file)
                img_array = np.array(image)
                
                predicted_class, confidence, _ = predict_helmet(img_array, model)
                
                if predicted_class is not None:
                    # Apply same class logic: Class 0 = With Helmet, Class 1 = Without Helmet
                    is_helmet_detected = predicted_class == 0
                    detected_text = "With Helmet" if predicted_class == 0 else "Without Helmet"
                    
                    is_compliant = is_helmet_detected and confidence >= threshold
                    
                    results.append({
                        'Image': uploaded_file.name,
                        'Status': '✅ Compliant' if is_compliant else '❌ Violation',
                        'Detected': detected_text,
                        'Confidence': f"{confidence:.1%}",
                        'Safe': 'Yes' if is_compliant else 'No'
                    })
                    
                    # Log each result
                    violation_type = "Compliant" if is_compliant else "No Helmet Detected"
                    log_violation(uploaded_file.name, confidence, violation_type)
                
                progress_bar.progress((i + 1) / len(uploaded_files))
            
            status_text.text("✅ Processing complete!")
            
            # Summary statistics
            st.subheader("📊 Batch Processing Summary")
            
            total_images = len(results)
            compliant_images = len([r for r in results if r['Safe'] == 'Yes'])
            violation_images = total_images - compliant_images
            compliance_rate = (compliant_images / total_images) * 100 if total_images > 0 else 0
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Images", total_images)
            col2.metric("✅ Compliant", compliant_images)
            col3.metric("❌ Violations", violation_images)
            col4.metric("📈 Compliance Rate", f"{compliance_rate:.1f}%")
            
            # Detailed results table
            st.subheader("📋 Detailed Results")
            st.dataframe(results, use_container_width=True)
            
            # Download results
            if st.button("📥 Download Results as CSV"):
                import pandas as pd
                df = pd.DataFrame(results)
                csv = df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"helmet_compliance_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
    
    elif mode == "📊 Violation Logs":
        st.header("📊 Safety Violation Logs")
        
        if st.session_state.violation_logs:
            st.write(f"📋 Total logged events: {len(st.session_state.violation_logs)}")
            
            # Convert to dataframe for better display
            import pandas as pd
            df = pd.DataFrame(st.session_state.violation_logs)
            
            # Summary stats
            violations = df[df['violation'] != 'Compliant']
            
            if len(violations) > 0:
                st.error(f"⚠️ {len(violations)} safety violations detected!")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("🚨 Recent Violations")
                    st.dataframe(violations.tail(10), use_container_width=True)
                
                with col2:
                    # Violation timeline (simple)
                    violation_counts = violations.groupby(violations['timestamp'].str[:10]).size()
                    if len(violation_counts) > 0:
                        st.subheader("📈 Violations by Date")
                        st.bar_chart(violation_counts)
            
            # Full log
            st.subheader("📝 Complete Log")
            st.dataframe(df, use_container_width=True)
            
            # Clear logs button
            if st.button("🗑️ Clear All Logs", type="secondary"):
                st.session_state.violation_logs = []
                st.success("Logs cleared!")
                st.rerun()
        else:
            st.info("📭 No violation logs yet. Start detecting to see logs here.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray;'>
        <p>🏭 <strong>Helmet Compliance Monitoring System</strong></p>
        <p>Ensuring workplace safety through AI-powered detection</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()