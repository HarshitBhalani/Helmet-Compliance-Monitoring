import streamlit as st
# import tensorflow as tf
import tflite_runtime.interpreter as tflite
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
# Convert H5 to TensorFlow Lite
def convert_h5_to_tflite():
    """Convert H5 model to TensorFlow Lite format - Run this once"""
    try:
        # Load the .h5 model
        model = tflite.keras.models.load_model("model.h5")
        
        # Convert the model to TensorFlow Lite format
        converter = tflite.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()
        
        # Save the .tflite model
        with open("model.tflite", "wb") as f:
            f.write(tflite_model)
        
        print("✅ Conversion complete: model.tflite saved")
        return True
    except Exception as e:
        print(f"❌ Conversion failed: {str(e)}")
        return False

# Load the TensorFlow Lite model
@st.cache_resource
def load_model():
    try:
        # Check if .tflite model exists, if not convert from .h5
        if not os.path.exists('model.tflite'):
            st.info("🔄 Converting H5 model to TensorFlow Lite...")
            if not convert_h5_to_tflite():
                st.error("❌ Failed to convert model")
                return None
        
        # Load the TensorFlow Lite model
        # interpreter = tflite.lite.Interpreter(model_path='model.tflite')
        interpreter = tflite.Interpreter(model_path="model.tflite")

        interpreter.allocate_tensors()
        
        return interpreter
    except FileNotFoundError:
        st.error("❌ model.h5 file not found! Please ensure model.h5 is in the same folder as app.py")
        return None
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None

# FIXED: Improved prediction function with better class handling
# FIXED: Improved prediction function with TensorFlow Lite
def predict_helmet(image, interpreter):
    try:
        # Get input and output details
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        # Resize image to model input size (usually 224x224)
        input_shape = input_details[0]['shape']
        img_height, img_width = input_shape[1], input_shape[2]
        
        img = cv2.resize(image, (img_width, img_height))
        
        # Convert to RGB if needed
        if len(img.shape) == 3 and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Normalize pixel values to 0-1
        img = img.astype(np.float32) / 255.0
        
        # Add batch dimension
        img = np.expand_dims(img, axis=0)
        
        # Set input tensor
        interpreter.set_tensor(input_details[0]['index'], img)
        
        # Run inference
        interpreter.invoke()
        
        # Get prediction results
        prediction = interpreter.get_tensor(output_details[0]['index'])
        confidence_scores = prediction[0]
        predicted_class = np.argmax(confidence_scores)
        max_confidence = float(np.max(confidence_scores))
        
        return predicted_class, max_confidence, confidence_scores
        
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None, None, None

# CRITICAL FIX: Corrected class interpretation function
def interpret_prediction(predicted_class, confidence_scores, threshold=0.7):
    """
    FIXED: Proper interpretation of helmet detection
    Class 0 = Helmet (SAFE)
    Class 1 = No Helmet (VIOLATION)
    """
    
    # CORRECT class mapping
    CLASS_MAPPING = {0: "Helmet", 1: "No Helmet"}
    
    class_name = CLASS_MAPPING.get(predicted_class, "Unknown")
    max_confidence = float(np.max(confidence_scores))
    
    # FIXED: Correct helmet detection logic
    is_helmet_detected = (predicted_class == 0)  # Class 0 = Helmet
    is_no_helmet = (predicted_class == 1)        # Class 1 = No Helmet
    
    # FIXED: Proper compliance determination
    is_compliant = is_helmet_detected and max_confidence >= threshold
    
    # FIXED: Correct status messages
    if is_helmet_detected and max_confidence >= threshold:
        status_message = "✅ HELMET DETECTED - SAFE"
        status_color = "green"
    elif is_helmet_detected and max_confidence < threshold:
        status_message = f"⚠️ HELMET DETECTED - LOW CONFIDENCE ({max_confidence:.1%})"
        status_color = "yellow"
    elif is_no_helmet and max_confidence >= threshold:
        status_message = "❌ NO HELMET DETECTED - VIOLATION"
        status_color = "red"
    else:
        status_message = f"⚠️ UNCERTAIN DETECTION - LOW CONFIDENCE ({max_confidence:.1%})"
        status_color = "yellow"
    
    return is_helmet_detected, is_compliant, status_message, class_name, CLASS_MAPPING, status_color

# Initialize session state for violation logs
if 'violation_logs' not in st.session_state:
    st.session_state.violation_logs = []

def log_violation(image_name, confidence, violation_type, class_name):
    """Log safety violations with more details"""
    log_entry = {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'image': image_name,
        'detected_class': class_name,
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
        # Check what files exist
        st.subheader("📁 Files in Current Directory:")
        current_files = [f for f in os.listdir('.') if os.path.isfile(f)]
        
        if current_files:
            for file in current_files:
                if file.endswith('.h5'):
                    st.success(f"✅ Found H5 model: {file}")
                elif file.endswith('.tflite'):
                    st.success(f"✅ Found TFLite model: {file}")
                else:
                    st.write(f"📄 {file}")
        else:
            st.write("No files found")
        
        st.markdown("""
        **🔧 Fix Steps:**
        1. Ensure your `model.h5` file is in the same folder as `app.py`
        2. The app will automatically convert it to `model.tflite` format
        3. File should be named exactly `model.h5`
        4. Restart the Streamlit app after placing the file
        """)
        return
    
    # Success message
    # st.success("✅ Model loaded successfully!")f
    st.success("✅ TensorFlow Lite model loaded successfully!")
    
    # Model info
    # Model info
    with st.expander("📊 Model Information"):
        try:
            input_details = model.get_input_details()
            output_details = model.get_output_details()
            st.write(f"**Input Shape:** {input_details[0]['shape']}")
            st.write(f"**Output Shape:** {output_details[0]['shape']}")
            st.write(f"**Model Type:** TensorFlow Lite")
            st.write(f"**Input Type:** {input_details[0]['dtype']}")
            st.write(f"**Output Type:** {output_details[0]['dtype']}")
        except:
            st.write("Model information not available")
    
    # Sidebar controls
    st.sidebar.title("🎛️ Detection Controls")
    
    # Detection threshold
    threshold = st.sidebar.slider("Detection Threshold", 0.1, 0.95, 0.7, 0.05)
    st.sidebar.info(f"Current threshold: {threshold}")
    
    # ADDED: Confidence validation
    st.sidebar.markdown("**Confidence Threshold Guide:**")
    st.sidebar.markdown("- 0.9+ = Very High Confidence")
    st.sidebar.markdown("- 0.7-0.9 = High Confidence")
    st.sidebar.markdown("- 0.5-0.7 = Medium Confidence")
    st.sidebar.markdown("- <0.5 = Low Confidence")
    
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
        compliance_rate = ((total_checks - violations) / total_checks) * 100 if total_checks > 0 else 0
        
        st.sidebar.metric("Total Checks", total_checks)
        st.sidebar.metric("Compliance Rate", f"{compliance_rate:.1f}%")
        st.sidebar.metric("Violations", violations)
    
    # FIXED: Correct class mapping
    current_mapping = {0: "Helmet", 1: "No Helmet"}
    
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
                    
                    # FIXED: Use corrected interpretation
                    is_helmet_detected, is_compliant, status_message, class_name, _, status_color = interpret_prediction(
                        predicted_class, all_scores, threshold
                    )
                    
                    # Show all confidence scores with proper labels
                    st.write("**Confidence Scores:**")
                    for i, score in enumerate(all_scores):
                        label = current_mapping.get(i, f"Class {i}")
                        if i == predicted_class:
                            st.write(f"**{label}: {score:.3f} ({score*100:.1f}%)**")
                        else:
                            st.write(f"{label}: {score:.3f} ({score*100:.1f}%)")
                    
                    st.markdown("---")
                    
                    # FIXED: Correct status display based on actual detection
                    if status_color == "green":
                        st.success(status_message)
                        st.metric("Status", "SAFE", "✓")
                        violation_type = "Compliant"
                    elif status_color == "yellow":
                        st.warning(status_message)
                        st.metric("Status", "UNCERTAIN", "⚠️")
                        violation_type = "Low Confidence Detection"
                    else:  # red
                        st.error(status_message)
                        st.metric("Status", "UNSAFE", "⚠️")
                        violation_type = "No Helmet Detected"
                        
                        # Safety alert for violations only
                        st.warning("🚨 **SAFETY VIOLATION ALERT**")
                        st.markdown("""
                        **Immediate Actions Required:**
                        - 🛑 Stop work immediately
                        - 🪖 Provide safety helmet
                        - 📋 Brief worker on safety protocols
                        - 📝 Document the incident
                        """)
                    
                    st.metric("Confidence Level", f"{confidence:.1%}")
                    st.metric("Detected Class", class_name)
                    
                    # ENHANCED: Debug information
                    with st.expander("🔍 Debug Information"):
                        st.write(f"**Predicted class index:** {predicted_class}")
                        st.write(f"**Detected class name:** {class_name}")
                        st.write(f"**Is helmet detected:** {is_helmet_detected}")
                        st.write(f"**Confidence:** {confidence:.3f}")
                        st.write(f"**Threshold:** {threshold}")
                        st.write(f"**Is compliant:** {is_compliant}")
                        st.write(f"**Class mapping:** {current_mapping}")
                        st.write(f"**Raw confidence scores:** {all_scores}")
                    
                    # Log the result
                    log_violation(uploaded_file.name, confidence, violation_type, class_name)
                    
                    # Additional warnings
                    if confidence < 0.6:
                        st.error("⚠️ **LOW CONFIDENCE WARNING**: Model prediction may be unreliable")
                    elif confidence < threshold:
                        st.info(f"ℹ️ Detection confidence ({confidence:.1%}) is below threshold ({threshold:.1%})")
    
    elif mode == "📹 Live Camera":
        st.header("📹 Live Camera Detection")
        st.info("📱 Use your device camera for real-time helmet detection")
        
        # Camera controls
        st.markdown("### Camera Controls")
        col_cam1, col_cam2 = st.columns([1, 1])
        
        with col_cam1:
            # Auto-refresh option
            auto_refresh = st.checkbox("🔄 Auto-refresh detection", value=False)
            if auto_refresh:
                st.info("Camera will automatically retake photos every 3 seconds")
        
        with col_cam2:
            # Manual refresh button
            if st.button("🔄 Refresh Camera", type="primary"):
                st.rerun()
        
        # Camera input with error handling
        try:
            camera_input = st.camera_input(
                "📷 Take a picture for helmet detection",
                key="helmet_camera"
            )
        except Exception as e:
            st.error(f"❌ Camera access error: {str(e)}")
            st.info("💡 Try refreshing the page or checking camera permissions")
            camera_input = None
        
        if camera_input is not None:
            try:
                image = Image.open(camera_input)
                
                # Validate image
                if image.size[0] < 50 or image.size[1] < 50:
                    st.error("❌ Image too small for detection")
                    return
                
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.subheader("📸 Live Camera Feed")
                    #st.image(image, use_container_width=True)
                    st.image(image, width=700)  # You can adjust width to suit your layout
                    st.caption(f"Size: {image.size[0]}x{image.size[1]} pixels")
                
                # CRITICAL FIX: Proper image preprocessing for camera
                img_array = np.array(image)
                
                # Handle different image formats from camera
                if len(img_array.shape) == 3:
                    if img_array.shape[2] == 4:  # RGBA
                        img_array = img_array[:, :, :3]  # Remove alpha channel
                    elif img_array.shape[2] == 1:  # Grayscale
                        img_array = np.repeat(img_array, 3, axis=2)  # Convert to RGB
                
                # Real-time processing indicator
                with st.spinner("🔍 Live Detection Processing..."):
                    predicted_class, confidence, all_scores = predict_helmet(img_array, model)
                
                if predicted_class is not None:
                    with col2:
                        st.subheader("⚡ Live Detection Results")
                        
                        # CRITICAL FIX: Use corrected interpretation for live camera
                        is_helmet_detected, is_compliant, status_message, class_name, _, status_color = interpret_prediction(
                            predicted_class, all_scores, threshold
                        )
                        
                        # ENHANCED: Clear status display with CORRECT colors
                        if status_color == "green":
                            st.success(status_message)
                            st.markdown("🟢 **STATUS: SAFE - HELMET DETECTED**")
                            violation_type = "Compliant"
                        elif status_color == "yellow":
                            st.warning(status_message)
                            st.markdown("🟡 **STATUS: UNCERTAIN - CHECK DETECTION**")
                            violation_type = "Low Confidence Detection"
                        else:  # red
                            st.error(status_message)
                            st.markdown("🔴 **STATUS: DANGER - NO HELMET**")
                            
                            # IMMEDIATE ALERT for violations ONLY when no helmet
                            st.markdown("""
                            <div style='background-color: #ff4444; color: white; padding: 10px; border-radius: 5px; margin: 10px 0;'>
                            🚨 <strong>IMMEDIATE ACTION REQUIRED</strong><br>
                            🛑 STOP WORK ACTIVITY<br>
                            🪖 PROVIDE SAFETY HELMET<br>
                            📞 NOTIFY SUPERVISOR
                            </div>
                            """, unsafe_allow_html=True)
                            violation_type = "No Helmet Detected"
                        
                        # ENHANCED: Live metrics display
                        col_met1, col_met2 = st.columns(2)
                        with col_met1:
                            st.metric("🎯 Confidence", f"{confidence:.1%}", f"{confidence-threshold:.1%}")
                        with col_met2:
                            st.metric("📊 Class", class_name)
                        
                        # LIVE: Confidence visualization with correct colors
                        if status_color == "green":
                            confidence_color = "#28a745"  # Green for safe
                        elif status_color == "yellow":
                            confidence_color = "#ffc107"  # Yellow for uncertain
                        else:
                            confidence_color = "#dc3545"  # Red for danger
                            
                        st.markdown(f"""
                        <div style='background: linear-gradient(90deg, {confidence_color} {confidence*100:.0f}%, #ddd {confidence*100:.0f}%); 
                                   height: 20px; border-radius: 10px; margin: 10px 0;'></div>
                        <p style='text-align: center; margin: 0;'>Confidence: {confidence:.1%}</p>
                        """, unsafe_allow_html=True)
                        
                        # Real-time scores
                        st.markdown("**📊 Live Class Scores:**")
                        for i, score in enumerate(all_scores):
                            label = current_mapping.get(i, f"Class {i}")
                            if i == 0:  # Helmet class
                                bar_color = "#28a745" if score > 0.5 else "#dc3545"
                            else:  # No helmet class
                                bar_color = "#dc3545" if score > 0.5 else "#28a745"
                            
                            is_predicted = (i == predicted_class)
                            st.markdown(f"""
                            <div style='margin: 5px 0;'>
                                <strong>{'>>> ' if is_predicted else ''}{label}:</strong> {score:.3f} ({score*100:.1f}%)
                                <div style='background: {bar_color}; width: {score*100:.0f}%; height: 8px; border-radius: 4px;'></div>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # CRITICAL: Safety status summary
                        if not is_compliant:
                            st.markdown("---")
                            current_time = datetime.now().strftime("%H:%M:%S")
                            if predicted_class == 1:  # No helmet detected
                                st.error(f"⏰ SAFETY VIOLATION ALERT at {current_time}")
                            else:
                                st.warning(f"⏰ Low Confidence Alert at {current_time}")
                            
                            if confidence < 0.5:
                                st.warning("⚠️ Very low confidence - Consider retaking photo")
                        else:
                            current_time = datetime.now().strftime("%H:%M:%S")
                            st.success(f"✅ Safety Check Passed at {current_time}")
                        
                        # Enhanced debug for live camera
                        with st.expander("🔍 Live Debug Information"):
                            st.json({
                                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                "predicted_class_index": int(predicted_class),
                                "detected_class_name": class_name,
                                "is_helmet_detected": is_helmet_detected,
                                "confidence": round(float(confidence), 4),
                                "threshold": threshold,
                                "is_compliant": is_compliant,
                                "status_color": status_color,
                                "class_mapping": current_mapping,
                                "image_shape": img_array.shape,
                                "all_scores": [round(float(x), 4) for x in all_scores]
                            })
                        
                        # Log result with timestamp
                        timestamp = datetime.now().strftime("%H:%M:%S")
                        log_violation(f"LiveCamera_{timestamp}", confidence, violation_type, class_name)
                        
                        # Auto-refresh functionality
                        if auto_refresh:
                            import time
                            time.sleep(3)
                            st.rerun()
                            
                else:
                    st.error("❌ Detection failed - Please retake photo")
                    
            except Exception as e:
                st.error(f"❌ Camera processing error: {str(e)}")
                st.info("💡 Try taking another photo or refresh the page")
        else:
            # Better guidance when no camera input
            st.info("📷 Click the camera button above to start detection")
            st.markdown("""
            **Camera Tips:**
            - Ensure good lighting
            - Keep person's head clearly visible
            - Hold camera steady
            - Allow camera permissions if prompted
            
            **Expected Results:**
            - 🟢 **Green (Safe)**: When helmet is detected
            - 🔴 **Red (Violation)**: When no helmet is detected
            - 🟡 **Yellow (Uncertain)**: When confidence is low
            """)
    
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
                
                predicted_class, confidence, all_scores = predict_helmet(img_array, model)
                
                if predicted_class is not None:
                    # FIXED: Use corrected interpretation for batch processing
                    is_helmet_detected, is_compliant, status_message, class_name, _, status_color = interpret_prediction(
                        predicted_class, all_scores, threshold
                    )
                    
                    if status_color == "green":
                        status = '✅ Compliant'
                        violation_type = "Compliant"
                    elif status_color == "yellow":
                        status = '⚠️ Low Confidence'
                        violation_type = "Low Confidence Detection"
                    else:  # red
                        status = '❌ Violation'
                        violation_type = "No Helmet Detected"
                    
                    results.append({
                        'Image': uploaded_file.name,
                        'Status': status,
                        'Detected': class_name,
                        'Confidence': f"{confidence:.1%}",
                        'Safe': 'Yes' if is_compliant else 'No'
                    })
                    
                    # Log each result
                    log_violation(uploaded_file.name, confidence, violation_type, class_name)
                
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
    
    # ENHANCED: Model testing section
    st.markdown("---")
    with st.expander("🧪 Model Testing & Validation"):
        st.markdown("""
        **FIXED: Corrected Detection Logic**
        
        **How the detection now works:**
        - 🟢 **Class 0 (Helmet)** → GREEN status → SAFE
        - 🔴 **Class 1 (No Helmet)** → RED status → VIOLATION
        
        **To verify your model is working correctly:**
        
        1. **Test with person wearing helmet** → should show GREEN "HELMET DETECTED - SAFE"
        2. **Test with person not wearing helmet** → should show RED "NO HELMET DETECTED - VIOLATION"
        3. **Check confidence scores** → should be >0.7 for reliable predictions
        
        **If results are still wrong:**
        - Your model might have reversed class labels during training
        - Check if your training data was labeled correctly
        - Verify image quality and lighting conditions
        """)
        
        # ADDED: Quick test section
        st.markdown("**🔍 Quick Test Guide:**")
        st.markdown("""
        1. Take a photo wearing a helmet → Should see: 🟢 SAFE
        2. Take a photo without helmet → Should see: 🔴 VIOLATION
        3. If reversed, your model needs retraining with correct labels
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray;'>
        <p>🏭 <strong>Helmet Compliance Monitoring System</strong></p>
        <p>Ensuring workplace safety through AI-powered detection</p>
       
    </div>
    """, unsafe_allow_html=True)
    #  <p><em>⚠️ FIXED: Now correctly identifies helmet presence/absence</em></p>

if __name__ == "__main__":
    main()
