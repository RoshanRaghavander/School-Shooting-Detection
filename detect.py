import cv2
import numpy as np
import tensorflow as tf
import streamlit as st
import os
import tempfile

# Load your pre-trained image detection model
model = tf.keras.models.load_model('keras_model.h5')

# Constants
MATCHING_THRESHOLD = 0.5  # Adjust this threshold as needed

def detect_objects(frame: np.ndarray) -> float:
    """
    Detect potential threats in a given frame.

    Args:
        frame (np.ndarray): A grayscale frame from a video feed.

    Returns:
        float: Matching percentage indicating the likelihood of a potential threat.
    """
    # Preprocess the frame for the model
    resized_frame = cv2.resize(frame, (224, 224))
    normalized_frame = resized_frame / 255.0
    input_tensor = np.expand_dims(normalized_frame, axis=0)

    # Perform prediction using the loaded model
    prediction = model.predict(input_tensor)

    # Extract the predicted label (assuming binary classification)
    predicted_label = np.argmax(prediction, axis=1)[0]

    # Use the confidence of the predicted label as the matching percentage
    matching_percentage = prediction[0][predicted_label]

    return matching_percentage

def main():
    st.title('School Shooting Detection System')

    # Upload a video file
    uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi"])

    if uploaded_file is not None:
        # Save the uploaded file to a temporary location
        temp_file_path = os.path.join(tempfile.gettempdir(), "temp_video.mp4")
        with open(temp_file_path, "wb") as temp_file:
            temp_file.write(uploaded_file.read())

        cap = cv2.VideoCapture(temp_file_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        progress_bar = st.progress(0)
        progress_text = st.empty()

        max_matching_percentage = 0
        max_matching_frame = None

        for frame_index in range(total_frames):
            ret, frame = cap.read()

            if not ret:
                break

            # Perform object detection
            matching_percentage = detect_objects(frame)

            # Update progress
            progress_percentage = (frame_index + 1) / total_frames
            progress_bar.progress(progress_percentage)
            progress_text.text(f"Processing: {progress_percentage:.1%}")

            # Process detection results
            if matching_percentage > MATCHING_THRESHOLD and matching_percentage > max_matching_percentage:
                max_matching_percentage = matching_percentage
                max_matching_frame = frame.copy()

        cap.release()

        if max_matching_frame is not None:
            st.warning(f"Highest matching percentage: {max_matching_percentage:.2%}. Potential threat detected! Alerting authorities.")
            # Display the frame with the highest matching percentage
            max_matching_frame_rgb = cv2.cvtColor(max_matching_frame, cv2.COLOR_BGR2RGB)
            st.image(max_matching_frame_rgb, channels="RGB", use_column_width=True)
        else:
            st.info("No potential threat detected.")

if __name__ == "__main__":
    main()
