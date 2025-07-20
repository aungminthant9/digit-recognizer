# Real-time Handwritten Digit Recognition System

This project develops a real-time handwritten digit recognition system using a Convolutional Neural Network (CNN) and deploys it as a web application.

## Project Phases

The project is structured into three main phases:

### Phase 1: Data Preparation & EDA

**Objective**: To load, preprocess, and understand the MNIST dataset.

**Activities**:
-   Loading the MNIST dataset (handwritten digits 0-9).
-   Normalizing pixel values to a [0, 1] range.
-   Reshaping the image data to include a channel dimension, suitable for CNN input.
-   Converting labels to one-hot encoding.
-   Visualizing sample images and the distribution of digits in the dataset.
-   Saving the processed data (`X_train.npy`, `y_train.npy`, `X_test.npy`, `y_test.npy`) for subsequent phases.

**Key Files**: `mnist_data_prep.py`

### Phase 2: CNN Model Development

**Objective**: To design, train, and evaluate a Convolutional Neural Network for digit recognition.

**Activities**:
-   Designing a CNN architecture with multiple convolutional and pooling layers, followed by dense layers.
-   Compiling the model with an appropriate optimizer (Adam) and loss function (categorical cross-entropy).
-   Training the model on the prepared MNIST training data.
-   Implementing callbacks such as Early Stopping (to prevent overfitting) and Model Checkpoint (to save the best performing model).
-   Evaluating the trained model's performance on the test set.
-   Visualizing training history (accuracy and loss plots) and generating a confusion matrix to assess classification performance.
-   Saving the trained model (`mnist_cnn_model.h5`).

**Key Files**: `mnist_model.py`

### Phase 3: Model Deployment (Web App)

**Objective**: To create an interactive web application for real-time digit recognition.

**Activities**:
-   Loading the pre-trained CNN model.
-   Developing a web interface using Gradio, allowing users to draw digits on a canvas.
-   Implementing a preprocessing function to transform the drawn image into the format expected by the model.
-   Integrating the model's prediction logic into the web app to provide real-time predictions, confidence scores, and probability distributions.
-   Providing example images for quick testing.

**Key Files**: `app.py`

### Phase 4: Advanced Model Optimization and Deployment Enhancements

**Objective**: To refine the existing system for improved performance, robustness, and user experience.

**Activities**:
-   **Explore Advanced Architectures**: Investigating and potentially implementing more complex CNN architectures (e.g., LeNet, VGG-like blocks) to achieve higher accuracy or better generalization.
-   **Hyperparameter Tuning**: Systematically tuning hyperparameters (e.g., learning rate, batch size, dropout rate) using techniques like grid search or random search to find optimal configurations.
-   **Data Augmentation**: Implementing data augmentation techniques (e.g., rotation, shifting, zooming) to increase training data diversity and improve model robustness.
-   **Model Quantization/Pruning (Optional)**: Exploring techniques to reduce model size and inference time for deployment efficiency.
-   **Enhanced Web App Features**: Adding advanced features to the web application, such as image upload, confidence thresholds, and feedback mechanisms.

**Key Files**: `mnist_advanced_model.py` (newly created for this phase)

## Technologies Used

-   **Python**: The primary programming language for the entire project.
-   **TensorFlow/Keras**: For building, training, and evaluating the Convolutional Neural Network models.
-   **NumPy**: For numerical operations and efficient handling of array data.
-   **Matplotlib**: For data visualization, including plotting training history, sample images, and confusion matrices.
-   **Scikit-learn**: For data splitting (e.g., `train_test_split`) and evaluation metrics (e.g., `confusion_matrix`, `classification_report`).
-   **Streamlit**: For rapidly building and deploying the interactive web application.
-   **OpenCV (cv2)**: (Potentially) For image processing tasks within the web application, such as resizing and color conversion.

## How to Run the Project

1.  **Clone the repository** (if applicable).
2.  **Navigate to the project directory**:
    ```bash
    cd D:\digit-recognizer
    ```
3.  **Install dependencies** (ensure you have `pip` installed):
    ```bash
    pip install tensorflow numpy matplotlib scikit-learn streamlit opencv-python
    ```
4.  **Run Phase 1 (Data Preparation)**:
    ```bash
    python mnist_data_prep.py
    ```
    This will generate `X_train.npy`, `y_train.npy`, `X_test.npy`, `y_test.npy`.
5.  **Run Phase 2 (Model Training)**:
    ```bash
    python mnist_model.py
    ```
    This will train the initial model and save `mnist_cnn_model.h5`.
6.  **Run Phase 3 (Web Application)**:
    ```bash
    streamlit run app.py
    ```
    This will launch the Streamlit web interface in your browser.

Feel free to explore each script and modify them as needed for further experimentation and improvement.