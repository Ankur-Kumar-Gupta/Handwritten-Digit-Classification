# Handwritten-Digit-Classification
Machine learning project for handwritten digit classification using the MNIST dataset, implemented in Python with TensorFlow. Trained a model to achieve high accuracy in identifying digits (0-9) and evaluated its performance through testing and prediction.

This project demonstrates a basic machine learning model for handwritten digit classification using the MNIST dataset. The goal is to train a model that can accurately identify the digits from 0 to 9 based on images of handwritten digits.

The MNIST dataset consists of 60,000 training images and 10,000 test images, each with a resolution of 28x28 pixels. The pixel values range from 0 to 255 and represent the grayscale intensity of the image.

The project is implemented in Python using the TensorFlow library. TensorFlow is a popular deep learning framework that provides a high-level API for building and training machine learning models.

The main steps of the project include:

1. **Dataset Preparation**: The MNIST dataset is downloaded using the `mnist.load_data()` function from the `tensorflow.keras.datasets` module. The dataset is split into training and testing sets.

2. **Data Preprocessing**: The pixel values of the images are normalized between 0 and 1 by dividing each pixel value by 255. This normalization step helps improve the training process.

3. **Model Building**: The machine learning model is constructed using the Sequential API from TensorFlow's `tf.keras` module. The model architecture consists of a single flatten layer to convert the 2D image into a 1D vector, followed by a dense layer with ReLU activation and 128 neurons, and a final dense layer with softmax activation and 10 neurons representing the output classes (digits 0 to 9).

4. **Model Compilation and Training**: The model is compiled with the Adam optimizer, sparse categorical cross-entropy loss function, and accuracy as the evaluation metric. The `model.fit()` function is used to train the model on the training dataset for a specified number of epochs and batch size.

5. **Model Evaluation**: After training, the model is evaluated on the test dataset using the `model.evaluate()` function. The test loss and accuracy values are calculated and displayed.

6. **Prediction**: Finally, the trained model is used to make predictions on a subset of the test dataset. The `model.predict()` function is applied to obtain the predicted labels for the input images.

By following this project, you can understand the basic steps involved in training a machine learning model for image classification. This serves as a foundation for more complex projects and provides a starting point for exploring advanced techniques in deep learning.

Feel free to customize and extend the project according to your requirements and further experiment with different models, architectures, or hyperparameters.
