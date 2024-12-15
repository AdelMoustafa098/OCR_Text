# OCR Text Recognition Project

## Introduction

This project is an Optical Character Recognition (OCR) system designed to process images and extract text from them. It utilizes a Convolutional Recurrent Neural Network (CRNN) architecture, providing accurate predictions of text in both English and Arabic scripts. The project includes components for preprocessing data, training the model, converting it to ONNX and TensorRT formats, and deploying the model using a Django API.


Key features:

- CRNN model architecture combining convolutional layers for feature extraction and recurrent layers for sequence modeling.
- Compatibility with ONNX and TensorRT for efficient inference.
- Django-based REST API for deploying the OCR model.
- End-to-end pipeline for training, testing, and deployment.  

## Installation

Follow these steps to set up the project on your local machine:  

1. Clone the Repository
    ```bash
    git clone https://github.com/AdelMoustafa098/ocr-text-recognition.git
    cd ocr-text-recognition

    ```

2. Set Up a Virtual Environment
    ```bash
    python3 -m venv ocr_env
    source ocr_env/bin/activate
    ```

3. Install Dependencies Use the provided Makefile for quick installation:

    ```bash
    make install
    ```

4. Run Django Server

    ```bash
    cd ocr_api
    python manage.py runserver

    ```
Access the API at http://127.0.0.1:8000/api/predict/

## Repository Structure
```graphql
    ocr-text-recognition/
├── Tests/                         # Unit and integration tests for the project
├── ocr_api/                       # Django application for deploying the OCR model
│   ├── api/                       # API application folder
│   │   ├── migrations/            # Django database migrations
│   │   ├── __init__.py            # Init file for the API application
│   │   ├── admin.py               # Admin panel configuration
│   │   ├── apps.py                # App configuration
│   │   ├── models.py              # Django models (if any)
│   │   ├── tests.py               # Test cases for the API
│   │   ├── urls.py                # URL routing for the API
│   │   ├── views.py               # API logic and endpoints
│   ├── ocr_api/                   # Django project folder
│       ├── __init__.py            # Init file for the project
│       ├── asgi.py                # ASGI configuration
│       ├── settings.py            # Django project settings
│       ├── urls.py                # URL routing for the project
│       ├── wsgi.py                # WSGI configuration
│   ├── manage.py                  # Django project management script
├── api_request.py                 # Script to send POST requests to the OCR API
├── config.py                      # Configuration settings for the project
├── data_loader.py                 # Data loading and preprocessing logic
├── main.py                        # Script for inference or specific tasks
├── model.py                       # CRNN model architecture definition
├── onnx_trt_conversion.py         # ONNX and TensorRT conversion script
├── requirements.txt               # Python dependencies for the project
├── text_processor.py              # Text preprocessing and sequence handling logic
├── train.py                       # Model training script
├── utils.py                       # Utility functions for the project
├── Makefile                       # Makefile for quick setup and testing
├── README.md                      # Project documentation
├── LICENSE                        # Project license


```


## Usage


1. Train the Model To train the CRNN model, run:
    ```bash
    python train.py
    ```

2. Convert to ONNX and TensorRT After training, convert the model to ONNX and TensorRT formats for efficient inference:

    ```bash
    python onnx_trt_conversion.py
    ```

3. Deploy Using Django Start the Django server to deploy the OCR API:

    ```bash
    cd ocr_api
    python manage.py runserver
    ```

4. Send Requests Use api_request.py or tools like Postman to send POST requests to the API with an image file.