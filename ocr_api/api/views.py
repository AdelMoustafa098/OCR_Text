from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import json

# Load the model (assume it's in the current directory)
model = load_model("/home/adel-moustafa/github_repo/OCR_Text/crnn_ocr_model_cross_entropy.keras")

@csrf_exempt
def predict(request):
    if request.method == "POST":
        try:
            # Get image file from request
            file = request.FILES.get("image")
            if not file:
                return JsonResponse({"error": "No image provided"}, status=400)

            # Read image
            img_array = np.frombuffer(file.read(), np.uint8)
            image = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)
            image = cv2.resize(image, (500, 80))  # Resize to match input dimensions
            image = np.expand_dims(image, axis=(0, -1))  # Add batch and channel dimensions

            # Predict
            predictions = model.predict(image)
            decoded = np.argmax(predictions, axis=-1)
            decoded_text = "".join([CHARACTERS[idx] for idx in decoded[0]])

            return JsonResponse({"text": decoded_text})
        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)

    return JsonResponse({"error": "GET method not allowed"}, status=405)
