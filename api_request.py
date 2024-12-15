import requests

url = "http://127.0.0.1:8000/api/predict/"
file_path = "path/to/your/image.jpg"

with open(file_path, "rb") as img_file:
    response = requests.post(url, files={"image": img_file})

print(response.json())
