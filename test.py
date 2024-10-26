import requests
import time
import os

BASE_DIR = os.path.join("D:", os.sep, "aa", "flask", "images")
pneumonia_files = [f"{BASE_DIR}/pneumonia/{i}.jpeg" for i in range(1, 11)]
tb_files = [f"{BASE_DIR}/tb/{i}.jpg" for i in range(1, 11)]

# Define the endpoints
pneumonia_endpoint = (
    "https://flask-app-616464352400.us-central1.run.app/predict_pneumonia"
)

tb_endpoint = "https://flask-app-616464352400.us-central1.run.app/predict_tb"


def test_model(endpoint, files):
    results = []
    for file_path in files:
        if os.path.exists(file_path):
            start_time = time.time()  # Start the timer
            with open(file_path, "rb") as f:
                response = requests.post(endpoint, files={"file": f})
            end_time = time.time()  # End the timer

            elapsed_time = end_time - start_time
            results.append((file_path, response.json(), elapsed_time))
        else:
            print(f"File not found: {file_path}")

    return results


a = int(input("Choose 1 or 2\n"))
if a == 1:
    # Test pneumonia model
    print("Testing Pneumonia Model...")
    pneumonia_results = test_model(pneumonia_endpoint, pneumonia_files)

    # Summary of results
    print("\nPneumonia Model Results:")
    for file, prediction, elapsed in pneumonia_results:
        print(f"File: {file}, Prediction: {prediction}, Time: {elapsed:.2f} seconds")
elif a == 2:
    # Test pneumonia model
    print("Testing TB Model...")
    tb_results = test_model(tb_endpoint, tb_files)

    # Summary of results
    print("\nPneumonia Model Results:")
    for file, prediction, elapsed in tb_results:
        print(f"File: {file}, Prediction: {prediction}, Time: {elapsed:.2f} seconds")
else:
    print("juice pila do!! Mausambi ka")
