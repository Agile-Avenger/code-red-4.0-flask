import requests
import time
import os

BASE_DIR = os.path.join("D:", os.sep, "aa", "flask", "images")
pneumonia_files = [f"{BASE_DIR}/pneumonia/{i}.jpeg" for i in range(1, 2)]

# Define the endpoints
pneumonia_endpoint = "https://flask-app-616464352400.us-central1.run.app/generate-pneumonia-report"


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


report = test_model(pneumonia_endpoint, pneumonia_files)

# Save the report to a .txt file
report_file_path = os.path.join(BASE_DIR, "report.txt")
with open(report_file_path, "w") as report_file:
    for file_path, response_json, elapsed_time in report:

        report_file.write(f"{response_json}\n")

print(f"Report saved to: {report_file_path}")
