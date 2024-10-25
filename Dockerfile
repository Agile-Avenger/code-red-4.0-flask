# Use the official Python image as the base image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy requirements.txt file to the working directory
COPY requirements.txt .

# Install the required packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire Flask app into the container
COPY . .

# Copy model files to the container
COPY models/ ./models/

# Expose the port the app runs on
EXPOSE 5000

# Set the environment variable for Flask
ENV FLASK_APP=app.py

# Command to run the application
CMD ["flask", "run", "--host=0.0.0.0"]
