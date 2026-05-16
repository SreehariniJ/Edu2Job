# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies (needed for scikit-learn/pandas if not precompiled)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the current directory contents into the container at /app
COPY . .

# Ensure the upload folder exists
RUN mkdir -p static/uploads

# Expose port 5000 for the app
EXPOSE 5000

# Run gunicorn when the container launches
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]
