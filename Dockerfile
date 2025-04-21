# Use an official Python base image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy requirements first (to leverage Docker caching)
COPY requirements.txt .

# Install dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -v -r requirements.txt

# Copy the rest of the app code
COPY . .

# Command to run your app (change this if your entry point is different)
CMD ["python", "app.py"]