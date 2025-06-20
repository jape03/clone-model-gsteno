# Base image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# System dependencies (for PIL, TensorFlow, etc.)
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Copy dependencies
COPY requirements.txt .

# Install Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy your app code
COPY . .

# Expose the port that Railway will use
EXPOSE 10000

# Start the FastAPI app using uvicorn
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "10000"]
