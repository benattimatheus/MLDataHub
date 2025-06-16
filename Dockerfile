# Use official lightweight Python image
FROM python:3.11.7

# Set working directory
WORKDIR /app

# Copy requirements file if exists
COPY requirements.txt .

# Install system dependencies required for pycaret and others
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libssl-dev \
    libffi-dev \
    python3-dev \
    gcc \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install required Python packages
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy all app files
COPY . .

# Expose Streamlit default port
EXPOSE 8501

# Run the Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
