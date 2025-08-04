FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install torch
RUN pip install torchvision
RUN pip install python-abc
RUN pip install numpy
RUN pip install scikit-learn
RUN pip install wandb
RUN pip install pandas
RUN pip install matplotlib
RUN pip install opencv-python-headless
RUN pip install Pillow
RUN pip install scipy
RUN pip install flask
RUN pip install flask-cors

WORKDIR /workspace
COPY . /workspace

# Expose port for Flask API
EXPOSE 5000

# Start the API service
CMD ["python", "src/app/main.py"]

