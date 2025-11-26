# Use a basic Python image with version 3.10 as the starting environment
FROM python:3.10-slim

# Set a directory inside the container called /app
WORKDIR /app

# Copy the requirements.txt file to install any needed packages
COPY requirements.txt requirements.txt

# Install the packages listed in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Force-remove GUI OpenCV builds and install clean headless contrib version
RUN pip uninstall -y opencv-python opencv-contrib-python || true \
 && pip install --no-cache-dir opencv-contrib-python-headless

# Copy the rest of the files (your Gradio app code) into the container
COPY . .

# Open ports for Gradio and Prometheus metrics
EXPOSE 7860
EXPOSE 8000

# Command to run your Gradio app
CMD ["python", "deploy.py"]
