# Use a minimal Python 3.9 image
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Step 1: Copy requirements.txt and install dependencies
COPY src/requirements.txt /app/requirements.txt
RUN ls -al /app && cat /app/requirements.txt  # Debugging step
RUN pip install --no-cache-dir -r /app/requirements.txt && pip list

# Step 2: Copy the application files
COPY src/ /app/

# Step 3: Copy the model weights file
COPY resnet18_carparts.pth /app/

# Step 4: Debug - List files after copying
RUN ls -al /app

# Expose Flask port
EXPOSE 5000

# Run the Flask app
CMD ["python", "app.py"]
