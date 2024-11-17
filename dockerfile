# Use the official Python image
FROM python:3.9-slim

# Set a working directory in the container
WORKDIR /app

# Copy the application files into the container
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port your Flask app runs on
EXPOSE 5000

# Set the environment variable to run Flask in production
ENV FLASK_ENV=production

# Command to run the Flask app
CMD ["python", "app.py"]
