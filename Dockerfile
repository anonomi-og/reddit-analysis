# Use an official Python runtime as a parent image
FROM python:3.9

# Set the working directory in the container
WORKDIR /app

# Copy current directory contents into the container
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port 8080 (Cloud Run default)
EXPOSE 8080

# Run the application
CMD ["gunicorn", "-b", "0.0.0.0:8080", "app:app"]
