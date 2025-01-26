# Base image
FROM python:3.9-slim

# Working directory
WORKDIR /app

# Copy application code and Model
COPY . app/app.py /app/
COPY . model/model.pkl /app/
# Install dependencies
RUN pip install --no-cache-dir flask numpy scikit-learn

# Expose port
EXPOSE 80

# Run the Flask application
CMD ["python", "app/app.py"]
