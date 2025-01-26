# Base image
FROM python:3.8-slim

# Working directory
WORKDIR /app

# Copy application code and Model
COPY . App/app.py /app/
COPY . Model/model.pkl /app/
# Install dependencies
RUN pip install --no-cache-dir flask numpy scikit-learn

# Expose port
EXPOSE 5000

# Run the Flask application
CMD ["python", "App/app.py"]
