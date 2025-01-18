FROM python:3.9-slim

WORKDIR /app

# Copy all project files to the container
COPY . .

# Copy the model file into the container (ensure the model is in the 'models' directory)
COPY models/trained_model.pkl /app/models/trained_model.pkl

# Install dependencies from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Expose port 8000 for FastAPI
EXPOSE 8000

# Run the FastAPI app with Uvicorn
CMD ["uvicorn", "app.app:app", "--host", "0.0.0.0", "--port", "8000"]
