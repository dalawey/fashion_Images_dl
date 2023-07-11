
FROM python:3.9.7-slim

# Create dirs and copy the sources

RUN mkdir -p /app

WORKDIR /app 

COPY . .

RUN python -m pip install --upgrade pip 

RUN pip install -r requirements.txt

# Set docker launch entry point

CMD ["python", "api.py"]
