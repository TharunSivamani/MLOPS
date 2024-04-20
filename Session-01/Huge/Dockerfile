FROM python:3.9-slim-bullseye

WORKDIR /app

COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt 

COPY . .

ENTRYPOINT ["python3", "inference.py"]