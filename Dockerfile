FROM python:3.8-slim

WORKDIR /app

COPY . /app

RUN pip install -r requirements.txt
RUN pip install mlflow

EXPOSE 5000

CMD ["python", "src/app.py"]