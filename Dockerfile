# Use an official lightweight Python image
FROM python:3.11-slim


WORKDIR /app


COPY requirements.txt .


RUN pip install --no-cache-dir -r requirements.txt


COPY . .


EXPOSE 8080

ENV FLASK_APP=backend_api.py
ENV FLASK_RUN_HOST=0.0.0.0
ENV PORT=8080

CMD ["flask", "run", "--host=0.0.0.0", "--port=8080"]
