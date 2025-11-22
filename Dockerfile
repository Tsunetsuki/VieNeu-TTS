# syntax=docker/dockerfile:1
FROM python:3.10-alpine
WORKDIR /code
# RUN apk add --no-cache gcc musl-dev linux-headers
RUN apk add --no-cache espeak
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
EXPOSE 8000
COPY . .
CMD ["fastapi", "dev", "endpoint.py", "--host", "0.0.0.0"]

