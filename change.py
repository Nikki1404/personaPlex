FROM python:3.11-slim

ARG USE_PROXY=false

RUN if [ "$USE_PROXY" = "true" ]; then \
        echo " Enabling proxy"; \
        export http_proxy="http://163.116.128.80:8080"; \
        export https_proxy="http://163.116.128.80:8080"; \
        ENV http_proxy="http://163.116.128.80:8080"; \
        ENV https_proxy="http://163.116.128.80:8080"; \
    else \
        echo " Proxy disabled"; \
    fi

WORKDIR /srv

COPY download_model/nemotron-speech-streaming/nemotron-speech-streaming-en-0.6b.nemo nemotron-speech-streaming-en-0.6b.nemo

COPY requirements.txt requirements.txt

RUN pip install --upgrade pip setuptools wheel
RUN python3 -m pip install --no-cache-dir -r requirements.txt

COPY app app

COPY app/google_credentials.json google_credentials.json


ENV GOOGLE_APPLICATION_CREDENTIALS=/srv/google_credentials.json
ENV GOOGLE_RECOGNIZER=projects/eci-ugi-digital-ccaipoc/locations/us-central1/recognizers/google-stt-default
ENV GOOGLE_REGION=us-central1
ENV GOOGLE_LANGUAGE=en-US
ENV GOOGLE_MODEL=latest_short
ENV GOOGLE_INTERIM=true
ENV GOOGLE_EXPLICIT_DECODING=true

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8002"]
