FROM --platform=linux/amd64 python:3.8-slim

RUN pip install poetry
RUN apt-get update && apt-get install -y build-essential

COPY engine /opt/sensei-engine
WORKDIR /opt/sensei-engine
RUN pip install  /opt/sensei-engine

COPY api/requirements.txt /opt/sensei-api/requirements.txt
WORKDIR /opt/sensei-api

RUN pip install -r /opt/sensei-api/requirements.txt
COPY api /opt/sensei-api
COPY tests /opt/sensei-tests

EXPOSE 8088

RUN python /opt/sensei-tests/compile-trigger.py
CMD ["uvicorn", "sensei.api:app", "--host", "0.0.0.0", "--port", "8088", "--log-config", "logging.yaml"]
