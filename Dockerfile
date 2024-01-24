# FROM --platform=linux/amd64 python:3.11.7-bullseye
FROM python:3.11.7-bookworm
COPY ./requirements.txt /requirements.txt
RUN pip install --no-cache-dir --upgrade -r /requirements.txt
COPY . /code
RUN apt-get update && apt-get install -y --no-install-recommends sqlite3