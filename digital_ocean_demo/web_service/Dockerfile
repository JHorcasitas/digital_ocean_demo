FROM python:3.10-slim AS compile-image
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
COPY requirements.txt .
RUN pip install -r requirements.txt
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

FROM python:3.10-slim AS build-image
COPY --from=compile-image /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
WORKDIR /home/kosmos
RUN mkdir /home/kosmos/logs /home/kosmos/database
COPY wsgi.py .
COPY gunicorn_config.py .
COPY app/ /home/kosmos/app