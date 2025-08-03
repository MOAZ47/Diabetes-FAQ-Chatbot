FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN --mount=type=cache,target=/root/.cache/pip \
    pip --default-timeout=600 install -r requirements.txt

COPY . .

#1RUN rm -rf diabetes_faq

RUN apt-get update && apt-get install -y curl && \
    curl -LsSf https://astral.sh/uv/install.sh | sh

RUN uv tool install crewai



CMD [ "streamlit", "run", "app.py" ]