FROM python:3.12-slim

WORKDIR "/app/Stage 2/starter-kit"

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY ["Stage 2/starter-kit/requirements.txt", "./"]
RUN pip install --no-cache-dir -r requirements.txt

COPY ["Stage 2/starter-kit/", "./"]

CMD ["uvicorn", "api_docs.app:app", "--host", "0.0.0.0", "--port", "8000"]
