FROM python:3.12-slim

WORKDIR /app/solution

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY ["solution/requirements.txt", "./"]
RUN pip install --no-cache-dir -r requirements.txt

COPY ["solution/app.py", "solution/multi_agent_tools.py", "./"]
COPY ["solution/globalcart_agent/", "./globalcart_agent/"]
COPY ["solution/data/", "./data/"]
# Keep the layout expected by globalcart_agent.resolver's mock-service import.
COPY ["Stage 1/starter-kit/mock_services.py", "/app/Stage 1/starter-kit/mock_services.py"]
COPY ["Stage 1/starter-kit/data/", "/app/Stage 1/starter-kit/data/"]

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
