FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt pyproject.toml ./
COPY src ./src
COPY app ./app
COPY configs ./configs
COPY data ./data
RUN pip install --no-cache-dir -r requirements.txt && pip install -e .
EXPOSE 7860
CMD ["python", "app/gradio_app.py"]
