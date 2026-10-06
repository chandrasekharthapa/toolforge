FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml README.md ./
COPY toolforge ./toolforge
RUN pip install --no-cache-dir ".[all]" && useradd --create-home forge
USER forge
ENV TOOLFORGE_DB=/home/forge/toolforge.db
EXPOSE 8000
CMD ["toolforge", "serve", "--host", "0.0.0.0", "--port", "8000"]
