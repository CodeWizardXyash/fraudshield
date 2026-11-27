FROM python:3.13-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 MPLCONFIGDIR=/tmp/mpl
WORKDIR /app
COPY requirements.txt ./
RUN pip install -r requirements.txt
COPY pyproject.toml README.md ./
COPY src ./src
COPY configs ./configs
RUN pip install --no-deps .

# Train inside the image so the model matches the installed library versions.
FROM base AS trainer
RUN fraudshield simulate && fraudshield train

FROM base AS runtime
RUN useradd --system --uid 10001 app
COPY --from=trainer /app/models ./models
# Persistent state directory for SQLite
RUN mkdir -p /app/state && chown app:app /app/state
VOLUME ["/app/state"]
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=20s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"
CMD ["fraudshield", "serve", "--host", "0.0.0.0", "--port", "8000", "--state-db", "/app/state/customers.db"]
