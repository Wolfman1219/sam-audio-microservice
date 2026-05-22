# Per-service image. Inherits from sam-audio-base and just lays the pipeline
# code on top. The SERVICE arg selects which uvicorn entrypoint to run.
#
# Built per service via docker-compose.

ARG BASE_IMAGE=sam-audio-base:latest
FROM ${BASE_IMAGE}

ARG SERVICE
ENV SERVICE=${SERVICE}

COPY pipeline /app/pipeline

# All services listen on port 8000 inside their container; compose maps them
# differently on the host if you want host-side access (we usually don't).
EXPOSE 8000

# uvicorn picks the FastAPI app from pipeline.services.<service>.app:app
CMD ["sh", "-c", "exec uvicorn pipeline.services.${SERVICE}.app:app \
        --host 0.0.0.0 --port 8000 \
        --workers 1 \
        --log-level info \
        --no-access-log"]
