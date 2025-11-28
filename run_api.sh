#!/bin/bash

# Local-only API server for development and testing
# Accessible only on the same machine (not exposed to network)

uvicorn app.api:app \
    --host 127.0.0.1 \
    --port 8000 \
    --reload
