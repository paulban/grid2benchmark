# Development image for grid2benchmark.
# Python version matches .python-version; dependencies are installed by uv
# from uv.lock in postCreateCommand (see devcontainer.json).
FROM mcr.microsoft.com/devcontainers/python:1-3.12-bookworm

COPY --from=ghcr.io/astral-sh/uv:0.8.15 /uv /uvx /usr/local/bin/

# Use the interpreter that ships with the image instead of downloading one,
# and copy (not hardlink) from the cache since it lives on another filesystem.
ENV UV_PYTHON_PREFERENCE=only-system \
    UV_LINK_MODE=copy
