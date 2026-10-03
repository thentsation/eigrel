FROM python:3.14-slim AS builder

WORKDIR /app

COPY config/requirements.lock ./config/
RUN pip install --no-cache-dir --prefix=/install -r ./config/requirements.lock

COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install --no-cache-dir --no-deps --prefix=/install .

FROM python:3.14-slim

RUN apt-get update && apt-get upgrade -y && rm -rf /var/lib/apt/lists/*

COPY --from=builder /install /usr/local

RUN useradd --create-home eigrel
USER eigrel
WORKDIR /work

ENTRYPOINT ["eigrel"]
CMD ["--help"]
