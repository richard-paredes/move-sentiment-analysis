FROM python:3.10-alpine

WORKDIR /app

RUN apk add --no-cache gcc musl-dev linux-headers

COPY pyproject.toml pyproject.toml
COPY poetry.lock poetry.lock

RUN pip install poetry

RUN poetry install

EXPOSE 8000

COPY . .

CMD ["poetry", "run", "fastapi", "dev", "src/app.py"]