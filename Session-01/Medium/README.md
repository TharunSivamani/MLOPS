# Medium Build

```
# multi stage build
# stage 1
FROM python:3.9-slim-buster as build

RUN apt-get update -y && apt install -y --no-install-recommends git\
    && pip install --no-cache-dir -U pip

COPY requirements.txt .

RUN pip install --user --no-cache-dir -r requirements.txt

# stage 2
FROM python:3.9-slim-buster 

COPY --from=build /root/.local /root/.local

ENV PATH=/root/.local/bin:$PATH

WORKDIR /app

COPY . .

ENTRYPOINT [ "python3", "inference.py" ]
```