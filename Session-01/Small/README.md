# Small build

```
# multi stage build
# stage 1
FROM python:3.9-slim-buster as build

RUN python3 -m venv /opt/venv && \
    /opt/venv/bin/pip install --disable-pip-version-check --no-cache-dir https://download.pytorch.org/whl/cpu/torch-1.11.0%2Bcpu-cp39-cp39-linux_x86_64.whl \
    https://download.pytorch.org/whl/cpu/torchvision-0.12.0%2Bcpu-cp39-cp39-linux_x86_64.whl \
    https://files.pythonhosted.org/packages/72/ed/358a8bc5685c31c0fe7765351b202cf6a8c087893b5d2d64f63c950f8beb/timm-0.6.7-py3-none-any.whl

RUN rm -rf /opt/venv/bin/pip* && rm -rf /opt/venv/bin/*-info

# stage 2
FROM gcr.io/distroless/python3-debian11

COPY . /usr/app

WORKDIR /usr/app

ENV PYTHONPATH /usr/app

COPY --from=build /opt/venv/lib/python3.9/site-packages /usr/app

ENTRYPOINT [ "python3", "/usr/app/inference.py" ]
```
