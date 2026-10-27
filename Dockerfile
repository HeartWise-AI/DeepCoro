FROM python:3.11-slim
WORKDIR /opt/deepcoro

RUN apt-get update \
    && apt-get install --no-install-recommends -y ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md LICENSE.txt main.py utils.py ./
COPY deepcoro ./deepcoro
RUN pip install --no-cache-dir .

ENV HF_HOME=/cache/huggingface
VOLUME ["/dcm_input", "/results", "/cache/huggingface"]
ENTRYPOINT ["deepcoro"]
CMD ["--input-file-path", "/dcm_input/input_file.csv", "--save-dir", "/results"]
