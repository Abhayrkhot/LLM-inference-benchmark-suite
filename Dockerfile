# Client image: run the load generator next to (not inside) the inference server.
#   docker build -t llmbench .
#   docker run --rm --network host -v $PWD/results:/results llmbench run \
#     --base-url http://127.0.0.1:8000 --model Qwen/Qwen2.5-7B-Instruct --tokenizer Qwen/Qwen2.5-7B-Instruct \
#     --dataset random --rate 8 -n 500 --out /results
FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install ".[hf,fast]" && useradd --create-home bench
USER bench
ENTRYPOINT ["llmbench"]
CMD ["--help"]
