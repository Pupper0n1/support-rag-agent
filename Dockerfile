FROM public.ecr.aws/lambda/python:3.12

# Model weights are baked into the image so a cold start does not download
# them; HF_HOME points at a writable-at-build, read-only-at-run location.
ENV HF_HOME=/opt/hf \
    TRANSFORMERS_OFFLINE=1 \
    HF_HUB_OFFLINE=1 \
    TOKENIZERS_PARALLELISM=false

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir . \
    && pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

RUN python -c "\
from sentence_transformers import SentenceTransformer, CrossEncoder; \
SentenceTransformer('sentence-transformers/all-mpnet-base-v2'); \
CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"

CMD ["support_agent.lambda_handler.handler"]
