# syntax=docker/dockerfile:1

# Multi-stage: the builder resolves dependencies, the runtime carries only what runs.
FROM python:3.12-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:0.12 /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

# Dependencies first, in their own layer. Application code changes far more often than
# the lockfile, so this layer is reused across nearly every rebuild.
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-install-project --no-dev

COPY app ./app
COPY evals ./evals
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev

# torch arrives as a dependency of sentence-transformers. Its default Linux wheel
# bundles the entire CUDA runtime -- ~2.9 GB of nvidia/* plus ~650 MB of triton --
# which this service never touches: embedding and reranking both run on CPU (see
# app/retrieval/embeddings.py). Swapping in the CPU-only build cuts the image by
# roughly 3.5 GB.
#
# Done here rather than in pyproject.toml because uv resolves ONE lock for every
# platform, and a per-platform index source for a single package did not take effect
# in uv 0.12. The lock stays authoritative for versions; this only changes which wheel
# of the locked version is fetched.
RUN --mount=type=cache,target=/root/.cache/uv \
    TORCH_VERSION="$(/app/.venv/bin/python -c 'import torch; print(torch.__version__.split("+")[0])')" \
    && uv pip install --python /app/.venv/bin/python \
        --index-url https://download.pytorch.org/whl/cpu \
        --reinstall-package torch \
        "torch==${TORCH_VERSION}" \
    && /app/.venv/bin/python -c "import torch; assert not torch.cuda.is_available(); print('cpu torch', torch.__version__)" \
    # The CUDA wheels were pulled in as dependencies of the default torch build and are
    # dead weight once torch is the CPU variant: nvidia/* is ~2.9 GB and triton ~650 MB.
    # Removing them is what actually shrinks the image; swapping torch alone saves 0.3 GB.
    && uv pip list --python /app/.venv/bin/python --format=freeze \
        | grep -E '^(nvidia-|triton|cuda-)' | cut -d= -f1 \
        | xargs -r uv pip uninstall --python /app/.venv/bin/python \
    && /app/.venv/bin/python -c "import torch, sentence_transformers; print('still imports:', torch.__version__)"


FROM python:3.12-slim AS runtime

# libgomp1 is required by faiss-cpu; without it the import fails at startup with an
# error that does not name the missing library.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Non-root. Nothing this service does needs write access outside its own scratch space.
RUN useradd --create-home --uid 1000 researcher
WORKDIR /app

COPY --from=builder --chown=researcher:researcher /app/.venv /app/.venv
COPY --chown=researcher:researcher app ./app
COPY --chown=researcher:researcher evals ./evals
COPY --chown=researcher:researcher scripts ./scripts
COPY --chown=researcher:researcher pyproject.toml README.md ./

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    # Nothing spends money unless this is switched on deliberately.
    CRA_LIVE_MODEL=0 \
    # Sentence-transformers caches models here; without it HOME is unset for the
    # non-root user and the first embed attempts to write to /.
    HF_HOME=/home/researcher/.cache/huggingface

USER researcher

# The corpus is NOT baked into the image: it is ~270 MB of third-party data with its own
# licences, and it is reproducible from scripts/. Mount data/ at run time.
VOLUME ["/app/data"]

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health',timeout=4).status==200 else 1)"

CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
