import os

import torch._dynamo

torch._dynamo.config.suppress_errors = True

# vLLM's env override runs os.environ.setdefault("TRITON_CACHE_AUTOTUNING", "1") when it is
# imported, which happens during collection wherever vLLM is installed (ModelOpt loads its
# plugin). The determinism suite then rejects the run because no TRITON_CACHE_DIR is set.
# Pin Triton's own default first; this package is imported before the root conftest.
os.environ.setdefault("TRITON_CACHE_AUTOTUNING", "0")
