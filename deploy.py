#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.12,<3.14"
# dependencies = ["chalkcompute==2.13.4"]
# ///
"""Deploy transaction enrichment (server.py + index.html) as a GPU ScalingGroup.

One replica holds both models on one GPU: ModernBERT-GLiNER for memo cleanup and Clef-Flash
(9B, about 19 GB in bf16) for decisions, so it needs a 48 GB card such as an L40S.

Credentials come from your existing `chalk login` (or the CHALK_* env vars):
    ./deploy.py [--env-id <environment>] [--gpu nvidia-l40s]
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
HF_HOME = "/models/hf"
GLINER = ("knowledgator/modern-gliner-bi-large-v1.0", "244e68379b54770302fb976059f7782169c9fd54")
# GLiNER's bi-encoder also loads its label encoder's and ModernBERT's config and tokenizer by
# name. Their weights are already inside the GLiNER checkpoint, so only the small files are needed.
GLINER_DEPS = ["BAAI/bge-base-en-v1.5", "answerdotai/ModernBERT-large"]
CLEF = ("Cloudflare/clef-flash", "8b2e5fd17c09fd49bc2880b805d5323ec7ff4ff3")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="transaction-enrichment")
    parser.add_argument("--env-id", default=None)
    parser.add_argument("--gpu", default="nvidia-l40s")
    parser.add_argument("--ready-timeout", type=float, default=2400.0)
    args = parser.parse_args()

    if args.env_id:
        os.environ["CHALK_ENVIRONMENT"] = args.env_id
    from chalkcompute import Image, ReadinessProbe, ScalingGroup

    gliner_fetch = "; ".join(
        [
            f"s('{GLINER[0]}', revision='{GLINER[1]}')",
            *(f"s('{m}', allow_patterns=['*.json', '*.txt'])" for m in GLINER_DEPS),
        ]
    )
    image = (
        Image.base("pytorch/pytorch:2.11.0-cuda12.8-cudnn9-runtime")
        # This image's Python is the system interpreter, which refuses pip installs (PEP 668)
        # unless told the container is the environment to manage.
        .env({"PIP_BREAK_SYSTEM_PACKAGES": "1", "UV_BREAK_SYSTEM_PACKAGES": "1"})
        .pip_install(
            [
                "transformers==5.10.2",
                "gliner==0.2.29",
                "accelerate==1.15.0",
                "fastapi==0.116.1",
                "uvicorn[standard]==0.35.0",
            ]
        )
        .env({"HF_HOME": HF_HOME, "HF_XET_HIGH_PERFORMANCE": "1"})
        # Clef's video processor imports torchvision; fail the build here rather than at boot.
        .run_commands("python -c 'import torchvision'")
        # The small GLiNER models go in the image; Clef-Flash's 19 GB download at startup.
        .run_commands(
            f'HF_HOME={HF_HOME} python -c "from huggingface_hub import snapshot_download as s; ' + gliner_fetch + '"'
        )
        # The app files together pass the 32 KiB inline-copy limit, so they ride on a volume;
        # every file under /app must use the same strategy, or the volume hides the copies.
        .add_local_file(str(HERE / "transactions.py"), "/app/transactions.py", strategy="volume")
        .add_local_file(str(HERE / "server.py"), "/app/server.py", strategy="volume")
        .add_local_file(str(HERE / "index.html"), "/app/index.html", strategy="volume")
        .workdir("/app")
    )

    sg = ScalingGroup(
        image=image,
        name=args.name,
        gpu=args.gpu,
        cpu="2",
        memory="24Gi",
        env={"CLEF_MODEL": CLEF[0], "CLEF_REVISION": CLEF[1]},
        entrypoint=[
            "bash",
            "-c",
            f"python -c \"from huggingface_hub import snapshot_download; snapshot_download('{CLEF[0]}', revision='{CLEF[1]}')\""
            + " && exec python /app/server.py",
        ],
        port=8080,
        readiness_probe=ReadinessProbe.http("/status"),
        min_replicas=1,
        max_replicas=1,
    )
    t0 = time.monotonic()
    try:
        sg.deploy(build_timeout=2400.0, ready_timeout=args.ready_timeout)
    finally:
        print(f"deploy returned after {time.monotonic() - t0:.0f}s")
    print("web_url:", sg.web_url)


if __name__ == "__main__":
    main()
