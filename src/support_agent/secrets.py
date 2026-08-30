"""Resolve secrets from AWS Secrets Manager into the environment at cold start.

Settings read plain env vars. In Lambda the values live in Secrets Manager
and only the ARNs are in the environment, so this shim runs before settings
are parsed. Locally the *_SECRET_ARN vars are unset and it is a no-op.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

SECRET_ENV_MAP: dict[str, str] = {
    "ES_API_KEY_SECRET_ARN": "ES_API_KEY",
    "ANTHROPIC_API_KEY_SECRET_ARN": "ANTHROPIC_API_KEY",
}


def hydrate_secrets(region: str) -> list[str]:
    pending = {arn_var: target for arn_var, target in SECRET_ENV_MAP.items() if os.environ.get(arn_var)}
    if not pending:
        return []

    import boto3

    client = boto3.client("secretsmanager", region_name=region)
    hydrated: list[str] = []
    for arn_var, target in pending.items():
        if os.environ.get(target):
            continue
        value = client.get_secret_value(SecretId=os.environ[arn_var])["SecretString"]
        os.environ[target] = value
        hydrated.append(target)
    logger.info("hydrated secrets: %s", ", ".join(hydrated) or "none")
    return hydrated
