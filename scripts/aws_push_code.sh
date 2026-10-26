#!/bin/bash
# Package the source and upload it to the project bucket, where the job instance picks it up.
# Usage: scripts/aws_push_code.sh   (run from the repo root, after `terraform apply` has created the bucket)
set -euo pipefail

BUCKET="$(terraform -chdir=infra/aws/storage output -raw bucket)"
TARBALL="$(mktemp --suffix=.tgz)"

tar czf "$TARBALL" src config.yaml pyproject.toml
aws s3 cp "$TARBALL" "s3://$BUCKET/code/footy-code.tgz"
rm -f "$TARBALL"
echo "uploaded code to s3://$BUCKET/code/footy-code.tgz"
