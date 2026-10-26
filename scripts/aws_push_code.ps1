# Package the source and upload it to the project bucket, where the job instance picks it up.
# Usage: scripts\aws_push_code.ps1   (run from the repo root, after `terraform apply` has created the bucket)
$ErrorActionPreference = 'Stop'

if (-not $env:AWS_PROFILE) {
    $env:AWS_PROFILE = Read-Host 'AWS profile name'
}

$bucket = (terraform -chdir=infra/aws/storage output -raw bucket)
if ($LASTEXITCODE -ne 0) { throw 'could not read the bucket name from the storage stack' }

$tarball = Join-Path ([System.IO.Path]::GetTempPath()) 'footy-code.tgz'

try {
    tar czf $tarball src config.yaml pyproject.toml
    if ($LASTEXITCODE -ne 0) { throw 'tar failed' }

    aws s3 cp $tarball "s3://$bucket/code/footy-code.tgz"
    if ($LASTEXITCODE -ne 0) { throw 'upload failed' }
}
finally {
    Remove-Item -Force -ErrorAction SilentlyContinue $tarball
}
Write-Host "uploaded code to s3://$bucket/code/footy-code.tgz"
