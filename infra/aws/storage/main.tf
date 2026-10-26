# Long-lived stack: the bucket for match footage, code and results. Costs only the storage (pennies).
# The job instance lives in ../compute and is created and destroyed separately.
terraform {
  required_version = ">= 1.5"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region  = var.region
  profile = var.aws_profile

  default_tags {
    tags = { project = "footy-analysis" }
  }
}

data "aws_caller_identity" "me" {}

locals {
  # The compute stack derives the same name; keep the two in step (or pass bucket_name there explicitly).
  bucket_name = "${var.name_prefix}-${data.aws_caller_identity.me.account_id}-${var.region}"
}

resource "aws_s3_bucket" "data" {
  bucket = local.bucket_name
  # Never auto-delete match footage: emptying the bucket is a deliberate manual step.
  force_destroy = false
}

resource "aws_s3_bucket_public_access_block" "data" {
  bucket                  = aws_s3_bucket.data.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_server_side_encryption_configuration" "data" {
  bucket = aws_s3_bucket.data.id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

# Layout: raw/ (uploaded Veo downloads), code/ (packaged source), results/<job>/ (outputs).
resource "aws_s3_bucket_lifecycle_configuration" "data" {
  bucket = aws_s3_bucket.data.id

  rule {
    id     = "abort-incomplete-uploads"
    status = "Enabled"
    filter {}
    abort_incomplete_multipart_upload {
      days_after_initiation = 3
    }
  }
}
