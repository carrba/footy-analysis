# Short-lived stack: one job instance plus its IAM role and security group.
# `terraform apply` launches the job; `terraform destroy` removes everything here. The bucket (../storage) is untouched.
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
  region = var.region

  default_tags {
    tags = { project = "footy-analysis" }
  }
}

data "aws_caller_identity" "me" {}

locals {
  bucket_name = var.bucket_name != "" ? var.bucket_name : "${var.name_prefix}-${data.aws_caller_identity.me.account_id}-${var.region}"
}

# Fails with a clear error if the storage stack has not been applied yet.
data "aws_s3_bucket" "data" {
  bucket = local.bucket_name
}

data "aws_vpc" "default" {
  default = true
}

data "aws_subnets" "default" {
  filter {
    name   = "vpc-id"
    values = [data.aws_vpc.default.id]
  }
}

# NVIDIA driver + CUDA base image (no frameworks); torch is pip-installed by the job.
# Verify the name pattern still matches with:
#   aws ec2 describe-images --owners amazon --region eu-west-2 \
#     --filters "Name=name,Values=Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04)*" \
#     --query 'sort_by(Images,&CreationDate)[-1].[Name,ImageId]'
data "aws_ami" "gpu" {
  most_recent = true
  owners      = ["amazon"]

  filter {
    name   = "name"
    values = ["Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04)*"]
  }
  filter {
    name   = "architecture"
    values = ["x86_64"]
  }
}
