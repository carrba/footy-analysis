# No inbound rules at all: the instance only needs outbound internet (pip, apt, S3, model weights).
resource "aws_security_group" "job" {
  name        = "${var.name_prefix}-job"
  description = "footy job instance: egress only"
  vpc_id      = data.aws_vpc.default.id

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

resource "aws_instance" "job" {
  ami                    = data.aws_ami.gpu.id
  instance_type          = var.instance_type
  subnet_id              = data.aws_subnets.default.ids[0]
  vpc_security_group_ids = [aws_security_group.job.id]
  iam_instance_profile   = aws_iam_instance_profile.job.name

  # The job ends with `shutdown -h now`; with "terminate" that deletes the instance and stops the billing.
  instance_initiated_shutdown_behavior = "terminate"

  dynamic "instance_market_options" {
    for_each = var.use_spot ? [1] : []
    content {
      market_type = "spot"
      spot_options {
        spot_instance_type             = "one-time"
        instance_interruption_behavior = "terminate"
      }
    }
  }

  metadata_options {
    http_tokens = "required" # IMDSv2 only
  }

  root_block_device {
    volume_size = var.root_volume_gb
    volume_type = "gp3"
    encrypted   = true
  }

  user_data = templatefile("${path.module}/user_data.sh.tftpl", {
    bucket              = data.aws_s3_bucket.data.bucket
    job_name            = var.job_name
    input_key           = var.input_key
    duration_seconds    = var.duration_seconds
    crf                 = var.crf
    max_runtime_minutes = var.max_runtime_minutes
  })

  tags = { Name = "${var.name_prefix}-${var.job_name}" }
}
