data "aws_iam_policy_document" "assume_ec2" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["ec2.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "job" {
  name               = "${var.name_prefix}-job"
  assume_role_policy = data.aws_iam_policy_document.assume_ec2.json
}

data "aws_iam_policy_document" "bucket_access" {
  statement {
    actions   = ["s3:ListBucket"]
    resources = [data.aws_s3_bucket.data.arn]
  }
  statement {
    actions   = ["s3:GetObject", "s3:PutObject"]
    resources = ["${data.aws_s3_bucket.data.arn}/*"]
  }
}

resource "aws_iam_role_policy" "bucket_access" {
  name   = "bucket-access"
  role   = aws_iam_role.job.id
  policy = data.aws_iam_policy_document.bucket_access.json
}

# Session Manager: shell access without SSH keys or open ports.
resource "aws_iam_role_policy_attachment" "ssm" {
  role       = aws_iam_role.job.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore"
}

resource "aws_iam_instance_profile" "job" {
  name = "${var.name_prefix}-job"
  role = aws_iam_role.job.name
}
