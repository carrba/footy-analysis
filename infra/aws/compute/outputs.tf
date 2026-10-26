output "instance_id" {
  value = aws_instance.job.id
}

output "results_prefix" {
  value = "s3://${data.aws_s3_bucket.data.bucket}/results/${var.job_name}/"
}
