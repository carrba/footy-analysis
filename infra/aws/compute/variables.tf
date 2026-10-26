variable "region" {
  description = "AWS region. Must match the storage stack."
  type        = string
  default     = "eu-west-2"
}

variable "name_prefix" {
  type    = string
  default = "footy"
}

variable "bucket_name" {
  description = "Leave empty to derive the same name the storage stack uses."
  type        = string
  default     = ""
}

variable "instance_type" {
  description = "g4dn.xlarge = 1x T4, 4 vCPU. The motion step is CPU-bound; g4dn.2xlarge (8 vCPU) is the next step up."
  type        = string
  default     = "g4dn.xlarge"
}

variable "use_spot" {
  description = "Spot is much cheaper but can be interrupted (job is lost and must be rerun)."
  type        = bool
  default     = true
}

variable "job_name" {
  description = "Results are written to s3://<bucket>/results/<job_name>/"
  type        = string
  default     = "job1"
}

variable "input_key" {
  description = "S3 key of the video to process, e.g. raw/match1.mp4"
  type        = string
}

variable "duration_seconds" {
  description = "How much of the video to ingest from the start. 180 for a test clip, ~6600 for a full match."
  type        = number
  default     = 6600
}

variable "crf" {
  description = "x264 quality for the 10 fps working copy. 18 is large (about 100 MB/min); 23 is about a third of that."
  type        = number
  default     = 23
}

variable "max_runtime_minutes" {
  description = "Watchdog: the instance shuts itself down (and terminates) after this long whatever happens."
  type        = number
  default     = 360
}

variable "root_volume_gb" {
  type    = number
  default = 200
}
