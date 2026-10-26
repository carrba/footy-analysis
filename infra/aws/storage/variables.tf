variable "region" {
  description = "AWS region. Must match the compute stack."
  type        = string
  default     = "eu-west-2"
}

variable "name_prefix" {
  type    = string
  default = "footy"
}
