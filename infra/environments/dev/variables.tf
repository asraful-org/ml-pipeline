variable "resource_group_name" {
  description = "Resource group name"
  type        = string
  default = ""
}

variable "storage_account_name" {
  description = "Storage account name"
  type        = string
  default = "mlpiplelinewor5882013883"
}

variable "container_name" {
  description = "Container name"
  type        = string
  default = "ml-pipeline-container"
}

variable "key" {
  description = "key value"
  type = string
  default = ""
}


variable "tanent_id" {
  description = "tanent_id"
  type = string
  default = ""
}

variable "subscription_id" {
  description = "subscription_id"
  type = string
  default = ""
}