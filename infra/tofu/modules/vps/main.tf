terraform {
  required_providers {
    arvan = { source = "arvancloud/arvan" }
  }
}

variable "region" { type = string }
variable "name" { type = string }
variable "flavor" {
  description = "Plan id, e.g. g2-8-4-0 (4 vCPU / 8 GB). List plans in the Arvan panel or API."
  type        = string
}
variable "disk_gb" {
  type    = number
  default = 80
}
variable "image_name" {
  type    = string
  default = "ubuntu/22.04"
}
variable "ssh_public_key" { type = string }
variable "security_group" { type = string }

data "arvan_iaas_image" "os" {
  region = var.region
  type   = "distributions"
  name   = var.image_name
}

resource "arvan_iaas_sshkey" "admin" {
  region     = var.region
  name       = "${var.name}-admin"
  public_key = var.ssh_public_key
}

resource "arvan_iaas_abrak" "node" {
  region          = var.region
  name            = var.name
  flavor          = var.flavor
  disk_size       = var.disk_gb
  key_name        = arvan_iaas_sshkey.admin.name
  ssh_key         = true
  security_groups = [var.security_group]

  image {
    type = "distributions"
    name = data.arvan_iaas_image.os.name
  }

  timeouts {
    create = "20m"
  }
}

output "addresses" { value = arvan_iaas_abrak.node.addresses }
output "name" { value = arvan_iaas_abrak.node.name }
