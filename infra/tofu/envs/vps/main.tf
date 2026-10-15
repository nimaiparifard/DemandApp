terraform {
  required_version = ">= 1.6"
  required_providers {
    arvan = { source = "arvancloud/arvan" }
    local = { source = "hashicorp/local" }
  }
}

variable "arvan_api_key" {
  type      = string
  sensitive = true
}
variable "region" {
  type    = string
  default = "ir-thr-ba1"
}
variable "name" {
  type    = string
  default = "demandops"
}
variable "flavor" {
  type    = string
  default = "g2-8-4-0"
}
variable "ssh_public_key" { type = string }
variable "admin_cidrs" { type = list(string) }
variable "domain" { type = string }

provider "arvan" {
  api_key = var.arvan_api_key
}

module "network" {
  source      = "../../modules/network"
  region      = var.region
  name        = var.name
  admin_cidrs = var.admin_cidrs
}

module "vps" {
  source         = "../../modules/vps"
  region         = var.region
  name           = var.name
  flavor         = var.flavor
  ssh_public_key = var.ssh_public_key
  security_group = module.network.security_group_name
}

locals {
  ip = try(module.vps.addresses[0], "0.0.0.0")
}

module "dns" {
  source = "../../modules/dns"
  ip     = local.ip
  domain = var.domain
}

# Hand-off to Ansible: the inventory is generated, never hand-edited.
resource "local_file" "inventory" {
  filename        = "${path.root}/../../../ansible/inventory.ini"
  file_permission = "0640"
  content         = <<-EOT
    [k3s_server]
    ${var.name} ansible_host=${local.ip} ansible_user=ubuntu
  EOT
}

output "ip" { value = local.ip }
output "hosts_file" { value = module.dns.hosts_file }
