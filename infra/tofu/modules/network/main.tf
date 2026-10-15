terraform {
  required_providers {
    arvan = { source = "arvancloud/arvan" }
  }
}

variable "region" { type = string }
variable "name" { type = string }
variable "admin_cidrs" {
  description = "Source IPs allowed to reach SSH and the Kubernetes API"
  type        = list(string)
}

resource "arvan_iaas_security_group" "this" {
  region      = var.region
  name        = "${var.name}-sg"
  description = "DemandOps k3s node"
}

locals {
  rules = {
    ssh   = { port = "22", ips = var.admin_cidrs, description = "SSH (admins only)" }
    kube  = { port = "6443", ips = var.admin_cidrs, description = "Kubernetes API (admins only)" }
    http  = { port = "80", ips = ["0.0.0.0/0"], description = "HTTP / ACME challenge" }
    https = { port = "443", ips = ["0.0.0.0/0"], description = "HTTPS ingress" }
  }
}

resource "arvan_iaas_security_group_rule" "ingress" {
  for_each            = local.rules
  region              = var.region
  security_group_uuid = arvan_iaas_security_group.this.id
  description         = each.value.description
  direction           = "ingress"
  protocol            = "tcp"
  port_from           = each.value.port
  port_to             = each.value.port
  ips                 = each.value.ips
}

output "security_group_name" { value = arvan_iaas_security_group.this.name }
