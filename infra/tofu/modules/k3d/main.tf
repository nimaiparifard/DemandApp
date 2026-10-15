# Local cluster for daily work (free): k3d = k3s in Docker, plus a local registry so images
# built on the laptop need no internet or Harbor.
variable "name" {
  type    = string
  default = "demandops"
}
variable "agents" {
  type    = number
  default = 1
}
variable "registry_port" {
  type    = number
  default = 5111
}
variable "registries_yaml" {
  description = "Path to infra/mirrors/registries.yaml (Docker Hub mirrors); empty = none"
  type        = string
  default     = ""
}

resource "terraform_data" "cluster" {
  input = { name = var.name }

  provisioner "local-exec" {
    command = <<-EOT
      k3d registry create ${var.name}-registry.localhost --port ${var.registry_port} || true
      k3d cluster create ${var.name} --agents ${var.agents} \
        --registry-use k3d-${var.name}-registry.localhost:${var.registry_port} \
        ${var.registries_yaml != "" ? "--registry-config ${var.registries_yaml}" : ""} \
        -p "80:80@loadbalancer" -p "443:443@loadbalancer" \
        --k3s-arg "--disable=traefik@server:*" --wait
    EOT
  }

  provisioner "local-exec" {
    when    = destroy
    command = "k3d cluster delete ${self.input.name} ; k3d registry delete k3d-${self.input.name}-registry.localhost || true"
  }
}

output "kube_context" { value = "k3d-${var.name}" }
output "registry" { value = "k3d-${var.name}-registry.localhost:${var.registry_port}" }
