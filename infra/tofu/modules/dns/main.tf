# Provider-agnostic: renders the host names the platform exposes. Feed `hosts_file` to /etc/hosts
# for a lab, or create the same records in your DNS provider (add that resource here without
# touching the other modules).
variable "ip" { type = string }
variable "domain" { type = string }
variable "subdomains" {
  type    = list(string)
  default = ["forecast", "app", "grafana", "harbor", "argocd", "airflow", "mlflow"]
}

locals {
  fqdns = [for s in var.subdomains : "${s}.${var.domain}"]
}

output "fqdns" { value = local.fqdns }
output "hosts_file" { value = "${var.ip} ${join(" ", local.fqdns)}" }
