# Bootstraps Argo CD; from here on EVERYTHING else is installed by Argo CD from git (app-of-apps).
terraform {
  required_providers {
    helm = { source = "hashicorp/helm" }
  }
}

variable "chart_version" {
  type    = string
  default = "10.10.5"
}
variable "chart_repository" {
  description = "Helm repo URL (or your own mirror)"
  type        = string
  default     = "https://argoproj.github.io/argo-helm"
}
variable "root_app_manifest" {
  description = "Path to infra/argocd/root-app.yaml"
  type        = string
}
variable "kube_context" { type = string }

resource "helm_release" "argocd" {
  name             = "argocd"
  namespace        = "argocd"
  create_namespace = true
  repository       = var.chart_repository
  chart            = "argo-cd"
  version          = var.chart_version
  wait             = true
  timeout          = 600
}

resource "terraform_data" "root_app" {
  depends_on       = [helm_release.argocd]
  triggers_replace = [filesha256(var.root_app_manifest)]

  provisioner "local-exec" {
    command = "kubectl --context ${var.kube_context} apply -f ${var.root_app_manifest}"
  }
}
