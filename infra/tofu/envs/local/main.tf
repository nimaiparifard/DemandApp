terraform {
  required_version = ">= 1.6"
  required_providers {
    helm = { source = "hashicorp/helm" }
  }
}

module "cluster" {
  source          = "../../modules/k3d"
  registries_yaml = abspath("${path.root}/../../../mirrors/registries.yaml")
}

provider "helm" {
  kubernetes = {
    config_path    = "~/.kube/config"
    config_context = module.cluster.kube_context
  }
}

module "argocd" {
  source            = "../../modules/argocd"
  kube_context      = module.cluster.kube_context
  root_app_manifest = abspath("${path.root}/../../../argocd/root-app.yaml")
  depends_on        = [module.cluster]
}

output "registry" { value = module.cluster.registry }
output "next_steps" {
  value = "kubectl -n argocd get applications  # push images to ${module.cluster.registry}"
}
