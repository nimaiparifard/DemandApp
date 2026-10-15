"""Model CD: open a Pull Request that changes ONLY the model version in the GitOps values file.

Merging that PR makes Argo CD sync and Argo Rollouts run the 10% -> 50% -> 100% canary.
Works with GitHub (or Gitea, which has a compatible API: set GITOPS_API_URL).
"""

from __future__ import annotations

import base64
import re

import httpx

VERSION_RE = re.compile(r"^(?P<prefix>\s*version:\s*)\"?(?P<v>[^\s\"#]+)\"?(?P<rest>.*)$", re.M)


def bump_model_version(values_yaml: str, new_version: str) -> str:
    """Replace `model:\\n  version: "N"` in a values.yaml, leaving every other line untouched."""
    lines = values_yaml.splitlines(keepends=True)
    in_model = False
    for i, line in enumerate(lines):
        if re.match(r"^model:\s*$", line):
            in_model = True
            continue
        if in_model and re.match(r"^\S", line):  # left the `model:` block
            break
        if in_model:
            m = VERSION_RE.match(line.rstrip("\n"))
            if m:
                lines[i] = f'{m["prefix"]}"{new_version}"{m["rest"]}' + (
                    "\n" if line.endswith("\n") else ""
                )
                return "".join(lines)
    raise ValueError("no `model.version` key found in values file")


def open_model_bump_pr(
    repo: str,
    token: str,
    new_version: str,
    values_path: str = "infra/helm/demandops/values.yaml",
    base: str = "main",
    api_url: str = "https://api.github.com",
    client: httpx.Client | None = None,
    body: str = "",
) -> str:
    """Returns the PR html_url."""
    http = client or httpx.Client(
        base_url=api_url, headers={"Authorization": f"Bearer {token}"}, timeout=30
    )
    branch = f"model/bump-v{new_version}"

    base_sha = (
        http.get(f"/repos/{repo}/git/ref/heads/{base}").raise_for_status().json()["object"]["sha"]
    )
    http.post(
        f"/repos/{repo}/git/refs", json={"ref": f"refs/heads/{branch}", "sha": base_sha}
    ).raise_for_status()

    f = (
        http.get(f"/repos/{repo}/contents/{values_path}", params={"ref": branch})
        .raise_for_status()
        .json()
    )
    current = base64.b64decode(f["content"]).decode()
    updated = bump_model_version(current, new_version)
    http.put(
        f"/repos/{repo}/contents/{values_path}",
        json={
            "message": f"chore(model): serve demand-forecaster v{new_version}",
            "content": base64.b64encode(updated.encode()).decode(),
            "sha": f["sha"],
            "branch": branch,
        },
    ).raise_for_status()
    pr = (
        http.post(
            f"/repos/{repo}/pulls",
            json={
                "title": f"Promote demand-forecaster v{new_version} (canary)",
                "head": branch,
                "base": base,
                "body": body
                or "Automated by the `retrain` DAG: challenger passed all promotion gates.",
            },
        )
        .raise_for_status()
        .json()
    )
    return str(pr["html_url"])
