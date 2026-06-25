"""
Refreshes _data/tools.yaml from live GitHub and HuggingFace data.

What it does:
  - For every existing entry with a `repo:` field (any GitHub owner/repo), refetch
    its description from the GitHub API. Removes the entry if the repo is gone.
  - For every existing entry in the `huggingface` group, checks that the HF
    dataset/space still exists. Removes the entry if it is gone.
  - Lists all repos in the THUIR GitHub org (minus EXCLUDED_GITHUB_REPOS) and
    appends any repo not yet referenced by an existing entry as a new `code`
    entry.
  - Lists all datasets/spaces owned by the THUIR HuggingFace org (minus
    EXCLUDED_HF_ITEMS) and appends any item not yet referenced as a new
    `huggingface` entry.

Never touches `group`, `title`, `image`, `tags`, or curated fields of existing
entries -- only descriptions get refreshed, dead entries get removed, and new
entries get appended. Re-categorizing a newly-added entry (e.g. moving it from
`code` to `toolkit`/`dataset`) is left to manual editing.
"""

import os
import re
import sys
from pathlib import Path

import requests
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
TOOLS_YAML = REPO_ROOT / "_data" / "tools.yaml"

GITHUB_ORG = "THUIR"
HF_ORG = "THUIR"

# Repos that are site infrastructure, forks of unrelated tools, or not
# research artifacts -- never synced onto the opensource page.
EXCLUDED_GITHUB_REPOS = {
    "THUIR.github.io",
    "THUIR-website",
    "test",
    ".github",
    "skill-grep",
    "goassia.github.io",
    "TianchiCompetition",
}

# HF items that are org infrastructure (e.g. the org profile "README" space).
EXCLUDED_HF_ITEMS = {
    ("space", "THUIR/README"),
}

GITHUB_API = "https://api.github.com"
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")
SESSION = requests.Session()
if GITHUB_TOKEN:
    SESSION.headers["Authorization"] = f"Bearer {GITHUB_TOKEN}"
SESSION.headers["Accept"] = "application/vnd.github+json"


def github_get(path, **params):
    r = SESSION.get(f"{GITHUB_API}{path}", params=params, timeout=30)
    if r.status_code == 404:
        return None
    r.raise_for_status()
    return r.json()


def list_org_repos(org):
    repos, page = [], 1
    while True:
        batch = github_get(f"/orgs/{org}/repos", per_page=100, page=page, type="public")
        if not batch:
            break
        repos.extend(batch)
        if len(batch) < 100:
            break
        page += 1
    return repos


def readme_first_paragraph(get_readme_text):
    """Given a callable returning raw README markdown (or None), extract a
    plausible one-paragraph description, skipping badges/images/headers."""
    text = get_readme_text()
    if not text:
        return ""
    # strip YAML frontmatter
    text = re.sub(r"^---\n.*?\n---\n", "", text, flags=re.S)
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(("#", "[![", "![", "<img", "<p", "<div", "<h", "[!")):
            continue
        if set(line) <= set("-=*_ "):
            continue
        return re.sub(r"\s+", " ", line)[:500]
    return ""


def github_readme_text(owner_repo):
    r = SESSION.get(
        f"{GITHUB_API}/repos/{owner_repo}/readme",
        headers={**SESSION.headers, "Accept": "application/vnd.github.raw+json"},
        timeout=30,
    )
    if r.status_code != 200:
        return None
    return r.text


def hf_readme_text(kind, hf_id):
    prefix = "" if kind == "dataset" else "spaces/"
    url = f"https://huggingface.co/{prefix}{hf_id}/raw/main/README.md"
    r = requests.get(url, timeout=30)
    if r.status_code != 200:
        return None
    return r.text


def parse_github_link(link):
    m = re.match(r"https?://github\.com/([^/\s]+)/([^/\s]+)/?$", link or "")
    return f"{m.group(1)}/{m.group(2)}" if m else None


def parse_hf_link(link):
    m = re.match(r"https?://huggingface\.co/(datasets|spaces)/([^/\s]+)/([^/\s]+)/?$", link or "")
    if not m:
        return None
    kind = "dataset" if m.group(1) == "datasets" else "space"
    return kind, f"{m.group(2)}/{m.group(3)}"


def refresh_existing_entries(entries):
    # Only fill in a *missing* description from the live API -- never replace
    # a hand-curated one. GitHub's one-line repo tagline is usually a worse
    # description than what a maintainer wrote by hand.
    kept = []
    for e in entries:
        repo = e.get("repo")
        if repo and "/" in repo:
            info = github_get(f"/repos/{repo}")
            if info is None:
                print(f"  [drop] {e.get('title')} -- github.com/{repo} no longer exists")
                continue
            if not e.get("description") and info.get("description"):
                e["description"] = info["description"]
            kept.append(e)
            continue

        if e.get("group") == "huggingface":
            parsed = parse_hf_link(e.get("link", ""))
            if parsed:
                kind, hf_id = parsed
                url = (
                    f"https://huggingface.co/api/datasets/{hf_id}"
                    if kind == "dataset"
                    else f"https://huggingface.co/api/spaces/{hf_id}"
                )
                r = requests.get(url, timeout=30)
                if r.status_code == 404:
                    print(f"  [drop] {e.get('title')} -- huggingface.co/{hf_id} no longer exists")
                    continue
            kept.append(e)
            continue

        kept.append(e)
    return kept


def add_new_github_repos(entries):
    referenced = {e["repo"] for e in entries if e.get("repo") and "/" in e.get("repo", "")}
    referenced_names = {r.split("/")[1] for r in referenced if r.split("/")[0] == GITHUB_ORG}

    added = []
    for repo in list_org_repos(GITHUB_ORG):
        name = repo["name"]
        if name in EXCLUDED_GITHUB_REPOS or name in referenced_names:
            continue
        owner_repo = f"{GITHUB_ORG}/{name}"
        description = repo.get("description") or readme_first_paragraph(
            lambda owner_repo=owner_repo: github_readme_text(owner_repo)
        )
        entry = {
            "title": name,
            "group": "code",
            "link": repo["html_url"],
            "description": description,
            "repo": owner_repo,
            "tags": repo.get("topics") or [],
        }
        print(f"  [add] {owner_repo} -> group: code")
        added.append(entry)
    return entries + added


def add_new_hf_items(entries):
    referenced = set()
    for e in entries:
        parsed = parse_hf_link(e.get("link", ""))
        if parsed:
            referenced.add(parsed)

    added = []
    for kind, api_path in (("dataset", "datasets"), ("space", "spaces")):
        r = requests.get(
            "https://huggingface.co/api/" + api_path,
            params={"author": HF_ORG, "limit": 200},
            timeout=30,
        )
        r.raise_for_status()
        for item in r.json():
            hf_id = item["id"]
            if (kind, hf_id) in EXCLUDED_HF_ITEMS or (kind, hf_id) in referenced:
                continue
            name = hf_id.split("/")[-1]
            link_kind = "datasets" if kind == "dataset" else "spaces"
            description = readme_first_paragraph(lambda kind=kind, hf_id=hf_id: hf_readme_text(kind, hf_id))
            entry = {
                "title": name,
                "subtitle": f"huggingface.co/{link_kind}/{hf_id}",
                "group": "huggingface",
                "link": f"https://huggingface.co/{link_kind}/{hf_id}",
                "description": description,
                "tags": [],
            }
            print(f"  [add] huggingface.co/{link_kind}/{hf_id} -> group: huggingface")
            added.append(entry)
    return entries + added


GROUP_ORDER = ["toolkit", "dataset", "code", "huggingface"]
GROUP_TITLES = {
    "toolkit": "Toolkits",
    "dataset": "Datasets",
    "code": "Code / Paper Implementations",
    "huggingface": "HuggingFace Projects",
}
FIELD_ORDER = [
    "title", "subtitle", "group", "image", "link", "description", "repo", "tags",
]


class _IndentDumper(yaml.Dumper):
    """Indents block-sequence items under their parent key, matching the
    hand-written style already used throughout tools.yaml (`tags:\n    - x`
    instead of PyYAML's default `tags:\n  - x`)."""

    def increase_indent(self, flow=False, indentless=False):
        return super().increase_indent(flow, False)


def dump_entry(entry):
    ordered = {k: entry[k] for k in FIELD_ORDER if k in entry}
    for k in entry:
        if k not in ordered:
            ordered[k] = entry[k]
    text = yaml.dump(
        [ordered],
        Dumper=_IndentDumper,
        default_flow_style=False,
        sort_keys=False,
        allow_unicode=True,
        width=10**9,
    )
    return text


def write_tools_yaml(entries):
    by_group = {g: [] for g in GROUP_ORDER}
    for e in entries:
        by_group.setdefault(e.get("group", "code"), []).append(e)

    parts = []
    for group in GROUP_ORDER:
        items = by_group.get(group, [])
        if not items:
            continue
        parts.append("# " + "=" * 60)
        parts.append(f"# {GROUP_TITLES.get(group, group.title())}")
        parts.append("# " + "=" * 60)
        parts.append("")
        for entry in items:
            parts.append(dump_entry(entry).rstrip())
            parts.append("")
    TOOLS_YAML.write_text("\n".join(parts).rstrip() + "\n", encoding="utf-8")


def main():
    entries = yaml.safe_load(TOOLS_YAML.read_text(encoding="utf-8")) or []
    print(f"Loaded {len(entries)} existing entries")

    print("Refreshing existing entries...")
    entries = refresh_existing_entries(entries)

    print(f"Checking {GITHUB_ORG} GitHub org for new repos...")
    entries = add_new_github_repos(entries)

    print(f"Checking {HF_ORG} HuggingFace org for new datasets/spaces...")
    entries = add_new_hf_items(entries)

    write_tools_yaml(entries)
    print(f"Wrote {len(entries)} entries to {TOOLS_YAML}")


if __name__ == "__main__":
    sys.exit(main())
