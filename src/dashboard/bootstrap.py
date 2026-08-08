"""Optional runtime sync of real data from a private companion repo.

This dashboard's public code repository never contains the real thesis
outputs — ``data/real/`` is gitignored (see ``docs/DATA.md``), and a clean
``git clone`` only ever has the synthetic ``data/demo/`` sample.

For a deployment that should show real results (e.g. Streamlit Community
Cloud), this module pulls ``data/real/`` down at process startup from a
*separate, private, data-only* GitHub repository, authenticated with a
fine-grained, read-only personal access token supplied via Streamlit
secrets. The token and repo name never appear in this codebase — only in
the deployment's own secrets store.

Secrets format (``.streamlit/secrets.toml`` locally, or the "Secrets" box
in the Streamlit Community Cloud app settings)::

    [real_data_source]
    repo  = "your-github-username/your-private-data-repo"
    token = "github_pat_..."   # fine-grained PAT, Contents: Read-only, scoped to that one repo
    ref   = "main"             # optional, defaults to "main"

If no secrets are configured (e.g. a plain local `git clone`, or a
deployment you deliberately want to keep in demo mode), this is a no-op and
``dashboard.config`` falls back to the bundled synthetic sample exactly as
it does today.
"""

from __future__ import annotations

import io
import tarfile
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
REAL_DATA_DIR = REPO_ROOT / "data" / "real"


def sync_real_data() -> None:
    """Populate ``data/real/`` from a private GitHub repo, if configured.

    Safe to call unconditionally at app startup, before ``dashboard.config``
    is imported: it is a fast no-op whenever secrets aren't set or the data
    has already been synced in this process.
    """
    if REAL_DATA_DIR.exists() and any(REAL_DATA_DIR.iterdir()):
        return

    try:
        import streamlit as st

        source = st.secrets.get("real_data_source", {})
    except Exception:
        return

    repo = source.get("repo")
    token = source.get("token")
    ref = source.get("ref", "main")
    if not repo or not token:
        return

    try:
        _download_and_extract(repo=repo, token=token, ref=ref)
    except Exception as exc:  # noqa: BLE001 - never crash the app over this
        import streamlit as st

        st.warning(
            f"Could not sync real data from the private data source ({exc}). "
            "Falling back to the bundled synthetic demo dataset.",
            icon="⚠️",
        )


def _download_and_extract(*, repo: str, token: str, ref: str) -> None:
    url = f"https://api.github.com/repos/{repo}/tarball/{ref}"
    request = urllib.request.Request(
        url,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "User-Agent": "german-gdp-nowcast-dashboard",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            archive_bytes = response.read()
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"GitHub API returned {exc.code} for {repo}@{ref}") from exc

    REAL_DATA_DIR.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(archive_bytes), mode="r:gz") as tar:
        members = tar.getmembers()
        if not members:
            return
        # GitHub tarballs wrap everything in one "<owner>-<repo>-<sha>/" folder.
        prefix = members[0].name.split("/", 1)[0] + "/"
        extracted = []
        for member in members:
            if not member.name.startswith(prefix):
                continue
            member.name = member.name[len(prefix):]
            if member.name:
                extracted.append(member)
        tar.extractall(REAL_DATA_DIR, members=extracted)  # noqa: S202 - trusted, token-gated source
