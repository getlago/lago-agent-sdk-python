# Releasing

This SDK is published to **PyPI** as [`lago-agent-sdk`](https://pypi.org/project/lago-agent-sdk/).

Releases are triggered by pushing a `v*.*.*` git tag. The publish workflow:

1. Runs the full CI gate (ruff, mypy, pytest, coverage ≥ 80%) on Python 3.10/3.11/3.12
2. Verifies the tag's version matches `pyproject.toml`
3. Builds an sdist + wheel
4. Publishes to PyPI via **OIDC trusted publishing** (no token stored in GitHub)
5. Creates a GitHub Release with auto-generated notes + the built artifacts

## One-time setup (already done — for reference)

Configure the trusted publisher on PyPI:
**Account → Publishing → Add a new pending publisher**

| Field | Value |
| --- | --- |
| PyPI Project Name | `lago-agent-sdk` |
| Owner | `getlago` |
| Repository name | `lago-agent-sdk-python` |
| Workflow name | `publish.yml` |
| Environment name | `pypi` |

Then in this repo: **Settings → Environments → New environment** named `pypi`. (No secrets needed inside it — OIDC handles auth.)

## Cutting a release

```bash
# 1. Update the version
$EDITOR pyproject.toml              # bump version, e.g. 0.1.0 -> 0.2.0
$EDITOR CHANGELOG.md                # add release notes under a new heading

# 2. Commit + push
git commit -am "Release 0.2.0"
git push

# 3. Tag and push the tag — this triggers the publish workflow
git tag v0.2.0
git push --tags
```

Within ~5 minutes the workflow lands the package on PyPI and opens a GitHub Release. Customers can then:

```bash
pip install lago-agent-sdk==0.2.0
```

## If something goes wrong mid-release

- **CI fails before build:** fix the failure, delete the tag, retag, push.
  ```bash
  git tag -d v0.2.0
  git push --delete origin v0.2.0
  # fix the issue, recommit
  git tag v0.2.0
  git push --tags
  ```
- **Build succeeds but PyPI upload fails (rate-limit, transient):** re-running the workflow from the GitHub Actions UI is safe.
- **A bad version is already on PyPI:** PyPI does not allow re-publishing the same version. Yank it from the project page and release a fresh patch version (`v0.2.1`).

## Versioning policy

Pre-1.0 we follow `0.<minor>.<patch>` where:
- `<minor>` bumps for new features or breaking changes (we're in 0.x — breakages are allowed but documented in `CHANGELOG.md`).
- `<patch>` bumps for fixes only.

Post-1.0 we follow strict [semver](https://semver.org).
