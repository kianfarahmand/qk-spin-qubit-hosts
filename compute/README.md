# Compute: free CCE batches on GitHub Actions

**Why this setup.** Standard GitHub-hosted runners are free for public repositories. Each one is a Linux machine with 4 vCPU, 16 GB RAM and 14 GB SSD. A job can run for up to 6 h, and GitHub Free runs up to 20 jobs at once (decision ADR-0005).

**Flow**
1. In the private repo, a session writes `compute/requests/<id>.yaml`. If needed it also writes the CIFs into `compute/structures/`. MP structures can be fetched without an API key from the MP OPTIMADE endpoint (`qkhosts.data.optimade`; E0: `scripts/fetch_structures_optimade.py`, S010, Q9a).
2. `python scripts/sync_public.py ../qk-spin-qubit-hosts`, then commit and push the public clone. Push **one request file per push**.
3. The push triggers `.github/workflows/cce-batch.yml` in the public repo:
   - `plan` reads the request
   - `run` executes N shard jobs in parallel (matrix)
   - `collect` merges the shards
4. Claude sandboxes **cannot download Actions artifacts or logs** (the storage host is blocked), so results must arrive as git commits.
   Diagnostics-only requests (`mode: envcheck`) also fall back to the public branch `compute-results` when no token is set.
5. Results arrive automatically in the private repo at `results/cce/<id>/` (`merged.jsonl`, `summary.json`, `request.yaml`, `provenance.json`). This needs the secret `PRIVATE_REPO_TOKEN`. Without it, results stay as the workflow artifact `cce-<id>-merged` for 7 days. A **local** Claude Code session (Kian's `gh` login) then runs `python scripts/fetch_cce_results.py <id>`, which writes the same four files, with `provenance.json` saying `source: gh-artifact` (S010, Q9b). Sandboxed or cloud sessions cannot reach the artifact store and need the token.
6. A session pulls the private repo, checks the results, and records them in `experiments/REGISTRY.md`, `data/DATA_CARD.md` and `STATUS.md`.

**Drafts.** Requests that are written but not approved for launch live in `compute/drafts/` (private only; not in `public/MANIFEST`, so a sync cannot launch them). Launch = `git mv` to `compute/requests/` with Kian's OK. `tests/test_compute_drafts.py` checks that drafts parse.

**Request format.** See the docstring of `src/qkhosts/labels/batch.py`. Request ids must be unique. Never reuse one.

**Notes**
- Keep each shard under about 5.5 h (the job timeout is 355 min). Size shards from the per-material cost measured in E0.
- Artifacts in a public repo can be downloaded by any signed-in GitHub user while they are retained (3 days for shards, 7 for merged). The results in the private repo are private.
- Secrets live in the **public** repo's settings (Actions secrets), because that is where jobs run: `MP_API_KEY` and `PRIVATE_REPO_TOKEN`.
