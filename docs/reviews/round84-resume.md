# Round 84 resume — B38 parked before review

State: unfinished, not approved for merge. Branch `fix/r84-local-publication`.
Default branch: `main`; verified origin `git@github.com:ephes/podcast-transcript.git`.
Base: `3e2dea849748db928a354793714e18cc3fe5344d`.

## Completed implementation, pending review

B38, in report order:

1. `transcript-local-source-regular-bounded-admission`
2. `transcript-completion-marker-durable-publication-order`

Both mechanisms remain present on the inspected default branch; neither was skipped.
The changes reuse source byte/time settings for descriptor-based local hash/copy
admission and sync files/directories before completion publication. Model hashing
retains its independent behavior. README and Unreleased notes are updated.
Multi-file replacement remains nontransactional, as documented.

Validation: `just check` passes Ruff/format, mypy and 433 tests on Python 3.14;
`just test-oldest-python` passes all 433 tests on Python 3.10. All ten new
regressions fail against the original default-branch code in an isolated clone,
using synthetic files and instrumented I/O, without native inference.

The required `claude-review-loop` invocation used `--model claude-opus-5-5
--effort high`. It returned `PROVIDER_ERROR`, exit 2: OAuth session expired and
could not be refreshed. No code review ran and no verdict was obtained.
Per the owner's explicit instruction, stop here and park this branch; no merge.
Local artifacts are in the workspace sibling `reviews/b38-r1/`; the reviewer copy
was removed, with no skipped, redacted or truncated inputs.

## Resume

1. Restore Claude Code authentication outside this task, then fetch origin and
   rebase onto current `origin/main`. Re-verify the slice scope against that tip.
2. Run `just check` and `just test-oldest-python` with workspace-local tool caches.
3. Review the committed change using the installed `claude-review-loop` harness
   with `--baseline-ref origin/main --record-baseline`, a fresh `--run-dir`,
   `--model claude-opus-5-5 --effort high`, and workspace-local ledger/lock dirs.
   The explicit baseline is necessary because this parked work is committed.
   Review both B38 slices together. Continue repair/re-review only while progress
   is made, at most four rounds; stop if a required finding recurs.
4. Only after a clean or suggestion-only review, fetch immediately before merging;
   if origin moved, rebase and rerun local checks. Merge locally with `--no-ff`
   into `main`, run full local checks after the merge, push only `main`, and
   verify its SHA with `git ls-remote`. Do not run or wait for GitHub Actions.
5. Continue remaining bundles in order, re-verifying every slice on current main:
   - B39: `transcript-local-model-generation-publication-fence`, then
     `transcript-mlx-owned-worker-time-budget`.
   - B40: `transcript-native-json-output-admission-budget`, then
     `transcript-remote-decoder-preallocation-boundary`.

B39 and B40 were source-inspected and confirmed unresolved, but not implemented
or tested, because the OAuth failure requires stopping. Read their full sections
and validation notes in `~/workspaces/ws-work-planning/reviews/scout-round84.md`.
No deployment, releases, tags, pull requests, host access or work-app writes.
Use `[skip ci]` on this authorized unfinished-branch parking push. Never delete
remote branches.

Workflow incident: the required reviewer could not authenticate; local validation
was unaffected. Promotion: incident — authentication repair is external; retain
the existing fail-closed review rule and explicit parked-branch exception.
