# AGENTS.md

This repository uses a multi-agent workflow.

## Agent roles

### Jules
Primary implementation agent.
Responsibilities:
- implement scoped issues
- keep diffs minimal
- run or update tests
- open PRs with assumptions and risks

### Codex
Secondary review/hardening agent.
Responsibilities:
- review PRs for correctness
- find security and logic bugs
- identify missing tests
- propose minimal follow-up fixes

## Global rules
- Never rewrite unrelated parts of the codebase.
- Always prefer the smallest correct diff.
- Do not force-push over another agent's branch.
- Do not change CI, secrets, workflows, dependencies, or lockfiles unless required.
- If the task overlaps another agent's scope, stop and report overlap.
- Every PR must include:
  1. summary
  2. files changed
  3. behavior changed
  4. tests added/updated
  5. assumptions
  6. risks

## Branch rules
- Jules works on: `feat/jules-<issue-number>`
- Codex follow-up work uses: `fix/codex-<issue-number>`
- Never have both agents commit to the same branch.

## Review rules
Codex should prioritize:
- correctness bugs
- auth/authz mistakes
- input validation
- parser edge cases
- state transition bugs
- race conditions
- path traversal / SSRF / command execution surfaces
- tests that do not actually verify the claimed behavior

## Scope discipline
Allowed:
- task-specific implementation
- focused tests
- minimal refactors required by task completion

Not allowed:
- unrelated cleanup
- mass renames
- style-only churn
- broad architectural changes unless the issue explicitly requires them
