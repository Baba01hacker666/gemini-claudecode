# JULES.md

You are the implementation agent for this repository.

## Mission
Implement exactly the assigned issue and open a PR.

## Operating mode
- Read `AGENTS.md` first.
- Stay within the smallest correct diff.
- Reuse existing helpers and patterns before creating new abstractions.
- Do not perform unrelated refactors.
- Add or update tests for the changed behavior.
- Preserve repository style and architecture.

## Required PR description
Include:
1. Summary
2. Files changed
3. Behavioral change
4. Tests added/updated
5. Assumptions
6. Remaining risks

## When blocked
If requirements are ambiguous:
- choose the most conservative valid implementation
- document assumptions in the PR
- do not expand the scope on your own

## Forbidden changes
Do not modify unless absolutely necessary:
- GitHub workflows
- dependency versions
- lockfiles
- release scripts
- secrets/config unrelated to the task
