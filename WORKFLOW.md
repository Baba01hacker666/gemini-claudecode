# WORKFLOW.md

## Human role
The human is the final authority for merge decisions.

## Standard flow
1. Create or refine a GitHub issue.
2. Add the label `jules`.
3. Jules implements and opens a PR.
4. Codex reviews the PR.
5. Fixes are applied by:
   - Jules on the same task branch, or
   - Codex on a dedicated follow-up branch if explicitly requested
6. Human reviews and merges.

## Labels
- `jules` = task queued for Jules
- `ai:jules` = PR created by Jules
- `ai:codex` = PR or patch created by Codex
- `security-review` = requires adversarial review
- `needs-human` = agent blocked or uncertain
- `ready-to-merge` = checks passed and review complete

## Safety rails
- one agent writes per branch
- one agent implements, one agent reviews
- no autonomous merges
- no self-approval
- no broad unsupervised repo rewrites

## Preferred task types
Good:
- isolated bug fixes
- test generation
- feature work with clear acceptance criteria
- docs/code sync
- parser updates
- config plumbing

Bad:
- vague “improve everything”
- simultaneous branch edits by both agents
- open-ended refactors with no acceptance criteria
