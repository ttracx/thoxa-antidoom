# THOX GitHub Agent Team

This repository is wired for THOX automated issue review, pull request review, safe merge handling, and merged-branch pruning.

## Agent teams

| Team | Trigger | Responsibility | Merge authority |
|---|---:|---|---:|
| Issue Intake Team | `issues`, scheduled review | Classify issues, add routing labels, flag security/manual-review items | No |
| PR Review Team | `pull_request`, scheduled review | Review pull request readiness, check labels, check draft status, check status gates | No |
| Merge Steward Team | `agent:auto-merge` label | Squash merge only when configured gates pass | Yes |
| Branch Janitor Team | Post-merge | Delete merged same-repository branches when safe | Yes |

## Safe merge policy

Auto-merge is only attempted when all conditions are true:

1. Pull request has the `agent:auto-merge` label.
2. Pull request is not a draft.
3. Pull request does not have any blocking label:
   - `agent:blocked`
   - `human-required`
   - `do-not-merge`
   - `security:manual-review`
4. Pull request source branch is in the same repository, not from a fork.
5. Pull request source branch is not the default branch.
6. GitHub status/check gates are passing or neutral/skipped.
7. Merge method is squash merge.

## Branch pruning policy

After a successful THOX auto-merge, the workflow attempts to delete only the merged head branch. It never intentionally deletes the default branch or fork-owned branches. If GitHub branch protection or permissions block deletion, the workflow leaves a PR comment instead of forcing deletion.

## Issue labels

The workflow ensures these labels exist:

- `agent:resolve`
- `agent:review`
- `agent:auto-merge`
- `agent:blocked`
- `human-required`
- `do-not-merge`
- `security:manual-review`
- `type:bug`
- `type:dependency`
- `type:security`
- `area:docs`

## Daily review

The scheduled run reviews open pull requests, checks safety gates, comments status, and merges only explicitly labeled eligible PRs.

## Repository bootstrap status

- Workflow installed: `.github/workflows/thox-agent-team.yml`
- Documentation installed: `docs/thox-agent-team.md`
- Default branch: `main`
- Branch pruning documented: yes
- Safe merge gates documented: yes
