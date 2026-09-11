# Work-item format

Follow repository issue templates and language conventions first. Use these sections only when the repository has no stronger format. Omit empty optional sections rather than leaving placeholders.

## Issue body

```markdown
## Outcome

State the observable result in one short paragraph.

## Context

Explain why the work matters, the current behavior, and enough repository context for a cold start.

## Scope

### In scope

- Concrete change

### Out of scope

- Explicit boundary

## Acceptance criteria

- [ ] Observable behavior or artifact
- [ ] Exact validation command and expected result
- [ ] Required documentation, migration, or rollout evidence

## Validation

- `command`: expected signal
- Manual check: expected behavior

## Dependencies

- Related: owner/repo#123
```

Make acceptance criteria independently verifiable. Avoid vague items such as “works correctly,” “tests pass,” or “handle edge cases” without naming the expected behavior or test.

## Managed blocker section

When blocked, add or update this section without rewriting unrelated body content:

```markdown
## Blocked by

- Blocker: owner/repo#123 or a precise external cause
- Since: YYYY-MM-DD
- Unblock condition: concrete event or evidence
```

When unblocked, remove resolved blockers, preserve unresolved ones, and record the resolution in a comment. Remove the entire section only when no blocker remains.

## Implementation note

Put chronological notes in comments:

```markdown
**Implementation note (YYYY-MM-DD):** Decision, discovery, or progress. Evidence: PR/commit/path/test. Remaining: concrete next step.
```

Add a note for a durable decision, non-obvious gotcha, partial handoff, validation result, or commit/PR reference. Do not add routine narration.

## Completion comment

```markdown
**Completed (YYYY-MM-DD)**

- Outcome: what changed
- Changes: PR, commit, or artifact links
- Validation: commands and observed results
- Scope changes: none, or an explicit descoping decision
- Follow-ups: issue links, proposals, or none
```

Do not mark acceptance criteria complete merely because the implementation exists. Record the evidence that proves each criterion.
