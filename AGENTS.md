# Repository engineering instructions

These instructions govern repository work. The task defines the requested
outcome; the README defines the pipeline's user-facing behavior and archive
layout.

## Scope and investigation

- Keep reviews and audits read-only unless edits, a report, or an issue are
  requested. Record the baseline branch and commit for an audit.
- Implement one bounded issue per PR. Feature work and fixes are both valid;
  avoid adjacent cleanup, unrelated renames, and formatting changes. Preserve
  unrelated user changes.
- Before editing, inspect the relevant implementation, callers, tests,
  documentation, and open issues/PRs. Identify overlapping or prerequisite work.
  Documentation and tests do not replace reading the implementation.
- For changes to responsibilities, dependencies, or public behavior, explain the
  owner, affected components, and contract changes before editing. Use **Used
  by**, **Depends on**, and **Coordinates with** where useful. Routine edits do
  not need a formal architecture report.
- Keep architectural changes explicit in the issue. Where ownership is unclear,
  explain the design choice rather than hiding new behavior in a generic utility.

## Ownership and dependencies

- Keep orchestration, credentials, CDS requests, GRIB conversion, Zarr storage,
  and catalog state as clear responsibilities. Use functions and modules where
  sufficient; this list does not require a service or framework for each one.
- Orchestration coordinates the workflow. Storage and catalog publication use
  explicit contracts. Lower-level adapters must not depend on the command-line
  workflow or a particular consuming application.
- Justify new dependencies and abstractions with a concrete need. Reuse existing
  behavior when its inputs, errors, lifecycle, and performance semantics match.
  Similar names or large files alone do not justify consolidation or extraction.
- During initial development, use the current schema and file formats. Do not
  add schema versions, migrations, legacy adapters, or deprecated fields unless
  the task explicitly requires supporting an existing deployed format. Earlier
  development archives and test fixtures are not a compatibility requirement.
- Keep progress in its owning component. Use the catalog's recorded paths and
  completion state instead of reconstructing that state with directory scans or
  maintaining a second set of checkpoint manifests.

## Data integrity and resource use

- Add validation only for concrete workflow or archive-integrity requirements.
  Trust configuration and established internal contracts; let library operations
  reject invalid inputs instead of anticipating every possible failure.
- Trust completed artifacts and metadata produced by this pipeline. Do not
  repeatedly hash or revalidate immutable data, or add recovery for hypothetical
  tampering or spontaneous changes. Validate external input and incomplete work
  at their actual boundaries; explain the concrete failure a new check prevents.
- Preserve source units, grid coordinates, UTC timestamps, interval semantics,
  and decoded numerical precision. Lossless compression does not justify a
  lossy conversion of numeric types.
- Treat each published revision as immutable while readers use it. Additions
  and ERA5 corrections produce replacement monthly revisions, preserving other
  stored variables and hours.
- Keep one writer per archive, coordinate readers with publication, and make
  interrupted publication recoverable. SQLite transactions do not make changes
  to Zarr files atomic.
- Track completion by variable, hour, and source selection. Record completion
  only after validated data are durably published. Remove corresponding GRIB
  downloads only after publication and catalog recording succeed.
- Use mature libraries to decode data formats instead of implementing custom
  format checks. Check the archive invariants needed for publication; avoid
  rescanning or rehashing the entire archive without a demonstrated requirement.
- Process large date ranges through bounded batches. Control per-batch memory,
  temporary storage, concurrency, and retries; respect CDS request limits.
  Use retries with backoff and preserve recoverable progress on interruption.
- Keep credentials out of Git, logs, and archive metadata. Use placeholders in
  examples and check staged files for accidental credential inclusion.

## Python and documentation

- Prefer straightforward code. Avoid thin passthrough helpers, speculative
  abstractions, and checks that duplicate library behavior. Extract a function
  when it gives substantial logic a clear responsibility or enables real reuse.
- Let unexpected errors propagate with their original traceback. Do not catch
  and translate exceptions speculatively; handle them only for a concrete
  recovery action or to keep credentials out of output.
- Give new or materially changed functions, methods, and contracts complete type
  annotations and concise Google-style docstrings. Explain purpose first, then
  `Args`, `Returns`, and `Raises` where applicable.
- Use names that describe the object or action. A name and its first documentation
  sentence should make sense to someone unfamiliar with the implementation.
- Keep human-facing documentation in the README: requirements, setup, inputs,
  usage, and output layout. Describe the current design in present tense.
  Exclude implementation recaps, plans, progress reports, change histories, and
  downstream application use cases.
- Prefer links to existing configuration and executable examples over duplicated
  instructions. Do not create feature manuals under `docs/` unless requested.
  Update the README when user-facing guidance changes, not for routine refactors.
- Write command-line help and errors in plain language, explaining what happened
  and what the user can do next. Avoid unrelated documentation churn.

## Verification

- Test changed behavior in its owning component and interactions at real
  boundaries. Cover relevant failure cases: interrupted downloads, missing
  coverage, repeat runs, monthly replacement, publication failure, and GRIB
  retention until successful publication.
- Do not change tests merely to accept accidental coupling or lost guarantees.
  Do not add tests that only repeat implementation details.
- Test observable workflow behavior and real failure boundaries. Do not invent
  mutations of immutable internal data, invalid internal variables, or legacy
  formats solely to justify defensive code. Remove tests for deliberately
  removed compatibility behavior rather than preserving that behavior for tests.
- Require evidence for debt and performance claims. Measure before and after
  changes to download, conversion, or archive-access performance.
- Run relevant checks during development and all applicable required checks
  before completion. Report exact commands and skipped checks with reasons.
  Claim a check passed only when it ran against the reported code revision.

## GitHub workflow

- Create or identify an issue before implementation. Assign every issue and PR
  to `richpsharp`. If no issue number is available, ask before creating a branch.
- Use one lowercase kebab-case branch name with the appropriate prefix:
  - `feature/<issue-number>-<summary>` for features;
  - `bugfix/<issue-number>-<summary>` for fixes;
  - `task/<issue-number>-<summary>` for docs, maintenance, infrastructure, or refactors.
- Do not use agent names in branch names or other project identifiers. Use
  `richpsharp` where a username is needed.
- Base independent branches on current `main`. Do not stack PRs unless their
  dependency is explicit and unavoidable.
- Create PRs as actual GitHub drafts. After creating or updating a PR, verify its
  draft status and assignee through GitHub. Keep it in draft until `richpsharp`
  personally verifies the relevant behavior. Passing CI alone does not authorize
  marking it ready. Do not merge without an explicit instruction.
- Use a separate commit for each independently reviewable correction. Group
  review comments only when one coherent change addresses the same root cause.
- Reply to each review comment with the correction and a clickable fixing commit
  hash. Leave review threads unresolved for the user.
- Keep PR descriptions proportional to the change: explain the resulting
  behavior, affected responsibilities or contracts, relevant tradeoffs, and
  checks actually run. State skipped checks and their reasons; avoid process
  narratives and reporting templates that add no useful information.
