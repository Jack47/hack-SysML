# SysML Knowledge Base Guide

This repository is a personal knowledge base about systems for machine learning. Treat it as a persistent, compounding wiki: the human curates sources and asks questions; the agent performs the filing, synthesis, cross-linking, and maintenance.

## Repository Layers

- `raw/` contains immutable source material. Add sources here, but never rewrite an ingested source.
- `wiki/` contains agent-maintained synthesis. The agent may create, revise, merge, and cross-link these pages.
- Existing topic directories such as `papers/`, `frameworks/`, and `memory-efficiency/` are legacy notes. Keep their paths stable. Consult them when answering questions and migrate useful knowledge into `wiki/` only when the user requests it or a new ingest materially updates it.
- `wiki/index.md` is the content-oriented catalog of every article under `wiki/`.
- `wiki/log.md` is the append-only history of ingests, archived queries, and lint passes.

## Working Rules

- Match the user's language. Preserve useful technical terms, symbols, and paper titles in their original language.
- Prefer primary sources: papers, official documentation, specifications, and source code. Distinguish sourced facts from inference.
- Use standard Markdown and relative links. Keep topic nesting to `wiki/<topic>/<article>.md` and `raw/<topic>/<source>.md`.
- Name files in lowercase kebab-case. Use one focused concept per wiki article.
- Do not fabricate citations, publication dates, measurements, or experimental conclusions. Mark unknown metadata as `Unknown`.
- Make the smallest coherent change. Do not reorganize unrelated legacy notes or repair unrelated links during an ingest.

## Ingest Workflow

When the user asks to add, ingest, file, or learn from a source:

1. Search `raw/`, `wiki/index.md`, and relevant legacy notes for duplicates and related material.
2. Save the source to `raw/<topic>/YYYY-MM-DD-<slug>.md`. Include title, source URL or origin, collected date, and published date. Preserve the source faithfully; remove only navigation or formatting noise.
3. Compile the source into `wiki/`: merge it into an article with the same core thesis or create a focused article for a new concept. Attribute disagreements rather than flattening them.
4. Scan related pages for materially affected claims and add useful cross-links.
5. Add or update every touched article in `wiki/index.md` with a one-line summary and `Updated` date.
6. Append an entry to `wiki/log.md` using `## [YYYY-MM-DD] ingest | <primary article title>` and list cascade-updated pages below it.

A wiki article starts with:

```markdown
# Title

> Sources: Author or organization, YYYY-MM-DD
> Raw: [Source](../../raw/topic/source.md)
```

Follow this with `## Overview`, concept-specific sections, and `## See Also` when relevant. Synthesize instead of copying the raw source.

## Query Workflow

When answering a knowledge question:

1. Read `wiki/index.md` first.
2. Search relevant wiki pages, then legacy notes, then external primary sources if the repository is insufficient or freshness matters.
3. Answer with links to the wiki or legacy pages used and external citations when applicable.
4. Do not write the answer into the repository unless the user explicitly asks to archive or save it.

For an archived answer, create a new focused page under `wiki/`, prefix its index summary with `[Archived]`, and append `## [YYYY-MM-DD] query | Archived: <title>` to the log.

## Lint Workflow

When asked to lint or health-check the knowledge base:

- Auto-fix deterministic issues: missing index entries, uniquely resolvable broken internal links, broken raw paths with one unambiguous match, and obvious stale `See Also` links.
- Report rather than auto-fix judgment calls: contradictory claims, stale conclusions, missing topic pages, cross-topic gaps, and orphan articles.
- Append `## [YYYY-MM-DD] lint | <N> issues found, <M> auto-fixed` to `wiki/log.md`.

## Verification

Before finishing a write operation:

- Confirm every new relative link resolves.
- Confirm every wiki article is represented exactly once in `wiki/index.md`.
- Confirm raw sources were not modified after ingestion.
- Run any repository-provided checks. If none exist, report the link and index checks performed.

