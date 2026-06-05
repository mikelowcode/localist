# LORA Wiki Schema

This file defines the structure and conventions for wiki pages in this knowledge base.
Replace this placeholder with your real SCHEMA.md before running the WikiAgent.

## Page Types

### concept
A concise explanation of a single idea, term, or technique.
Fields: title, summary, detail, related_concepts, sources.

### research-note
A synthesis of findings from one or more source documents.
Fields: title, query, summary, findings, gaps, sources.

## Conventions

- All pages are Markdown files stored in `wiki/`.
- Page filenames use kebab-case, e.g. `attention-mechanism.md`.
- Each page begins with a YAML front-matter block (title, type, created, updated).
- Sources are cited inline as `(source: <page-or-file-name>)`.
