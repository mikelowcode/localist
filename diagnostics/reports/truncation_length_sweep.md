# Truncation Length Sweep — does the 500-char embedding ceiling cost retrieval quality?

Third diagnostic in this series. Reuses the exact 8 true-positive / 8 true-negative queries from diag_discrimination_band.py. Sweeps document-embedding truncation length (500 = current production ceiling in memory_manager.py's index_document(); 1500; 3000; full-text = no truncation, matching the prior two diagnostics). Query text is never truncated, matching production's query_corpus() behavior — only stored document embeddings are truncated.

**Observational only.** memory_manager.py's content[:500] truncation is unmodified; no re-index of the real database occurred; no threshold or ceiling value is changed by this script.

Corpus: 16 real documents from /Users/michaelfilanc/Projects/lora-app-demo/backend/localist_memory.db

## MLX — gap vs. truncation length

Length      TP mean     TN mean     Gap         Overlap?  
----------------------------------------------------------
500         0.7262      0.3805      +0.0603     NO        
1500        0.7036      0.3696      +0.0488     NO        
3000        0.6898      0.3602      +0.0586     NO        
full-text   0.6898      0.3584      +0.0732     NO        

## Ollama — gap vs. truncation length

Length      TP mean     TN mean     Gap         Overlap?  
----------------------------------------------------------
500         0.7400      0.4065      +0.1617     NO        
1500        0.7036      0.3972      +0.1413     NO        
3000        0.7013      0.3965      +0.1471     NO        
full-text   0.7014      0.3982      +0.1481     NO        

## Trend summary (observation, not a decision)

- MLX: gap at 500 chars = +0.0603; gap at full-text = +0.0732 — going from the production ceiling to full-text **widens** the gap (delta +0.0130). Full progression: 500=+0.0603, 1500=+0.0488, 3000=+0.0586, full-text=+0.0732
- Ollama: gap at 500 chars = +0.1617; gap at full-text = +0.1481 — going from the production ceiling to full-text **narrows** the gap (delta -0.0135). Full progression: 500=+0.1617, 1500=+0.1413, 3000=+0.1471, full-text=+0.1481

## What falls inside vs. outside the truncation window (qualitative)

For 4 of the corpus's longer documents, showing what content each truncation length actually captures — to interpret *why* any pattern above occurs, not just that it occurs.

### `wiki:MEMORY` (full length: 4070 chars)

- **500 chars**: cuts off after `...' Brave Search.\n- **project_fact** (90%, '` → `'general, model_extracted) — The user is '...`
- **1500 chars**: cuts off after `...'ted) — The user specifies that Localist '` → `'should support cloud models as stateless'...`
- **3000 chars**: cuts off after `...'esign\n\n## June 29, 2026\n\n- **project_fac'` → `'t** (90%, general, model_extracted) — Th'...`

### `raw:Localist Master Project Outline` (full length: 3989 chars)

- **500 chars**: cuts off after `...'i pages.\n- **Research Retrieval:** Resea'` → `'rchAgent queries the SQLite corpus, extr'...`
- **1500 chars**: cuts off after `...'ndalone MLX-LM embedding loader\n  base_r'` → `'untime_client.py        — Runtime protoc'...`
- **3000 chars**: cuts off after `...'ite index, keyword + cosine retrieval, r'` → `'etrieval cache |\n| Embeddings | embeddin'...`

### `raw:Localist Build Order` (full length: 3625 chars)

- **500 chars**: cuts off after `...'e 2 — Agent Core and Controller\n- Define'` → `' AgentInterface protocol and SubTask / A'...`
- **1500 chars**: cuts off after `...'ommunity/embeddinggemma-300m-4bit.\n- Wir'` → `'e embed_fn into MemoryManager for cosine'...`
- **3000 chars**: cuts off after `...'aining raw files: Localist Build Order.m'` → `'d, Localist Software Stack.md.\n- Validat'...`

### `wiki:localist-design-philosophy` (full length: 2086 chars)

- **500 chars**: cuts off after `...'t separation between deterministic plann'` → `'ing and model execution.\n\n## Details\n\n##'...`
- **1500 chars**: cuts off after `...' References SQLite and the local inferen'` → `'ce stack.\n- [[how-localist-works]] — Ref'...`
- **3000 chars**: covers the entire document (shorter than this length).

Observed in this corpus: none of these 4 documents' 500-char cutoff lands on a clean sentence boundary — each truncates mid-word or mid-clause into what would otherwise be substantive content (not filler). `wiki:localist-design-philosophy` additionally spends ~140 of its 500-char budget on YAML frontmatter (`---\ntitle: ...\n---\n\n## Summary\n\n`) before reaching any actual body text, so its effective substantive-content window is closer to ~360 chars, not the full 500. Neither pattern is generalized here to "most wiki docs" — this is 4 documents, reported individually; see the per-doc detail above.

## Appendix — raw per-query scores at every truncation length

### MLX

#### Truncation = 500

True-positive queries:

- 'What are the sequential build phases for developing Localist?' → `wiki:localist-build-order` (0.7786)
- 'What are the five core design pillars of the Localist design philosophy?' → `wiki:localist-design-philosophy` (0.8205)
- 'What tools does LORA have access to, like web search and file operations?' → `wiki:lora-persona` (0.7509)
- 'Where does Michael live?' → `raw:michael` (0.6708)
- 'What inference engines does the Localist Runtime Backend Layer support?' → `wiki:localist-master-project-outline` (0.7170)
- 'What is the high-level vision and roadmap for the Localist project?' → `raw:Localist Master Project Outline` (0.6959)
- 'What hardware and models make up the Localist software stack?' → `wiki:localist-software-stack` (0.8047)
- 'What is the MEMORY.md human-readable snapshot?' → `wiki:MEMORY` (0.5709)

True-negative queries:

- "What's a good recipe for chocolate chip cookies?" → `wiki:michael` (0.3789)
- 'Who won the most recent Super Bowl?' → `wiki:localist-software-stack` (0.3488)
- 'What caused the fall of the Roman Empire?' → `wiki:how-localist-works` (0.3206)
- "How's the weather looking for the weekend?" → `wiki:localist-software-stack` (0.3893)
- 'How do I create a pivot table in Microsoft Excel?' → `raw:Localist Master Project Outline` (0.5106)
- 'What is the derivative of x squared?' → `wiki:localist-software-stack` (0.3689)
- 'What are the best beaches to visit in Thailand?' → `wiki:localist-design-philosophy` (0.3371)
- 'How does photosynthesis work in plants?' → `wiki:how-localist-works` (0.3897)

#### Truncation = 1500

True-positive queries:

- 'What are the sequential build phases for developing Localist?' → `wiki:localist-build-order` (0.7506)
- 'What are the five core design pillars of the Localist design philosophy?' → `wiki:localist-design-philosophy` (0.7718)
- 'What tools does LORA have access to, like web search and file operations?' → `wiki:lora-persona` (0.7192)
- 'Where does Michael live?' → `raw:michael` (0.6362)
- 'What inference engines does the Localist Runtime Backend Layer support?' → `wiki:localist-software-stack` (0.7473)
- 'What is the high-level vision and roadmap for the Localist project?' → `wiki:localist-build-order` (0.6578)
- 'What hardware and models make up the Localist software stack?' → `wiki:localist-software-stack` (0.7929)
- 'What is the MEMORY.md human-readable snapshot?' → `wiki:MEMORY` (0.5531)

True-negative queries:

- "What's a good recipe for chocolate chip cookies?" → `wiki:lora-persona` (0.3921)
- 'Who won the most recent Super Bowl?' → `wiki:localist-design-philosophy` (0.3234)
- 'What caused the fall of the Roman Empire?' → `wiki:lora-persona` (0.2988)
- "How's the weather looking for the weekend?" → `wiki:localist-software-stack` (0.3765)
- 'How do I create a pivot table in Microsoft Excel?' → `raw:Localist Master Project Outline` (0.5043)
- 'What is the derivative of x squared?' → `raw:Localist Software Stack` (0.3472)
- 'What are the best beaches to visit in Thailand?' → `wiki:lora-persona` (0.3321)
- 'How does photosynthesis work in plants?' → `wiki:lora-persona` (0.3823)

#### Truncation = 3000

True-positive queries:

- 'What are the sequential build phases for developing Localist?' → `wiki:localist-build-order` (0.7482)
- 'What are the five core design pillars of the Localist design philosophy?' → `wiki:localist-design-philosophy` (0.7475)
- 'What tools does LORA have access to, like web search and file operations?' → `wiki:lora-persona` (0.6742)
- 'Where does Michael live?' → `raw:michael` (0.6362)
- 'What inference engines does the Localist Runtime Backend Layer support?' → `wiki:localist-runtime-tooling-update` (0.7296)
- 'What is the high-level vision and roadmap for the Localist project?' → `wiki:localist-build-order` (0.6589)
- 'What hardware and models make up the Localist software stack?' → `wiki:localist-software-stack` (0.7853)
- 'What is the MEMORY.md human-readable snapshot?' → `wiki:how-localist-works` (0.5386)

True-negative queries:

- "What's a good recipe for chocolate chip cookies?" → `wiki:lora-persona` (0.3892)
- 'Who won the most recent Super Bowl?' → `raw:michael` (0.3227)
- 'What caused the fall of the Roman Empire?' → `wiki:localist-build-order` (0.2880)
- "How's the weather looking for the weekend?" → `raw:Localist Software Stack` (0.3680)
- 'How do I create a pivot table in Microsoft Excel?' → `raw:Localist Master Project Outline` (0.4800)
- 'What is the derivative of x squared?' → `raw:Localist Software Stack` (0.3430)
- 'What are the best beaches to visit in Thailand?' → `raw:michael` (0.3179)
- 'How does photosynthesis work in plants?' → `wiki:lora-persona` (0.3731)

#### Truncation = full-text

True-positive queries:

- 'What are the sequential build phases for developing Localist?' → `wiki:localist-build-order` (0.7482)
- 'What are the five core design pillars of the Localist design philosophy?' → `wiki:localist-design-philosophy` (0.7475)
- 'What tools does LORA have access to, like web search and file operations?' → `wiki:lora-persona` (0.6742)
- 'Where does Michael live?' → `raw:michael` (0.6362)
- 'What inference engines does the Localist Runtime Backend Layer support?' → `wiki:localist-runtime-tooling-update` (0.7296)
- 'What is the high-level vision and roadmap for the Localist project?' → `wiki:localist-build-order` (0.6589)
- 'What hardware and models make up the Localist software stack?' → `wiki:localist-software-stack` (0.7853)
- 'What is the MEMORY.md human-readable snapshot?' → `wiki:how-localist-works` (0.5386)

True-negative queries:

- "What's a good recipe for chocolate chip cookies?" → `wiki:lora-persona` (0.3892)
- 'Who won the most recent Super Bowl?' → `raw:michael` (0.3227)
- 'What caused the fall of the Roman Empire?' → `wiki:localist-build-order` (0.2880)
- "How's the weather looking for the weekend?" → `raw:Localist Software Stack` (0.3680)
- 'How do I create a pivot table in Microsoft Excel?' → `raw:Localist Master Project Outline` (0.4654)
- 'What is the derivative of x squared?' → `raw:Localist Software Stack` (0.3430)
- 'What are the best beaches to visit in Thailand?' → `raw:michael` (0.3179)
- 'How does photosynthesis work in plants?' → `wiki:lora-persona` (0.3731)

### Ollama

#### Truncation = 500

True-positive queries:

- 'What are the sequential build phases for developing Localist?' → `wiki:localist-build-order` (0.7732)
- 'What are the five core design pillars of the Localist design philosophy?' → `wiki:localist-design-philosophy` (0.8043)
- 'What tools does LORA have access to, like web search and file operations?' → `wiki:lora-persona` (0.7712)
- 'Where does Michael live?' → `raw:michael` (0.6084)
- 'What inference engines does the Localist Runtime Backend Layer support?' → `raw:Localist Runtime Tooling Upate` (0.8072)
- 'What is the high-level vision and roadmap for the Localist project?' → `wiki:localist-master-project-outline` (0.7542)
- 'What hardware and models make up the Localist software stack?' → `raw:Localist Software Stack` (0.7870)
- 'What is the MEMORY.md human-readable snapshot?' → `wiki:MEMORY` (0.6143)

True-negative queries:

- "What's a good recipe for chocolate chip cookies?" → `wiki:MEMORY` (0.3812)
- 'Who won the most recent Super Bowl?' → `wiki:MEMORY` (0.3862)
- 'What caused the fall of the Roman Empire?' → `wiki:MEMORY` (0.4372)
- "How's the weather looking for the weekend?" → `wiki:localist-software-stack` (0.4032)
- 'How do I create a pivot table in Microsoft Excel?' → `wiki:how-localist-works` (0.4468)
- 'What is the derivative of x squared?' → `wiki:localist-design-philosophy` (0.4105)
- 'What are the best beaches to visit in Thailand?' → `wiki:lora-persona` (0.3719)
- 'How does photosynthesis work in plants?' → `wiki:how-localist-works` (0.4147)

#### Truncation = 1500

True-positive queries:

- 'What are the sequential build phases for developing Localist?' → `wiki:localist-build-order` (0.7446)
- 'What are the five core design pillars of the Localist design philosophy?' → `wiki:localist-design-philosophy` (0.7383)
- 'What tools does LORA have access to, like web search and file operations?' → `wiki:lora-persona` (0.7328)
- 'Where does Michael live?' → `raw:michael` (0.5887)
- 'What inference engines does the Localist Runtime Backend Layer support?' → `raw:Localist Runtime Tooling Upate` (0.7677)
- 'What is the high-level vision and roadmap for the Localist project?' → `wiki:localist-build-order` (0.7302)
- 'What hardware and models make up the Localist software stack?' → `raw:Localist Software Stack` (0.7406)
- 'What is the MEMORY.md human-readable snapshot?' → `wiki:MEMORY` (0.5861)

True-negative queries:

- "What's a good recipe for chocolate chip cookies?" → `wiki:MEMORY` (0.3576)
- 'Who won the most recent Super Bowl?' → `raw:Localist Runtime Tooling Upate` (0.3729)
- 'What caused the fall of the Roman Empire?' → `wiki:localist-software-stack` (0.4200)
- "How's the weather looking for the weekend?" → `wiki:localist-software-stack` (0.4038)
- 'How do I create a pivot table in Microsoft Excel?' → `wiki:how-localist-works` (0.4449)
- 'What is the derivative of x squared?' → `wiki:michael` (0.4046)
- 'What are the best beaches to visit in Thailand?' → `wiki:lora-persona` (0.3534)
- 'How does photosynthesis work in plants?' → `wiki:how-localist-works` (0.4201)

#### Truncation = 3000

True-positive queries:

- 'What are the sequential build phases for developing Localist?' → `wiki:localist-build-order` (0.7451)
- 'What are the five core design pillars of the Localist design philosophy?' → `wiki:localist-design-philosophy` (0.7260)
- 'What tools does LORA have access to, like web search and file operations?' → `wiki:lora-persona` (0.7236)
- 'Where does Michael live?' → `raw:michael` (0.5887)
- 'What inference engines does the Localist Runtime Backend Layer support?' → `raw:Localist Runtime Tooling Upate` (0.7574)
- 'What is the high-level vision and roadmap for the Localist project?' → `wiki:localist-build-order` (0.7341)
- 'What hardware and models make up the Localist software stack?' → `raw:Localist Software Stack` (0.7533)
- 'What is the MEMORY.md human-readable snapshot?' → `wiki:MEMORY` (0.5822)

True-negative queries:

- "What's a good recipe for chocolate chip cookies?" → `wiki:MEMORY` (0.3682)
- 'Who won the most recent Super Bowl?' → `wiki:MEMORY` (0.3770)
- 'What caused the fall of the Roman Empire?' → `wiki:localist-software-stack` (0.4098)
- "How's the weather looking for the weekend?" → `raw:Localist Runtime Tooling Upate` (0.4061)
- 'How do I create a pivot table in Microsoft Excel?' → `wiki:localist-design-philosophy` (0.4351)
- 'What is the derivative of x squared?' → `wiki:michael` (0.4046)
- 'What are the best beaches to visit in Thailand?' → `wiki:MEMORY` (0.3620)
- 'How does photosynthesis work in plants?' → `wiki:how-localist-works` (0.4088)

#### Truncation = full-text

True-positive queries:

- 'What are the sequential build phases for developing Localist?' → `wiki:localist-build-order` (0.7451)
- 'What are the five core design pillars of the Localist design philosophy?' → `wiki:localist-design-philosophy` (0.7260)
- 'What tools does LORA have access to, like web search and file operations?' → `wiki:lora-persona` (0.7236)
- 'Where does Michael live?' → `raw:michael` (0.5887)
- 'What inference engines does the Localist Runtime Backend Layer support?' → `raw:Localist Runtime Tooling Upate` (0.7574)
- 'What is the high-level vision and roadmap for the Localist project?' → `wiki:localist-build-order` (0.7341)
- 'What hardware and models make up the Localist software stack?' → `raw:Localist Software Stack` (0.7533)
- 'What is the MEMORY.md human-readable snapshot?' → `wiki:MEMORY` (0.5832)

True-negative queries:

- "What's a good recipe for chocolate chip cookies?" → `wiki:MEMORY` (0.3664)
- 'Who won the most recent Super Bowl?' → `wiki:MEMORY` (0.3834)
- 'What caused the fall of the Roman Empire?' → `wiki:localist-software-stack` (0.4098)
- "How's the weather looking for the weekend?" → `raw:Localist Runtime Tooling Upate` (0.4061)
- 'How do I create a pivot table in Microsoft Excel?' → `wiki:localist-design-philosophy` (0.4351)
- 'What is the derivative of x squared?' → `wiki:michael` (0.4046)
- 'What are the best beaches to visit in Thailand?' → `wiki:MEMORY` (0.3712)
- 'How does photosynthesis work in plants?' → `wiki:how-localist-works` (0.4088)
