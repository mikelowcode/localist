# Discrimination Band Diagnostic — Finding 1 only

Companion to diag_compare_embedding_models.py. Investigates ONLY the narrow true-negative/true-positive score band; the 0.028 corpus-miss is out of scope and considered closed. Corpus duplication (Finding 2) is written up separately — see the accompanying prose report, not this script.

Corpus: 16 real documents from /Users/michaelfilanc/Projects/lora-app-demo/backend/localist_memory.db

## True-negative queries (8) — top-1 score against real corpus

Query                                                  MLX doc / score                         Ollama doc / score                      
---------------------------------------------------------------------------------------------------------------------------------------
What's a good recipe for chocolate chip cookies?       wiki:lora-persona (0.3892)              wiki:MEMORY (0.3664)                    
Who won the most recent Super Bowl?                    raw:michael (0.3227)                    wiki:MEMORY (0.3834)                    
What caused the fall of the Roman Empire?              wiki:localist-build-order (0.2880)      wiki:localist-software-stack (0.4098)   
How's the weather looking for the weekend?             raw:Localist Software Stack (0.3680)    raw:Localist Runtime Tooling Upate (0.4061)
How do I create a pivot table in Microsoft Excel?      raw:Localist Master Project Outline (0.4654)wiki:localist-design-philosophy (0.4351)
What is the derivative of x squared?                   raw:Localist Software Stack (0.3430)    wiki:michael (0.4046)                   
What are the best beaches to visit in Thailand?        raw:michael (0.3179)                    wiki:MEMORY (0.3712)                    
How does photosynthesis work in plants?                wiki:lora-persona (0.3731)              wiki:how-localist-works (0.4088)        

## True-positive queries (8) — top-1 score against real corpus

Query                                                  MLX doc / score                         Ollama doc / score                      
---------------------------------------------------------------------------------------------------------------------------------------
What are the sequential build phases for developing L  wiki:localist-build-order (0.7482)      wiki:localist-build-order (0.7451)      
What are the five core design pillars of the Localist  wiki:localist-design-philosophy (0.7475)wiki:localist-design-philosophy (0.7260)
What tools does LORA have access to, like web search   wiki:lora-persona (0.6742)              wiki:lora-persona (0.7236)              
Where does Michael live?                               raw:michael (0.6362)                    raw:michael (0.5887)                    
What inference engines does the Localist Runtime Back  wiki:localist-runtime-tooling-update (0.7296)raw:Localist Runtime Tooling Upate (0.7574)
What is the high-level vision and roadmap for the Loc  wiki:localist-build-order (0.6589)      wiki:localist-build-order (0.7341)      
What hardware and models make up the Localist softwar  wiki:localist-software-stack (0.7853)   raw:Localist Software Stack (0.7533)    
What is the MEMORY.md human-readable snapshot?         wiki:how-localist-works (0.5386)        wiki:MEMORY (0.5832)                    

Expected-doc top-1 hit rate — MLX: 5/8  Ollama: 4/8 (informal check that these queries do target an unambiguous doc; not the object of this diagnostic).

## Distribution stats — real corpus

- MLX true-negative:    mean=0.3584  min=0.2880  max=0.4654  stddev=0.0545
- MLX true-positive:    mean=0.6898  min=0.5386  max=0.7853  stddev=0.0796
- MLX gap: NO OVERLAP — clean gap of 0.0732 between neg.max and pos.min.

- Ollama true-negative: mean=0.3982  min=0.3664  max=0.4351  stddev=0.0229
- Ollama true-positive: mean=0.7014  min=0.5832  max=0.7574  stddev=0.0723
- Ollama gap: NO OVERLAP — clean gap of 0.1481 between neg.max and pos.min.

## True-negative queries vs. control documents (corpus-homogeneity check)

If scores against these topically unrelated control docs are similarly 'moderate' to scores against the real corpus, that points to model score-calibration (the model just doesn't produce very low cosine scores for short natural-language queries against any prose). If scores drop noticeably lower here, that points to corpus homogeneity instead (the real corpus's moderate negative scores are because all 16 docs share Localist/AI-assistant vocabulary, not because the model can't discriminate).

Query                                                  MLX control top-1                       Ollama control top-1                    
---------------------------------------------------------------------------------------------------------------------------------------
What's a good recipe for chocolate chip cookies?       control/cookie-recipe (0.6737)          control/cookie-recipe (0.7405)          
Who won the most recent Super Bowl?                    control/lorem-ipsum (0.3725)            control/astronomy (0.4419)              
What caused the fall of the Roman Empire?              control/lorem-ipsum (0.3740)            control/lorem-ipsum (0.4899)            
How's the weather looking for the weekend?             control/lorem-ipsum (0.3642)            control/astronomy (0.4035)              
How do I create a pivot table in Microsoft Excel?      control/lorem-ipsum (0.4422)            control/cookie-recipe (0.4376)          
What is the derivative of x squared?                   control/lorem-ipsum (0.3964)            control/lorem-ipsum (0.4383)            
What are the best beaches to visit in Thailand?        control/lorem-ipsum (0.3193)            control/astronomy (0.3943)              
How does photosynthesis work in plants?                control/lorem-ipsum (0.3989)            control/astronomy (0.4342)              

## Control-document stats vs. real-corpus true-negative stats

- MLX    true-neg vs. real corpus:   mean=0.3584  min=0.2880  max=0.4654  stddev=0.0545
- MLX    true-neg vs. control docs:  mean=0.4176  min=0.3193  max=0.6737  stddev=0.1092
- MLX    mean drop (real - control): -0.0592

- Ollama true-neg vs. real corpus:   mean=0.3982  min=0.3664  max=0.4351  stddev=0.0229
- Ollama true-neg vs. control docs:  mean=0.4725  min=0.3943  max=0.7405  stddev=0.1120
- Ollama mean drop (real - control): -0.0744

**Read this section, not a hardcoded conclusion below** — a large positive drop (scores meaningfully lower against control docs than against the real corpus) supports the corpus-homogeneity hypothesis; a near-zero or negative drop supports the model-calibration hypothesis. This script does not decide between them.

**Observational only.** No threshold change, no corpus change, no code change is proposed by this script.