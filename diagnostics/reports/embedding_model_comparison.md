# Embedding Model Comparison — MLX (embeddinggemma) vs. Ollama (nomic-embed-text)

Corpus: 16 documents from /Users/michaelfilanc/Projects/lora-app-demo/backend/localist_memory.db
MLX vectors: 16  |  Ollama vectors: 16

**Observational only — no threshold conclusions drawn here.** This report exists to compare score distributions and rankings; any similarity-threshold change is a separate follow-up decision.

### Query: 'Tell me how Localist works?'

Rank MLX (embeddinggemma)                         Ollama (nomic-embed-text)                    
-----------------------------------------------------------------------------------------------
1    wiki/localist-master-project-outline  (0.6479)raw/michael  (0.6981)                        
2    wiki/how-localist-works  (0.6375)            raw/Localist Design Philosophy Proposal  (0.6980)
3    wiki/localist-design-philosophy  (0.6350)    wiki/michael  (0.6909)                       
4    raw/Localist Design Philosophy Proposal  (0.6340)raw/Localist Software Stack  (0.6640)        
5    raw/how-localist-works  (0.6032)             raw/Localist Master Project Outline  (0.6521)

Top-1 agreement: NO

### Query: 'How does Localist work?'

Rank MLX (embeddinggemma)                         Ollama (nomic-embed-text)                    
-----------------------------------------------------------------------------------------------
1    wiki/localist-master-project-outline  (0.6702)raw/michael  (0.7075)                        
2    wiki/localist-design-philosophy  (0.6608)    wiki/michael  (0.6994)                       
3    wiki/how-localist-works  (0.6563)            raw/Localist Design Philosophy Proposal  (0.6970)
4    raw/Localist Design Philosophy Proposal  (0.6556)wiki/how-localist-works  (0.6664)            
5    wiki/localist-runtime-tooling-update  (0.6148)raw/Localist Master Project Outline  (0.6637)

Top-1 agreement: NO

### Query: 'localist design philosophy'

Rank MLX (embeddinggemma)                         Ollama (nomic-embed-text)                    
-----------------------------------------------------------------------------------------------
1    wiki/localist-design-philosophy  (0.7632)    raw/Localist Design Philosophy Proposal  (0.7048)
2    raw/Localist Design Philosophy Proposal  (0.7115)wiki/localist-design-philosophy  (0.6865)    
3    raw/michael  (0.6706)                        raw/michael  (0.6730)                        
4    wiki/localist-master-project-outline  (0.6529)wiki/michael  (0.6446)                       
5    wiki/michael  (0.6525)                       wiki/localist-master-project-outline  (0.6165)

Top-1 agreement: NO

### Query: 'What inference engines does Localist support?'

Rank MLX (embeddinggemma)                         Ollama (nomic-embed-text)                    
-----------------------------------------------------------------------------------------------
1    raw/Localist Design Philosophy Proposal  (0.7091)raw/Localist Design Philosophy Proposal  (0.7375)
2    wiki/localist-design-philosophy  (0.6986)    raw/Localist Runtime Tooling Upate  (0.7279) 
3    wiki/localist-master-project-outline  (0.6953)raw/Localist Software Stack  (0.7094)        
4    wiki/localist-software-stack  (0.6886)       raw/michael  (0.6992)                        
5    wiki/localist-runtime-tooling-update  (0.6770)raw/Localist Build Order  (0.6991)           

Top-1 agreement: YES

### Query: 'What is this project about?'

Rank MLX (embeddinggemma)                         Ollama (nomic-embed-text)                    
-----------------------------------------------------------------------------------------------
1    wiki/localist-build-order  (0.5535)          wiki/localist-master-project-outline  (0.5590)
2    wiki/localist-master-project-outline  (0.5467)wiki/localist-build-order  (0.5405)          
3    wiki/how-localist-works  (0.5347)            wiki/localist-design-philosophy  (0.5248)    
4    raw/Localist Master Project Outline  (0.5125)wiki/how-localist-works  (0.5238)            
5    raw/Localist Build Order  (0.4962)           wiki/localist-software-stack  (0.5149)       

Top-1 agreement: NO

### Query: "What's a good recipe for chocolate chip cookies?"

Rank MLX (embeddinggemma)                         Ollama (nomic-embed-text)                    
-----------------------------------------------------------------------------------------------
1    wiki/lora-persona  (0.3892)                  wiki/MEMORY  (0.4109)                        
2    wiki/how-localist-works  (0.3648)            wiki/michael  (0.3449)                       
3    raw/Localist Software Stack  (0.3642)        raw/Localist Design Philosophy Proposal  (0.3413)
4    raw/michael  (0.3617)                        wiki/localist-design-philosophy  (0.3411)    
5    wiki/michael  (0.3607)                       wiki/localist-runtime-tooling-update  (0.3404)

Top-1 agreement: NO

## Aggregate stats across all test queries

- MLX top-1 score:    mean=0.6222  min=0.3892  max=0.7632
- Ollama top-1 score: mean=0.6363  min=0.4109  max=0.7375
- Top-1 doc agreement between models: 1/6 queries
