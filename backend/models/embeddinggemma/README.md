---
license: gemma
pipeline_tag: sentence-similarity
library_name: sentence-transformers
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- mlx
extra_gated_heading: Access EmbeddingGemma on Hugging Face
extra_gated_prompt: To access EmbeddingGemma on Hugging Face, you’re required to review
  and agree to Google’s usage license. To do this, please ensure you’re logged in
  to Hugging Face and click below. Requests are processed immediately.
extra_gated_button_content: Acknowledge license
---

# mlx-community/embeddinggemma-300m-4bit

The Model [mlx-community/embeddinggemma-300m-4bit](https://huggingface.co/mlx-community/embeddinggemma-300m-4bit) was converted to MLX format from [google/embeddinggemma-300m-qat-q4_0-unquantized](https://huggingface.co/google/embeddinggemma-300m-qat-q4_0-unquantized) using mlx-lm version **0.0.4**.

## Use with mlx

```bash
pip install mlx-embeddings
```

```python
from mlx_embeddings import load, generate
import mlx.core as mx

model, tokenizer = load("mlx-community/embeddinggemma-300m-4bit")

# For text embedding
sentences = [
    "task: sentence similarity | query: Nothing really matters.",
    "task: sentence similarity | query: The dog is barking.",
    "task: sentence similarity | query: The dog is barking.",
]

encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='mlx')

# Compute token embeddings
input_ids = encoded_input['input_ids']
attention_mask = encoded_input['attention_mask']
output = model(input_ids, attention_mask)

embeddings = output.text_embeds  # Normalized embeddings

# Compute dot product between normalized embeddings
similarity_matrix = mx.matmul(embeddings, embeddings.T)

print("Similarity matrix between texts:")
print(similarity_matrix)


# You can use these task-specific prefixes for different tasks
task_prefixes = {
    "BitextMining": "task: search result | query: ",
    "Clustering": "task: clustering | query: ",
    "Classification": "task: classification | query: ",
    "MultilabelClassification": "task: classification | query: ",
    "PairClassification": "task: sentence similarity | query: ",
    "InstructionRetrieval": "task: code retrieval | query: ",
    "Reranking": "task: search result | query: ",
    "Retrieval": "task: search result | query: ",
    "Retrieval-query": "task: search result | query: ",
    "Retrieval-document": "title: none | text: ",
    "STS": "task: sentence similarity | query: ",
    "Summarization": "task: summarization | query: ",
    "document": "title: none | text: "
}


```
