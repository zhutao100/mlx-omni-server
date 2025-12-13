# Embeddings

This API is used for creating text embeddings.

```python
# Generate embedding for a single text
response = client.embeddings.create(
    model="mlx-community/all-MiniLM-L6-v2-4bit", input="I like reading"
)

# Examine the response structure
print(f"Response type: {type(response)}")
print(f"Model used: {response.model}")
print(f"Embedding dimension: {len(response.data[0].embedding)}")
```

<details>
<summary><strong>cURL Example</strong></summary>

```bash
curl http://localhost:10240/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-community/all-MiniLM-L6-v2-4bit",
    "input": ["Hello world!", "Embeddings are useful for semantic search."]
  }'
```

</details>
