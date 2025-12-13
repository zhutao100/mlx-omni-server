# Image Generation

This API is used for generating images.

```python
image_response = client.images.generate(
    model="argmaxinc/mlx-FLUX.1-schnell",
    prompt="A serene landscape with mountains and a lake",
    n=1,
    size="512x512"
)
```

<details>
<summary>Curl Example</summary>

```shell
curl http://localhost:10240/v1/images/generations \
  -H "Content-Type: application/json" \
  -d '{
    "model": "argmaxinc/mlx-FLUX.1-schnell",
    "prompt": "A cute baby sea otter",
    "n": 1,
    "size": "1024x1024"
  }'
```

</details>
