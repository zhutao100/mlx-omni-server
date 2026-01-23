# Image Generation

This API is used for generating images.

This route requires the optional extra: `pip install ".[images]"`. If not installed, the route returns `501 Not Implemented` with an install hint.

```python
image_response = client.images.generate(
    model="filipstrand/Z-Image-Turbo-mflux-4bit",
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
    "model": "filipstrand/Z-Image-Turbo-mflux-4bit",
    "prompt": "A cute baby sea otter",
    "n": 1,
    "size": "1024x1024",
    "response_format": "b64_json"
  }'
```

</details>
