import json
import logging
import os
import shutil
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from openai import NotFoundError, OpenAI

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@pytest.fixture
def mock_model_cache_and_client(tmp_path, monkeypatch):
    """Fixture to create a temporary model cache, set HF_HOME, and create a fresh app."""
    # Set HF_HOME environment variable
    hf_home = tmp_path / "hf_home"
    hf_home.mkdir()
    monkeypatch.setenv("HF_HOME", str(hf_home))

    # Define a dummy model to populate the cache
    model_id = "test-org/test-model"
    model_path = hf_home / "hub" / f"models--{model_id.replace('/', '--')}"
    model_path.mkdir(parents=True)

    # Create a dummy revision with a proper commit hash format
    snapshots_path = model_path / "snapshots"
    snapshots_path.mkdir()  # Create the snapshots directory first
    revision_path = snapshots_path / "a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0"  # 40-char hash
    revision_path.mkdir()

    # Create dummy config and other model files
    (revision_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "llama",
                "hidden_size": 64,
                "num_attention_heads": 4,
                "num_hidden_layers": 2,
            }
        )
    )
    (revision_path / "model.safetensors").write_text("dummy model weights")
    (revision_path / "tokenizer.json").write_text("dummy tokenizer data")

    # Create a symlink to the revision
    refs_path = model_path / "refs"
    refs_path.mkdir()
    (refs_path / "main").symlink_to("../snapshots/a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0")

    # Also create the blobs directory (may be needed by HF hub)
    blobs_path = model_path / "blobs"
    blobs_path.mkdir()

    # Now create a fresh app instance
    import importlib
    import sys
    
    # Remove modules from cache if they exist
    modules_to_remove = [
        'src.mlx_omni_server.chat.models.models_service',
        'src.mlx_omni_server.chat.models.router',
        'src.mlx_omni_server.routers',
        'src.mlx_omni_server.main'
    ]
    
    for module in modules_to_remove:
        if module in sys.modules:
            del sys.modules[module]
    
    # Import the main module which will create a fresh app with the new environment
    from src.mlx_omni_server.main import app
    
    client = TestClient(app)
    openai_client = OpenAI(
        base_url="http://test/v1",
        api_key="test",
        http_client=client,
    )
    
    return model_id, client, openai_client


def test_list_models_default(mock_model_cache_and_client):
    """Test listing models without details (default)."""
    model_id, client, openai_client = mock_model_cache_and_client
    model_list = openai_client.models.list()
    assert len(model_list.data) == 1
    for model in model_list.data:
        assert not hasattr(model, "details") or model.details is None


def test_list_models_with_details(mock_model_cache_and_client):
    """Test listing models with the show_details flag."""
    model_id, client, openai_client = mock_model_cache_and_client
    model_list = openai_client.models.list(extra_query={"include_details": True})
    assert len(model_list.data) == 1
    for model in model_list.data:
        assert model.details is not None
        assert isinstance(model.details, dict)


def test_get_existing_model_with_details(mock_model_cache_and_client):
    """Test retrieving a single, existing model with details."""
    model_id, client, openai_client = mock_model_cache_and_client
    model = openai_client.models.retrieve(
        model_id, extra_query={"include_details": True}
    )
    logger.info(f"Retrieved Model with details: {model}")

    assert model is not None
    assert model.id == model_id
    assert model.details is not None
    assert isinstance(model.details, dict)
    assert model.details.get("model_type") == "llama"


def test_get_existing_model_without_details(mock_model_cache_and_client):
    """Test retrieving a single, existing model without details."""
    model_id, client, openai_client = mock_model_cache_and_client
    model = openai_client.models.retrieve(
        model_id, extra_query={"include_details": False}
    )
    logger.info(f"Retrieved Model without details: {model}")

    assert model is not None
    assert model.id == model_id
    assert not hasattr(model, "details") or model.details is None


def test_get_non_existent_model(mock_model_cache_and_client):
    """Test retrieving a non-existent model."""
    model_id, client, openai_client = mock_model_cache_and_client
    non_existent_model_id = "non-existent/model-that-should-not-be-found"
    with pytest.raises(NotFoundError):
        openai_client.models.retrieve(non_existent_model_id)


def test_delete_existing_model(mock_model_cache_and_client):
    """Test deleting an existing model from the cache."""
    model_id, client, openai_client = mock_model_cache_and_client

    # Verify the model exists before deletion
    response = client.get(f"/v1/models/{model_id}")
    assert response.status_code == 200

    # Delete the model
    delete_response = client.delete(f"/v1/models/{model_id}")
    assert delete_response.status_code == 200
    delete_data = delete_response.json()
    assert delete_data["id"] == model_id
    assert delete_data["deleted"] is True

    # Verify the model is gone
    with pytest.raises(NotFoundError):
        openai_client.models.retrieve(model_id)


def test_delete_non_existent_model(mock_model_cache_and_client):
    """Test deleting a non-existent model."""
    model_id, client, openai_client = mock_model_cache_and_client
    non_existent_model_id = "non-existent/model-that-will-not-be-found"
    response = client.delete(f"/v1/models/{non_existent_model_id}")
    assert response.status_code == 404