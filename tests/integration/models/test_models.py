import json
import logging
import sys
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from openai import NotFoundError, OpenAI

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@pytest.fixture
def mock_model_cache_and_client(tmp_path):
    """Fixture to mock model cache and return a client."""

    # Create a dummy config file in tmp_path so it exists on disk (scanner reads it)
    config_path = tmp_path / "config.json"
    config_data = {
        "model_type": "llama",
        "hidden_size": 64,
        "num_attention_heads": 4,
        "num_hidden_layers": 2,
    }
    config_path.write_text(json.dumps(config_data))

    # Create the initial cache info with one model
    mock_cache_info_initial = MagicMock()
    mock_repo = MagicMock()
    mock_repo.repo_id = "test-org/test-model"
    mock_repo.repo_type = "model"
    mock_repo.last_modified = 1234567890

    mock_revision = MagicMock()
    mock_revision.commit_hash = "a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0"

    mock_file = MagicMock()
    mock_file.file_name = "config.json"
    mock_file.file_path = str(config_path)

    mock_revision.files = [mock_file]
    mock_repo.revisions = [mock_revision]

    mock_cache_info_initial.repos = [mock_repo]

    # Setup delete strategy
    mock_delete_strategy = MagicMock()
    mock_delete_strategy.expected_freed_size_str = "100MB"
    mock_cache_info_initial.delete_revisions.return_value = mock_delete_strategy

    # Create the empty cache info (after deletion)
    mock_cache_info_empty = MagicMock()
    mock_cache_info_empty.repos = []

    # Side effect: First call returns initial, subsequent calls return empty (simulating deletion)
    # We use a large list of 'empty' at the end just in case it's called more times
    side_effect = [mock_cache_info_initial] + [mock_cache_info_empty] * 5

    with patch("huggingface_hub.scan_cache_dir", side_effect=side_effect):

        # Remove modules to force reload so ModelsService is re-instantiated
        modules_to_remove = [
            "src.mlx_omni_server.chat.models.models_service",
            "src.mlx_omni_server.chat.models.router",
            "src.mlx_omni_server.routers",
            "src.mlx_omni_server.main",
        ]
        for module in modules_to_remove:
            if module in sys.modules:
                del sys.modules[module]

        # Import main which triggers ModelsService instantiation and scan_cache_dir call
        from src.mlx_omni_server.main import app

        client = TestClient(app)
        openai_client = OpenAI(
            base_url="http://test/v1",
            api_key="test",
            http_client=client,
        )

        yield "test-org/test-model", client, openai_client


def test_list_models_default(mock_model_cache_and_client):
    """Test listing models without details (default)."""
    model_id, client, openai_client = mock_model_cache_and_client
    model_list = openai_client.models.list()
    assert len(model_list.data) == 1
    for model in model_list.data:
        assert not hasattr(model, "details") or model.details is None
    assert model_list.data[0].id == model_id


def test_list_models_with_details(mock_model_cache_and_client):
    """Test listing models with the show_details flag."""
    model_id, client, openai_client = mock_model_cache_and_client
    model_list = openai_client.models.list(extra_query={"include_details": True})
    assert len(model_list.data) == 1
    for model in model_list.data:
        assert model.details is not None
        assert isinstance(model.details, dict)
    assert model_list.data[0].id == model_id


def test_get_existing_model_with_details(mock_model_cache_and_client):
    """Test retrieving a single, existing model with details."""
    model_id, client, openai_client = mock_model_cache_and_client
    model = openai_client.models.retrieve(model_id, extra_query={"include_details": True})
    logger.info(f"Retrieved Model with details: {model}")

    assert model is not None
    assert model.id == model_id
    assert model.details is not None
    assert isinstance(model.details, dict)
    assert model.details.get("model_type") == "llama"


def test_get_existing_model_without_details(mock_model_cache_and_client):
    """Test retrieving a single, existing model without details."""
    model_id, client, openai_client = mock_model_cache_and_client
    model = openai_client.models.retrieve(model_id, extra_query={"include_details": False})
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
