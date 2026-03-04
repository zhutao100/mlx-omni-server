import pytest

from mlx_omni_server.chat.models.models_service import ModelId, ModelLabel
from mlx_omni_server.chat.schema import (
    ChatCompletionChunk,
    ChatCompletionRequest,
    ChatCompletionResponse,
)
from mlx_omni_server.chat.text_models import BaseTextModel


class _DummyModelCache:
    created_count = 0

    def __init__(self, model_id: ModelId):
        type(self).created_count += 1
        self.model_id = model_id
        self.model_label = ModelLabel.LM


class _DummyTextModel(BaseTextModel):
    def __init__(self, *, model_cache: _DummyModelCache):
        self.model_cache = model_cache

    def generate(
        self,
        request: ChatCompletionRequest,
        *,
        should_cancel=None,
    ) -> ChatCompletionResponse:
        raise AssertionError("generate() should not be called in this test")

    def stream_generate(
        self,
        request: ChatCompletionRequest,
        *,
        should_cancel=None,
    ):
        raise AssertionError("stream_generate() should not be called in this test")
        if False:  # pragma: no cover
            yield ChatCompletionChunk(  # noqa: B018
                id="unused",
                created=0,
                model="unused",
                choices=[],
            )


@pytest.fixture
def _patched_models(monkeypatch):
    from mlx_omni_server.chat.models import models as models_module

    clear_cache_calls: list[None] = []

    def _clear_cache():
        clear_cache_calls.append(None)

    monkeypatch.setattr(models_module, "clear_cache", _clear_cache)
    monkeypatch.setattr(models_module.gc, "collect", lambda: 0)
    monkeypatch.setattr(models_module, "MlxModelCache", _DummyModelCache)
    monkeypatch.setattr(models_module, "MlxLmModel", _DummyTextModel)
    monkeypatch.setattr(models_module, "MlxVlmModel", _DummyTextModel)

    _DummyModelCache.created_count = 0
    models_module.model_cache_manager.clear()
    clear_cache_calls.clear()

    yield models_module, clear_cache_calls

    models_module.model_cache_manager.clear()


def test_load_model_reuses_same_model_id_without_reload(_patched_models):
    models_module, clear_cache_calls = _patched_models

    model1 = models_module.load_model(ModelId(name="test-model"))
    model2 = models_module.load_model(ModelId(name="test-model"))

    assert model1 is model2
    assert _DummyModelCache.created_count == 1
    assert clear_cache_calls == []


def test_load_model_switches_models_and_releases_previous(_patched_models):
    models_module, clear_cache_calls = _patched_models

    model1 = models_module.load_model(ModelId(name="model-a"))
    model2 = models_module.load_model(ModelId(name="model-b"))

    assert model1 is not model2
    assert _DummyModelCache.created_count == 2
    assert len(clear_cache_calls) == 1


def test_load_model_normalizes_empty_adapter_path(_patched_models):
    models_module, _ = _patched_models

    model1 = models_module.load_model(ModelId(name=" test-model ", adapter_path=""))
    model2 = models_module.load_model(ModelId(name="test-model", adapter_path=None))

    assert model1 is model2
    assert _DummyModelCache.created_count == 1


def test_singleton_init_does_not_reset_loaded_model(_patched_models):
    models_module, _ = _patched_models

    manager1 = models_module.MlxModelCacheManager()
    model1 = manager1.load_model(ModelId(name="test-model"))

    manager2 = models_module.MlxModelCacheManager()
    model2 = manager2.load_model(ModelId(name="test-model"))

    assert manager1 is manager2
    assert model1 is model2
    assert _DummyModelCache.created_count == 1


def test_reload_when_cache_exists_but_model_instance_missing(_patched_models):
    models_module, clear_cache_calls = _patched_models

    manager = models_module.model_cache_manager
    model1 = manager.load_model(ModelId(name="test-model"))
    assert _DummyModelCache.created_count == 1

    manager._mlx_model = None  # Simulate inconsistent internal state.
    model2 = manager.load_model(ModelId(name="test-model"))

    assert model1 is not model2
    assert _DummyModelCache.created_count == 2
    assert len(clear_cache_calls) == 1
