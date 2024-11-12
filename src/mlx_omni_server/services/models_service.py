import json
from typing import Dict, List, Optional, Tuple

from huggingface_hub import CachedRepoInfo, scan_cache_dir

from ..schemas.models_schema import Model, ModelDeletion, ModelList


class ModelCacheScanner:
    def __init__(self, target_architecture="CausalLM"):
        self.target_architecture = target_architecture
        self._cache_info = None

    @property
    def cache_info(self):
        """Lazy load and cache the scan_cache_dir result"""
        if self._cache_info is None:
            self._cache_info = scan_cache_dir()
        return self._cache_info

    def _refresh_cache_info(self):
        """Force refresh the cache info"""
        self._cache_info = scan_cache_dir()

    def is_model_supported(self, config_data: Dict) -> bool:
        """Check if the model is supported based on its architecture"""
        try:
            architectures = config_data.get("architectures", [])
            return any(self.target_architecture in arch for arch in architectures)
        except Exception:
            return False

    def find_models_in_cache(self) -> List[Tuple[CachedRepoInfo, Dict]]:
        """
        Scan local cache for available models
        Returns: List of tuples containing (CachedRepoInfo, config_dict)
        """
        supported_models = []

        for repo_info in self.cache_info.repos:
            if repo_info.repo_type != "model":
                continue

            first_revision = next(iter(repo_info.revisions), None)
            if not first_revision:
                continue

            config_file = next(
                (f for f in first_revision.files if f.file_name == "config.json"), None
            )
            if not config_file:
                continue

            try:
                with open(config_file.file_path, "r") as f:
                    config_data = json.load(f)
                if self.is_model_supported(config_data):
                    supported_models.append((repo_info, config_data))
            except Exception as e:
                print(f"Error reading config.json for {repo_info.repo_id}: {str(e)}")

        return supported_models

    def get_model_info(self, model_id: str) -> Optional[Tuple[CachedRepoInfo, Dict]]:
        """Get model information from cache"""
        for repo_info in self.cache_info.repos:
            if repo_info.repo_id == model_id and repo_info.repo_type == "model":
                first_revision = next(iter(repo_info.revisions), None)
                if not first_revision:
                    continue

                config_file = next(
                    (f for f in first_revision.files if f.file_name == "config.json"),
                    None,
                )
                if not config_file:
                    continue

                try:
                    with open(config_file.file_path, "r") as f:
                        config_data = json.load(f)
                    return (repo_info, config_data)
                except Exception as e:
                    print(
                        f"Error reading config.json for {repo_info.repo_id}: {str(e)}"
                    )

        return None

    def delete_model(self, model_id: str) -> bool:
        """Delete model from local cache"""
        for repo_info in self.cache_info.repos:
            if repo_info.repo_id == model_id:
                revision_hashes = [rev.commit_hash for rev in repo_info.revisions]
                if not revision_hashes:
                    return False

                try:
                    delete_strategy = self.cache_info.delete_revisions(*revision_hashes)
                    print(
                        f"Model '{model_id}': Will free {delete_strategy.expected_freed_size_str}"
                    )
                    delete_strategy.execute()
                    print(f"Model '{model_id}': Cache deletion completed")
                    self._refresh_cache_info()
                    return True
                except Exception as e:
                    print(f"Error deleting model '{model_id}': {str(e)}")
                    raise

        return False


class ModelsService:
    def __init__(self):
        self.scanner = ModelCacheScanner(target_architecture="CausalLM")
        self.available_models = self._scan_models()

    def _scan_models(self) -> List[Tuple[CachedRepoInfo, Dict]]:
        """Scan local cache for available CausalLM models"""
        try:
            return self.scanner.find_models_in_cache()
        except Exception as e:
            print(f"Error scanning cache: {str(e)}")
            return []

    @staticmethod
    def _get_model_owner(model_id: str) -> str:
        """Extract owner from model ID (part before the /)"""
        return model_id.split("/")[0] if "/" in model_id else model_id

    def list_models(self) -> ModelList:
        """List all available models"""
        models = []
        for repo_info, config_data in self.available_models:
            models.append(
                Model(
                    id=repo_info.repo_id,
                    created=int(repo_info.last_modified),
                    owned_by=self._get_model_owner(repo_info.repo_id),
                    config=config_data,
                )
            )
        return ModelList(data=models)

    def get_model(self, model_id: str) -> Optional[Model]:
        """Get information about a specific model"""
        model_info = self.scanner.get_model_info(model_id)
        if model_info:
            repo_info, config_data = model_info
            return Model(
                id=model_id,
                created=int(repo_info.last_modified),
                owned_by=self._get_model_owner(model_id),
                config=config_data,
            )
        return None

    def delete_model(self, model_id: str) -> ModelDeletion:
        """Delete a model from local cache"""
        if not self.scanner.delete_model(model_id):
            raise ValueError(f"Model '{model_id}' not found in cache")

        self.available_models = self._scan_models()
        return ModelDeletion(id=model_id, deleted=True)
