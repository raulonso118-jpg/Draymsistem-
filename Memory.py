import json
import os
from typing import Dict, Any
from config import DraymConfig

class MemoryManager:
    def __init__(self):
        self.file_path = DraymConfig.MEMORY_FILE
        self.data = self._load_memory()

    def _load_memory(self) -> Dict[str, Any]:
        if os.path.exists(self.file_path):
            try:
                with open(self.file_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return {"projects": {}, "learned_concepts": {}}
        return {"projects": {}, "learned_concepts": {}}

    def save_concept(self, key: str, value: Any):
        self.data["learned_concepts"][key] = value
        self._persist()

    def get_concept(self, key: str) -> Any:
        return self.data["learned_concepts"].get(key)

    def _persist(self):
        with open(self.file_path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)
