from functools import lru_cache
from typing import Dict, List, Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    model_backend: str = Field(default="mock", validation_alias="MODEL_BACKEND")
    model_checkpoint: Optional[str] = Field(default=None, validation_alias="MODEL_CHECKPOINT")
    model_config_path: Optional[str] = Field(default=None, validation_alias="MODEL_CONFIG")
    model_code_dir: Optional[str] = Field(default=None, validation_alias="MODEL_CODE_DIR")
    device: str = Field(default="cuda:0", validation_alias="DEVICE")
    model_version: str = Field(default="mask2former-hibou-pathtests-iter8000", validation_alias="MODEL_VERSION")
    min_contour_area_px: float = Field(default=32.0, validation_alias="MIN_CONTOUR_AREA_PX")
    disable_pretrained: bool = Field(default=True, validation_alias="DISABLE_PRETRAINED")

    class_names: List[str] = [
        "bg",
        "tumor_gland",
        "tumor_stroma",
        "normal_gland",
        "normal_stroma",
    ]

    segment_class_ids: Dict[str, List[int]] = {
        "all": [1, 2, 3, 4],
        "bg": [0],
        "tumor_gland": [1],
        "tumor_stroma": [2],
        "normal_gland": [3],
        "normal_stroma": [4],
    }


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
