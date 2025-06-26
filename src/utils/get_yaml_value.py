import yaml
from pathlib import Path
from functools import lru_cache


class ConfigError(Exception):
    pass


@lru_cache()
def load_config(config_path: str = "../configs/train_config.yaml") -> dict:
    """Загружает и кэширует конфиг."""
    path = Path('../configs/train_config.yaml')
    if not path.exists():
        raise ConfigError(f"Config file {config_path} not found!")

    with open(path, "r") as f:
        return yaml.safe_load(f)


def get_yaml_value(key: str, default=None, config_path: str = "../configs/train_config.yaml"):
    """Получает значение из конфига по ключу в формате 'nested.key'."""
    config = load_config(config_path)
    keys = key.split('.')
    value = config

    try:
        for k in keys:
            value = value[k]
        return value
    except (KeyError, TypeError):
        return default

