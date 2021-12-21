from typing import Optional, Dict, Any

import yaml


class ConfigParser:
    def __init__(self, defaults: Optional[Dict[str, Any]]) -> None:
        self._defaults = defaults
        self._config = None

    def read(self, config_path, verbose=False) -> None:
        config = self._defaults
        with open(config_path, 'rb') as f:
            config.update(yaml.safe_load(f))
        self._config = config
        if verbose:
            for k in self._config:
                print(f"{k}: {self._config[k]}")

    def __getitem__(self, item: str) -> Any:
        return self._config[item]
