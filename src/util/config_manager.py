import os
import yaml
from pathlib import Path
from typing import Dict, Any, List, Optional

class AppSettings:
    """
    Manages application settings, loading them from a YAML file.
    This class is a singleton, ensuring a single source of configuration.
    """
    _singleton_instance: Optional['AppSettings'] = None
    _loaded_settings: Optional[Dict[str, Any]] = None
    _config_location: Optional[str] = None

    POSSIBLE_CONFIG_PATHS: List[str] = [
        '/opt/airflow/dags/repo/config/config.yaml',
        'config.yaml',
        'config/config.yaml',
        '../config/config.yaml',
        '../../config/config.yaml',
        '../../../config/config.yaml',
        '/config/config.yaml',
        '../config/config.yaml'
    ]

    def __new__(cls) -> 'AppSettings':
        if cls._singleton_instance is None:
            cls._singleton_instance = super().__new__(cls)
        return cls._singleton_instance

    def __init__(self):
        if self._loaded_settings is None:
            self._fetch_and_load_settings()

    def _find_config_file(self) -> str:
        """
        Locates the configuration file by checking an environment variable and a list of predefined paths.
        """
        # Check environment variable first
        env_config_path = os.getenv("AIRFLOW_CONFIG_PATH")
        if env_config_path and os.path.exists(env_config_path):
            return env_config_path

        # Check predefined paths relative to the project root
        project_root = Path(__file__).resolve().parents[2]
        for path_str in self.POSSIBLE_CONFIG_PATHS:
            # Check relative path from current working dir
            if os.path.exists(path_str):
                return path_str
            
            # Check path relative to project root
            abs_path_candidate = (project_root / path_str).resolve()
            if abs_path_candidate.exists():
                return str(abs_path_candidate)

        raise FileNotFoundError(
            "CRITICAL: Application settings file could not be found. "
            "Set the AIRFLOW_CONFIG_PATH environment variable or place the config file in a recognized location."
        )

    def _fetch_and_load_settings(self) -> None:
        """
        Loads and parses the YAML settings file.
        """
        config_filepath = self._find_config_file()

        try:
            with open(config_filepath, 'r', encoding="UTF-8") as f:
                settings_data = yaml.safe_load(f)

            if not settings_data:
                raise ValueError(f"The settings file at {config_filepath} is empty or improperly formatted.")

            self._loaded_settings = settings_data
            self._config_location = config_filepath
            print(f"Configuration loaded successfully from: {config_filepath}")

        except yaml.YAMLError as exc:
            raise ValueError(f"CRITICAL: Malformed YAML in settings file {config_filepath}: {exc}")
        except Exception as exc:
            raise RuntimeError(f"CRITICAL: Could not process settings from {config_filepath}: {exc}")

    def get(self, setting_path: str, fallback: Any = None) -> Any:
        """
        Retrieves a setting value using a dot-separated path.
        Example: 'database.connection.host'
        """
        if self._loaded_settings is None:
            raise RuntimeError("Settings have not been loaded.")

        path_segments = setting_path.split('.')
        current_value = self._loaded_settings

        for segment in path_segments:
            if isinstance(current_value, dict) and segment in current_value:
                current_value = current_value[segment]
            else:
                return fallback # Path does not exist, return the fallback value

        return current_value
    
    def get_logging_config(self) -> Dict[str, Any]:
        """A more specific getter for the logging section."""
        logging_configuration = self.get("logging")
        if not isinstance(logging_configuration, dict):
            raise ValueError("The 'logging' section is missing or invalid in the configuration file.")
        return logging_configuration

# Global instance to be used across the application
settings = AppSettings()