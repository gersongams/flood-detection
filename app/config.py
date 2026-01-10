"""Application configuration."""
import os
from pathlib import Path
from dotenv import load_dotenv
from sentinelhub import SHConfig

load_dotenv()

# Sentinel Hub configuration (main service, not Copernicus Data Space)
SH_CLIENT_ID = os.getenv("SH_CLIENT_ID")
SH_CLIENT_SECRET = os.getenv("SH_CLIENT_SECRET")
SH_TOKEN_URL = "https://services.sentinel-hub.com/auth/realms/main/protocol/openid-connect/token"
SH_BASE_URL = "https://services.sentinel-hub.com"

if not SH_CLIENT_ID or not SH_CLIENT_SECRET:
    raise ValueError("SH_CLIENT_ID and SH_CLIENT_SECRET must be set in .env file")

# Redis configuration
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

# Data directories
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = Path(os.getenv("DATA_DIR", BASE_DIR / "data"))
IMAGES_DIR = DATA_DIR / "images"
MODELS_DIR = DATA_DIR / "models"

# Create directories
IMAGES_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Image acquisition settings
MAX_CLOUD_COVERAGE = 25  # Cloud coverage threshold (LEAST_CC still picks best available)
RESOLUTION = 30  # meters (30m for faster downloads, 10m for high quality)
YEARS_BACK = 5
IMAGES_PER_MONTH = 4  # Reduced for faster processing


def get_sentinel_config() -> SHConfig:
    """Get configured SentinelHub config object."""
    config = SHConfig()
    config.sh_client_id = SH_CLIENT_ID
    config.sh_client_secret = SH_CLIENT_SECRET
    config.sh_token_url = SH_TOKEN_URL
    config.sh_base_url = SH_BASE_URL
    return config
