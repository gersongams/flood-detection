"""Satellite data fetching utilities."""
import numpy as np
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from pathlib import Path
from typing import Generator
import json

from sentinelhub import (
    SHConfig, BBox, CRS, SentinelHubRequest, DataCollection,
    MimeType, bbox_to_dimensions, MosaickingOrder, SentinelHubCatalog,
    SentinelHubDownloadClient
)

from app.config import (
    get_sentinel_config, MAX_CLOUD_COVERAGE, RESOLUTION,
    YEARS_BACK, IMAGES_PER_MONTH, IMAGES_DIR
)

# Evalscript for multi-band data (RGB + NIR for NDVI/NDWI calculation)
EVALSCRIPT_ALL_BANDS = """
//VERSION=3
function setup() {
    return {
        input: [{ bands: ["B02", "B03", "B04", "B08"] }],
        output: { bands: 4, sampleType: "FLOAT32" }
    };
}

function evaluatePixel(sample) {
    return [sample.B02, sample.B03, sample.B04, sample.B08];
}
"""


def get_bbox_from_coords(min_lon: float, min_lat: float, max_lon: float, max_lat: float) -> BBox:
    """Create a BBox object from coordinates."""
    return BBox(bbox=[min_lon, min_lat, max_lon, max_lat], crs=CRS.WGS84)


MAX_IMAGE_SIZE = 2500  # Sentinel Hub API limit


def get_image_size(bbox: BBox, resolution: int = RESOLUTION) -> tuple[int, int]:
    """Calculate image dimensions from bbox and resolution.

    If the resulting image would exceed MAX_IMAGE_SIZE, adjust the resolution
    to fit within the limit.
    """
    width, height = bbox_to_dimensions(bbox, resolution=resolution)

    # Check if we need to adjust for size limits
    if width > MAX_IMAGE_SIZE or height > MAX_IMAGE_SIZE:
        # Calculate the scale factor needed
        scale = max(width / MAX_IMAGE_SIZE, height / MAX_IMAGE_SIZE)
        # Recalculate with adjusted resolution
        adjusted_resolution = int(resolution * scale) + 1
        width, height = bbox_to_dimensions(bbox, resolution=adjusted_resolution)

    return width, height


def get_adjusted_resolution(bbox: BBox, base_resolution: int = RESOLUTION) -> int:
    """Get the resolution needed to fit within API limits."""
    width, height = bbox_to_dimensions(bbox, resolution=base_resolution)

    if width > MAX_IMAGE_SIZE or height > MAX_IMAGE_SIZE:
        scale = max(width / MAX_IMAGE_SIZE, height / MAX_IMAGE_SIZE)
        return int(base_resolution * scale) + 1

    return base_resolution


def generate_time_intervals(
    years_back: int = YEARS_BACK,
    start_date: str | None = None,
    end_date: str | None = None,
    interval: str = "monthly"
) -> Generator[tuple[str, str], None, None]:
    """Generate time intervals for satellite image downloads.

    Args:
        years_back: Number of years back from today (ignored if start/end provided)
        start_date: Start date as "YYYY-MM-DD" string
        end_date: End date as "YYYY-MM-DD" string
        interval: "daily", "weekly", or "monthly"

    Yields:
        Tuples of (start_date, end_date) strings for each interval
    """
    if start_date and end_date:
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
    else:
        end = datetime.now()
        start = end - relativedelta(years=years_back)

    current = start

    while current < end:
        if interval == "daily":
            interval_start = current
            interval_end = current  # Same day
            current += timedelta(days=1)
        elif interval == "weekly":
            interval_start = current
            interval_end = min(current + timedelta(days=6), end)
            current += timedelta(days=7)
        else:  # monthly
            interval_start = current.replace(day=1)
            interval_end = current + relativedelta(months=1) - timedelta(days=1)
            if interval_end > end:
                interval_end = end
            current += relativedelta(months=1)

        yield (
            interval_start.strftime("%Y-%m-%d"),
            interval_end.strftime("%Y-%m-%d")
        )


def fetch_satellite_image(
    bbox: BBox,
    time_interval: tuple[str, str],
    config: SHConfig,
    save_path: Path | None = None
) -> np.ndarray | None:
    """Fetch satellite image for given bbox and time interval."""
    size = get_image_size(bbox)

    request = SentinelHubRequest(
        evalscript=EVALSCRIPT_ALL_BANDS,
        input_data=[SentinelHubRequest.input_data(
            time_interval=time_interval,
            data_collection=DataCollection.SENTINEL2_L2A,
            mosaicking_order=MosaickingOrder.LEAST_CC,
            other_args={'dataFilter': {'maxCloudCoverage': MAX_CLOUD_COVERAGE}}
        )],
        bbox=bbox,
        config=config,
        size=size,
        responses=[SentinelHubRequest.output_response('default', MimeType.TIFF)]
    )

    try:
        data = request.get_data()[0]

        if save_path:
            np.save(save_path, data)

        return data
    except Exception as e:
        print(f"Error fetching image for {time_interval}: {e}")
        return None


def create_download_request(
    bbox: BBox,
    time_interval: tuple[str, str],
    config: SHConfig
) -> SentinelHubRequest:
    """Create a SentinelHubRequest object (without executing it)."""
    size = get_image_size(bbox)

    return SentinelHubRequest(
        evalscript=EVALSCRIPT_ALL_BANDS,
        input_data=[SentinelHubRequest.input_data(
            time_interval=time_interval,
            data_collection=DataCollection.SENTINEL2_L2A,
            mosaicking_order=MosaickingOrder.LEAST_CC,
            other_args={'dataFilter': {'maxCloudCoverage': MAX_CLOUD_COVERAGE}}
        )],
        bbox=bbox,
        config=config,
        size=size,
        responses=[SentinelHubRequest.output_response('default', MimeType.TIFF)]
    )


def batch_download_images(
    bbox: BBox,
    time_intervals: list[tuple[str, str]],
    config: SHConfig,
    max_threads: int = 5
) -> list[np.ndarray | None]:
    """Download multiple images in parallel using multi-threading.

    Args:
        bbox: Bounding box for all requests
        time_intervals: List of (start_date, end_date) tuples
        config: SentinelHub configuration
        max_threads: Number of concurrent download threads

    Returns:
        List of numpy arrays (or None for failed downloads)
    """
    # Create all requests
    requests = [create_download_request(bbox, interval, config) for interval in time_intervals]

    # Extract download objects
    download_requests = [req.download_list[0] for req in requests]

    # Download in parallel
    client = SentinelHubDownloadClient(config=config)

    try:
        data = client.download(download_requests, max_threads=max_threads)
        return data
    except Exception as e:
        print(f"Batch download error: {e}")
        return [None] * len(time_intervals)


def calculate_ndvi(image: np.ndarray) -> np.ndarray:
    """Calculate NDVI from multi-band image.

    NDVI = (NIR - Red) / (NIR + Red) = (B08 - B04) / (B08 + B04)
    """
    b04 = image[:, :, 2]  # Red
    b08 = image[:, :, 3]  # NIR

    with np.errstate(divide='ignore', invalid='ignore'):
        ndvi = (b08 - b04) / (b08 + b04)
        ndvi = np.nan_to_num(ndvi, nan=0.0, posinf=1.0, neginf=-1.0)

    return ndvi


def calculate_ndwi(image: np.ndarray) -> np.ndarray:
    """Calculate NDWI from multi-band image.

    NDWI = (Green - NIR) / (Green + NIR) = (B03 - B08) / (B03 + B08)
    """
    b03 = image[:, :, 1]  # Green
    b08 = image[:, :, 3]  # NIR

    with np.errstate(divide='ignore', invalid='ignore'):
        ndwi = (b03 - b08) / (b03 + b08)
        ndwi = np.nan_to_num(ndwi, nan=0.0, posinf=1.0, neginf=-1.0)

    return ndwi


def check_available_images(
    bbox: BBox,
    time_interval: tuple[str, str],
    config: SHConfig
) -> int:
    """Check how many images are available for the given parameters."""
    catalog = SentinelHubCatalog(config=config)

    search_iterator = catalog.search(
        DataCollection.SENTINEL2_L2A,
        bbox=bbox,
        time=time_interval,
        fields={"include": ["id", "properties.datetime"], "exclude": []},
    )

    return len(list(search_iterator))


def save_metadata(
    job_id: str,
    bbox_coords: dict,
    num_images: int,
    time_range: tuple[str, str]
) -> None:
    """Save job metadata to JSON file."""
    metadata = {
        "job_id": job_id,
        "bbox": bbox_coords,
        "num_images": num_images,
        "time_range": time_range,
        "created_at": datetime.now().isoformat()
    }

    metadata_path = IMAGES_DIR / job_id / "metadata.json"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)


# Evalscript for DEM data (Copernicus DEM)
EVALSCRIPT_DEM = """
//VERSION=3
function setup() {
    return {
        input: [{ bands: ["DEM"] }],
        output: { bands: 1, sampleType: "FLOAT32" }
    };
}

function evaluatePixel(sample) {
    return [sample.DEM];
}
"""


def fetch_dem_data(
    bbox: BBox,
    config: SHConfig,
    save_path: Path | None = None
) -> np.ndarray | None:
    """Fetch Digital Elevation Model (DEM) data for given bbox.

    Uses Copernicus DEM at 10m resolution.
    Returns elevation in meters.
    """
    size = get_image_size(bbox)

    request = SentinelHubRequest(
        evalscript=EVALSCRIPT_DEM,
        input_data=[SentinelHubRequest.input_data(
            data_collection=DataCollection.DEM,
        )],
        bbox=bbox,
        config=config,
        size=size,
        responses=[SentinelHubRequest.output_response('default', MimeType.TIFF)]
    )

    try:
        data = request.get_data()[0]

        # Squeeze to 2D if needed (remove single channel dimension)
        if len(data.shape) == 3 and data.shape[2] == 1:
            data = data[:, :, 0]

        if save_path:
            np.save(save_path, data)

        return data
    except Exception as e:
        print(f"Error fetching DEM data: {e}")
        return None


def calculate_slope(dem: np.ndarray, resolution: float = 10.0) -> np.ndarray:
    """Calculate slope from DEM in degrees.

    Args:
        dem: Digital Elevation Model array (elevation in meters)
        resolution: Pixel resolution in meters (default 10m for Sentinel)

    Returns:
        Slope array in degrees (0-90)
    """
    # Calculate gradients
    dy, dx = np.gradient(dem, resolution)

    # Calculate slope in degrees
    slope = np.arctan(np.sqrt(dx**2 + dy**2)) * (180 / np.pi)

    return slope


def calculate_twi(dem: np.ndarray, resolution: float = 10.0) -> np.ndarray:
    """Calculate Topographic Wetness Index (TWI) using fast approximation.

    Uses slope-based TWI approximation without expensive flow accumulation.
    TWI ≈ ln(A / tan(β)) where A is approximated from local curvature.

    Higher TWI values indicate areas more prone to water accumulation/saturation.

    Args:
        dem: Digital Elevation Model array
        resolution: Pixel resolution in meters

    Returns:
        TWI array (higher = wetter/more flood prone)
    """
    from scipy import ndimage

    # Calculate slope
    dy, dx = np.gradient(dem, resolution)
    slope_rad = np.arctan(np.sqrt(dx**2 + dy**2))
    slope_rad = np.maximum(slope_rad, 0.001)  # Avoid division by zero

    # Approximate contributing area using Gaussian-smoothed inverse elevation
    # Low areas that are surrounded by higher areas get higher values
    smoothed = ndimage.gaussian_filter(dem, sigma=3)
    local_min = ndimage.minimum_filter(dem, size=5)

    # Areas close to local minimum get higher contributing area estimate
    elevation_diff = dem - local_min
    contributing_area = np.exp(-elevation_diff / 10.0) * (resolution ** 2)
    contributing_area = np.maximum(contributing_area, resolution ** 2)

    # Calculate TWI
    with np.errstate(divide='ignore', invalid='ignore'):
        twi = np.log(contributing_area / np.tan(slope_rad))
        twi = np.nan_to_num(twi, nan=0.0, posinf=15.0, neginf=0.0)

    return twi


def calculate_distance_to_water(ndwi: np.ndarray, water_threshold: float = 0.3, resolution: float = 10.0) -> np.ndarray:
    """Calculate distance to nearest water body based on NDWI.

    Args:
        ndwi: NDWI array (can be single image or mean of time series)
        water_threshold: NDWI threshold to classify as water (default 0.3)
        resolution: Pixel resolution in meters

    Returns:
        Distance array in meters (0 = water, higher = further from water)
    """
    from scipy import ndimage

    # Create water mask
    water_mask = ndwi > water_threshold

    # If no water detected, return max distance
    if not water_mask.any():
        return np.full_like(ndwi, fill_value=np.max(ndwi.shape) * resolution)

    # Calculate distance transform (in pixels)
    distance_pixels = ndimage.distance_transform_edt(~water_mask)

    # Convert to meters
    distance_meters = distance_pixels * resolution

    return distance_meters


def create_water_mask(ndwi: np.ndarray, threshold: float = 0.3) -> np.ndarray:
    """Create binary water mask from NDWI.

    Args:
        ndwi: NDWI array
        threshold: NDWI threshold for water classification

    Returns:
        Binary mask (True = water)
    """
    return ndwi > threshold


def calculate_composite_flood_susceptibility(
    dem: np.ndarray,
    slope: np.ndarray,
    twi: np.ndarray,
    distance_to_water: np.ndarray,
    weights: dict | None = None
) -> np.ndarray:
    """Calculate composite flood susceptibility from multiple factors.

    Normalizes and combines:
    - Elevation (lower = higher risk)
    - Slope (flatter = higher risk)
    - TWI (higher = higher risk)
    - Distance to water (closer = higher risk)

    Args:
        dem: Digital Elevation Model
        slope: Slope in degrees
        twi: Topographic Wetness Index
        distance_to_water: Distance to water in meters
        weights: Optional dict with weights for each factor
            Default: {"elevation": 0.25, "slope": 0.25, "twi": 0.3, "distance": 0.2}

    Returns:
        Composite susceptibility score (0-1, higher = more flood prone)
    """
    if weights is None:
        weights = {"elevation": 0.25, "slope": 0.25, "twi": 0.3, "distance": 0.2}

    def normalize(arr: np.ndarray, invert: bool = False) -> np.ndarray:
        """Normalize array to 0-1 range."""
        min_val, max_val = np.nanmin(arr), np.nanmax(arr)
        if max_val - min_val == 0:
            return np.zeros_like(arr)
        normalized = (arr - min_val) / (max_val - min_val)
        return 1 - normalized if invert else normalized

    # Normalize factors (invert where lower value = higher risk)
    elev_norm = normalize(dem, invert=True)  # Lower elevation = higher risk
    slope_norm = normalize(slope, invert=True)  # Flatter = higher risk
    twi_norm = normalize(twi, invert=False)  # Higher TWI = higher risk
    dist_norm = normalize(distance_to_water, invert=True)  # Closer to water = higher risk

    # Calculate weighted composite
    composite = (
        weights["elevation"] * elev_norm +
        weights["slope"] * slope_norm +
        weights["twi"] * twi_norm +
        weights["distance"] * dist_norm
    )

    return composite
