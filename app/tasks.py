"""Celery tasks for background processing."""
import json
import logging
import shutil
import numpy as np
from pathlib import Path
from datetime import datetime
import time

from app.celery_app import celery_app

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
from app.config import get_sentinel_config, IMAGES_DIR, MODELS_DIR, YEARS_BACK
from app.utils.satellite import (
    get_bbox_from_coords, generate_time_intervals,
    fetch_satellite_image, save_metadata,
    calculate_ndvi, calculate_ndwi,
    fetch_dem_data, calculate_slope, calculate_twi,
    batch_download_images
)
from app.models.flood_model import (
    prepare_training_data, train_model, predict_flood_risk, load_model
)


# Rate limiting settings - Copernicus is strict, go slow
BASE_DELAY = 3.0  # 3 seconds between requests
MAX_RETRIES = 3   # Max retries per image
BACKOFF_FACTOR = 2  # Exponential backoff multiplier


def load_download_progress(job_dir: Path) -> dict:
    """Load download progress from cache file."""
    progress_file = job_dir / "download_progress.json"
    if progress_file.exists():
        with open(progress_file) as f:
            return json.load(f)
    return {"downloaded": [], "failed": [], "skipped": []}


def save_download_progress(job_dir: Path, progress: dict):
    """Save download progress to cache file for resume capability."""
    progress_file = job_dir / "download_progress.json"
    with open(progress_file, "w") as f:
        json.dump(progress, f, indent=2)


@celery_app.task(bind=True)
def download_images_task(
    self,
    job_id: str,
    min_lon: float,
    min_lat: float,
    max_lon: float,
    max_lat: float,
    start_date: str | None = None,
    end_date: str | None = None,
    interval: str = "daily",
    years_back: int = YEARS_BACK,
    resume: bool = True
):
    """Download satellite images for the given bounding box.

    Supports custom date ranges, daily/weekly/monthly intervals,
    caching, and resume capability.

    Args:
        job_id: Unique job identifier
        min_lon, min_lat, max_lon, max_lat: Bounding box coordinates
        start_date: Start date "YYYY-MM-DD" (e.g., "2017-01-01" for Piura floods)
        end_date: End date "YYYY-MM-DD" (e.g., "2017-12-31")
        interval: "daily", "weekly", or "monthly"
        years_back: Years back from today (ignored if start/end provided)
        resume: If True, skip already downloaded images
    """
    logger.info(f"=== STARTING DOWNLOAD TASK ===")
    logger.info(f"Job ID: {job_id}")
    logger.info(f"BBox: ({min_lat}, {min_lon}) to ({max_lat}, {max_lon})")
    logger.info(f"Date range: {start_date} to {end_date}, interval: {interval}")

    config = get_sentinel_config()
    bbox = get_bbox_from_coords(min_lon, min_lat, max_lon, max_lat)
    logger.info(f"Sentinel Hub config loaded, bbox created")

    job_dir = IMAGES_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    # Global cache directory based on bbox (shared across jobs)
    bbox_key = f"{min_lat:.2f}_{min_lon:.2f}_{max_lat:.2f}_{max_lon:.2f}"
    global_cache_dir = IMAGES_DIR / "cache" / bbox_key
    global_cache_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Global cache dir: {global_cache_dir}")

    # Generate time intervals
    logger.info(f"Generating time intervals...")
    intervals = list(generate_time_intervals(
        years_back=years_back,
        start_date=start_date,
        end_date=end_date,
        interval=interval
    ))
    total = len(intervals)
    logger.info(f"Generated {total} intervals to download")

    # Load previous progress for resume capability
    progress = load_download_progress(job_dir) if resume else {"downloaded": [], "failed": [], "no_data": [], "cached": []}
    # Ensure all keys exist for backwards compatibility
    if "no_data" not in progress:
        progress["no_data"] = progress.get("skipped", [])
    if "cached" not in progress:
        progress["cached"] = []
    downloaded_dates = set(progress["downloaded"])

    self.update_state(
        state='PROGRESS',
        meta={
            'current': 0,
            'total': total,
            'status': f'Starting {interval} download ({total} intervals)...',
            'downloaded': len(progress["downloaded"]),
            'failed': len(progress["failed"]),
            'no_data': len(progress["no_data"]),
            'cached': len(progress["cached"])
        }
    )

    images_paths = []

    logger.info(f"=== STARTING SEQUENTIAL DOWNLOAD ===")

    for idx, time_interval in enumerate(intervals):
        date_key = f"{time_interval[0]}_{time_interval[1]}"
        filename = f"image_{date_key}.npy"
        save_path = job_dir / filename

        # Check global cache first
        global_cache_path = global_cache_dir / filename

        # Skip if already downloaded (cache hit - check both job dir and global cache)
        if date_key in downloaded_dates or save_path.exists() or global_cache_path.exists():
            # Copy from global cache to job dir if needed
            if global_cache_path.exists() and not save_path.exists():
                shutil.copy(global_cache_path, save_path)
            if save_path.exists():
                images_paths.append(str(save_path))
            progress["cached"].append(date_key)
            logger.info(f"[{idx+1}/{total}] {date_key} -> CACHED")
            continue

        # Download
        logger.info(f"[{idx+1}/{total}] {date_key} -> Downloading...")
        try:
            image = fetch_satellite_image(bbox, time_interval, config)

            if image is not None and not np.all(image == 0):
                np.save(save_path, image)
                np.save(global_cache_path, image)  # Save to global cache too
                images_paths.append(str(save_path))
                progress["downloaded"].append(date_key)
                logger.info(f"[{idx+1}/{total}] {date_key} -> DOWNLOADED (shape: {image.shape})")
            else:
                progress["no_data"].append(date_key)
                logger.info(f"[{idx+1}/{total}] {date_key} -> NO DATA")

        except Exception as e:
            logger.error(f"[{idx+1}/{total}] {date_key} -> ERROR: {e}")
            progress["failed"].append(date_key)

        # Save progress after each image
        save_download_progress(job_dir, progress)

        # Update UI
        self.update_state(
            state='PROGRESS',
            meta={
                'current': idx + 1,
                'total': total,
                'status': f'Processing {date_key}',
                'downloaded': len(progress["downloaded"]),
                'failed': len(progress["failed"]),
                'no_data': len(progress["no_data"]),
                'cached': len(progress["cached"])
            }
        )

        # Small delay to avoid rate limiting
        time.sleep(1)

    # Also load any previously downloaded images not in current paths
    for existing_file in job_dir.glob("image_*.npy"):
        if str(existing_file) not in images_paths:
            images_paths.append(str(existing_file))

    # Sort paths by date for consistent ordering
    images_paths.sort()

    # Fetch DEM data (only once - terrain doesn't change)
    dem_path = job_dir / "dem.npy"
    dem_data = None

    if not dem_path.exists():
        self.update_state(
            state='PROGRESS',
            meta={
                'current': total,
                'total': total,
                'status': 'Downloading DEM data...',
                'downloaded': len(progress["downloaded"]),
                'failed': len(progress["failed"]),
                'no_data': len(progress["no_data"]),
                'cached': len(progress["cached"])
            }
        )
        dem_data = fetch_dem_data(bbox, config, save_path=dem_path)

        if dem_data is not None:
            slope = calculate_slope(dem_data)
            twi = calculate_twi(dem_data)
            np.save(job_dir / "slope.npy", slope)
            np.save(job_dir / "twi.npy", twi)
    else:
        dem_data = np.load(dem_path)

    # Save metadata
    save_metadata(
        job_id=job_id,
        bbox_coords={
            'min_lon': min_lon, 'min_lat': min_lat,
            'max_lon': max_lon, 'max_lat': max_lat
        },
        num_images=len(images_paths),
        time_range=(intervals[0][0] if intervals else "", intervals[-1][1] if intervals else "")
    )

    # Save image paths list
    with open(job_dir / "images.json", "w") as f:
        json.dump(images_paths, f)

    logger.info(f"=== DOWNLOAD TASK COMPLETE ===")
    logger.info(f"Downloaded: {len(progress['downloaded'])}")
    logger.info(f"Cached: {len(progress['cached'])}")
    logger.info(f"No Data: {len(progress['no_data'])}")
    logger.info(f"Failed: {len(progress['failed'])}")
    logger.info(f"Total images available: {len(images_paths)}")
    logger.info(f"DEM available: {dem_data is not None}")

    return {
        'job_id': job_id,
        'downloaded': len(progress['downloaded']),
        'failed': len(progress['failed']),
        'no_data': len(progress['no_data']),
        'cached': len(progress['cached']),
        'total': total,
        'total_images': len(images_paths),
        'dem_available': dem_data is not None,
        'status': 'completed'
    }


@celery_app.task(bind=True)
def train_model_task(self, job_id: str, epochs: int = 50):
    """Train flood risk model on downloaded images.

    Args:
        job_id: Job identifier with downloaded images
        epochs: Number of training epochs
    """
    job_dir = IMAGES_DIR / job_id
    model_dir = MODELS_DIR / job_id
    model_dir.mkdir(parents=True, exist_ok=True)

    # Load image paths
    images_file = job_dir / "images.json"
    if not images_file.exists():
        return {'status': 'error', 'message': 'No images found for this job'}

    with open(images_file) as f:
        image_paths = json.load(f)

    if len(image_paths) < 5:
        return {'status': 'error', 'message': 'Not enough images for training (need at least 5)'}

    self.update_state(
        state='PROGRESS',
        meta={'current': 0, 'total': 100, 'status': 'Loading images...'}
    )

    # Load images
    images = []
    for path in image_paths:
        try:
            img = np.load(path)
            images.append(img)
        except Exception as e:
            print(f"Error loading {path}: {e}")

    if len(images) < 5:
        return {'status': 'error', 'message': 'Not enough valid images loaded'}

    self.update_state(
        state='PROGRESS',
        meta={'current': 5, 'total': 100, 'status': 'Loading DEM data...'}
    )

    # Load DEM data if available
    dem_data = None
    slope_data = None
    twi_data = None

    dem_path = job_dir / "dem.npy"
    slope_path = job_dir / "slope.npy"
    twi_path = job_dir / "twi.npy"

    if dem_path.exists() and slope_path.exists() and twi_path.exists():
        try:
            dem_data = np.load(dem_path)
            slope_data = np.load(slope_path)
            twi_data = np.load(twi_path)
        except Exception as e:
            print(f"Error loading DEM data: {e}")

    self.update_state(
        state='PROGRESS',
        meta={'current': 7, 'total': 100, 'status': 'Computing spectral indices...'}
    )

    # Calculate and save NDVI/NDWI time series for visualization
    ndvi_stack = []
    ndwi_stack = []
    for img in images:
        ndvi_stack.append(calculate_ndvi(img))
        ndwi_stack.append(calculate_ndwi(img))

    ndvi_stack = np.array(ndvi_stack)
    ndwi_stack = np.array(ndwi_stack)

    # Save mean NDVI and NDWI for visualization
    mean_ndvi = np.mean(ndvi_stack, axis=0)
    mean_ndwi = np.mean(ndwi_stack, axis=0)
    np.save(model_dir / "mean_ndvi.npy", mean_ndvi)
    np.save(model_dir / "mean_ndwi.npy", mean_ndwi)

    # Also save temporal stats
    ndvi_std = np.std(ndvi_stack, axis=0)
    ndwi_std = np.std(ndwi_stack, axis=0)
    np.save(model_dir / "std_ndvi.npy", ndvi_std)
    np.save(model_dir / "std_ndwi.npy", ndwi_std)

    self.update_state(
        state='PROGRESS',
        meta={'current': 10, 'total': 100, 'status': 'Preparing training data...'}
    )

    # Prepare training data with DEM features
    X, y, has_dem = prepare_training_data(
        images,
        dem=dem_data,
        slope=slope_data,
        twi=twi_data
    )
    num_dem_channels = 3 if has_dem else 0

    self.update_state(
        state='PROGRESS',
        meta={'current': 20, 'total': 100, 'status': 'Training model...'}
    )

    # Training progress callback
    def progress_callback(epoch, total_epochs, loss):
        progress = 20 + int(70 * epoch / total_epochs)
        self.update_state(
            state='PROGRESS',
            meta={
                'current': progress,
                'total': 100,
                'status': f'Training epoch {epoch}/{total_epochs} (loss: {loss:.4f})'
            }
        )

    # Train model
    model_path = model_dir / "flood_model.pt"
    model, metrics = train_model(
        X, y,
        epochs=epochs,
        save_path=model_path,
        progress_callback=progress_callback,
        num_dem_channels=num_dem_channels
    )

    # Save metrics separately for easy access
    metrics_path = model_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f)

    self.update_state(
        state='PROGRESS',
        meta={'current': 95, 'total': 100, 'status': 'Generating predictions...'}
    )

    # Generate final predictions
    predictions = predict_flood_risk(
        model, images,
        dem=dem_data,
        slope=slope_data,
        twi=twi_data
    )

    # Save predictions
    predictions_path = model_dir / "predictions.npy"
    np.save(predictions_path, predictions)

    # Save prediction as image for visualization
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(predictions, cmap='RdYlGn_r', vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, label='Flood Risk Probability')
    ax.set_title('Flood Risk Map')
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(model_dir / "flood_risk_map.png", dpi=150, bbox_inches='tight')
    plt.close()

    return {
        'job_id': job_id,
        'status': 'completed',
        'model_path': str(model_path),
        'predictions_path': str(predictions_path),
        'map_path': str(model_dir / "flood_risk_map.png"),
        'metrics': {
            'final_loss': metrics['final_loss'],
            'best_loss': metrics['best_loss'],
            'epochs_trained': metrics['epochs_trained'],
            'num_time_steps': metrics['num_time_steps'],
            'dem_features_used': has_dem,
            'num_dem_channels': num_dem_channels,
        }
    }


@celery_app.task
def get_job_status(job_id: str) -> dict:
    """Get the status of a job."""
    job_dir = IMAGES_DIR / job_id
    model_dir = MODELS_DIR / job_id

    status = {
        'job_id': job_id,
        'images_downloaded': False,
        'model_trained': False,
        'num_images': 0
    }

    # Check if images exist
    images_file = job_dir / "images.json"
    if images_file.exists():
        with open(images_file) as f:
            images = json.load(f)
        status['images_downloaded'] = True
        status['num_images'] = len(images)

    # Check if model exists
    model_file = model_dir / "flood_model.pt"
    if model_file.exists():
        status['model_trained'] = True

    # Check if predictions exist
    predictions_file = model_dir / "predictions.npy"
    map_file = model_dir / "flood_risk_map.png"
    if predictions_file.exists():
        status['predictions_available'] = True
    if map_file.exists():
        status['map_available'] = True
        status['map_path'] = str(map_file)

    return status
