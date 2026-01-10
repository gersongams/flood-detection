"""Streamlit application for flood risk detection."""
import sys
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import folium
from streamlit_folium import st_folium
import uuid
import json
import numpy as np
import datetime as dt

from celery.result import AsyncResult

# Import after path setup
from app.celery_app import celery_app
from app.tasks import download_images_task, train_model_task
from app.config import IMAGES_DIR, MODELS_DIR

# Default study area (user can draw their own)
DEFAULT_AREA = {
    "name": "Piura, Peru",
    "center": [-5.23, -80.63],
    "bbox": {"min_lon": -80.68, "min_lat": -5.28, "max_lon": -80.58, "max_lat": -5.18},
}

# Page configuration
st.set_page_config(
    page_title="Flood Risk Detector",
    page_icon="🌊",
    layout="wide"
)

# Initialize session state
if 'job_id' not in st.session_state:
    st.session_state.job_id = None
if 'download_task_id' not in st.session_state:
    st.session_state.download_task_id = None
if 'train_task_id' not in st.session_state:
    st.session_state.train_task_id = None
if 'bbox' not in st.session_state:
    st.session_state.bbox = None
if 'step' not in st.session_state:
    st.session_state.step = 1


def main():
    st.title("Flood Risk Detector")
    st.markdown("Analyze satellite imagery to identify flood-prone areas using deep learning.")

    # Sidebar for configuration
    with st.sidebar:
        st.header("Configuration")

        # Date range options
        use_custom_dates = st.checkbox("Use custom date range", value=True)

        if use_custom_dates:
            col1, col2 = st.columns(2)
            with col1:
                start_date = st.date_input(
                    "Start date",
                    value=dt.datetime(2018, 1, 1),
                    min_value=dt.datetime(2015, 1, 1),
                    max_value=dt.datetime.now()
                )
            with col2:
                end_date = st.date_input(
                    "End date",
                    value=dt.datetime(2025, 12, 31),
                    min_value=dt.datetime(2015, 1, 1),
                    max_value=dt.datetime.now()
                )
            years_back = None
        else:
            years_back = st.slider("Years of historical data", 1, 8, 5)
            start_date = None
            end_date = None

        # Interval selection
        interval = st.selectbox(
            "Download interval",
            ["weekly", "monthly", "daily"],
            index=0,
            help="Weekly recommended for training datasets"
        )

        epochs = st.slider("Training epochs", 10, 100, 50)

        st.divider()

        st.header("Current Job")
        if st.session_state.job_id:
            st.info(f"Job ID: `{st.session_state.job_id[:8]}...`")
        else:
            st.warning("No active job")

        if st.button("Start New Job"):
            st.session_state.job_id = None
            st.session_state.download_task_id = None
            st.session_state.train_task_id = None
            st.session_state.bbox = None
            st.session_state.step = 1
            st.rerun()

    # Step indicator
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if st.session_state.step >= 1:
            st.success("1. Select Area")
        else:
            st.info("1. Select Area")
    with col2:
        if st.session_state.step >= 2:
            st.success("2. Download Images")
        elif st.session_state.step == 1:
            st.info("2. Download Images")
    with col3:
        if st.session_state.step >= 3:
            st.success("3. Train Model")
        elif st.session_state.step <= 2:
            st.info("3. Train Model")
    with col4:
        if st.session_state.step >= 4:
            st.success("4. View Results")
        else:
            st.info("4. View Results")

    st.divider()

    # Step 1: Area Selection - Draw on map
    if st.session_state.step == 1:
        st.header("Step 1: Select Study Area")
        st.markdown("**Draw a rectangle** on the map to select your area of interest.")

        # Create map with draw controls
        m = folium.Map(location=DEFAULT_AREA["center"], zoom_start=11)

        # Show default area as reference
        bbox = DEFAULT_AREA["bbox"]
        folium.Rectangle(
            bounds=[[bbox['min_lat'], bbox['min_lon']], [bbox['max_lat'], bbox['max_lon']]],
            color='gray',
            fill=False,
            weight=1,
            dash_array='5',
            popup="Default area (reference)"
        ).add_to(m)

        # Add draw control for rectangle
        from folium.plugins import Draw
        draw = Draw(
            draw_options={
                'polyline': False,
                'polygon': False,
                'circle': False,
                'marker': False,
                'circlemarker': False,
                'rectangle': True
            },
            edit_options={'edit': False}
        )
        draw.add_to(m)

        # Display map
        output = st_folium(m, width=800, height=500, returned_objects=["all_drawings"])

        # Check if rectangle was drawn
        if output and output.get("all_drawings"):
            drawings = output["all_drawings"]
            if drawings:
                last_drawing = drawings[-1]
                if last_drawing.get("geometry", {}).get("type") == "Polygon":
                    coords = last_drawing["geometry"]["coordinates"][0]

                    min_lon = min(c[0] for c in coords)
                    max_lon = max(c[0] for c in coords)
                    min_lat = min(c[1] for c in coords)
                    max_lat = max(c[1] for c in coords)

                    st.session_state.bbox = {
                        'min_lon': min_lon,
                        'min_lat': min_lat,
                        'max_lon': max_lon,
                        'max_lat': max_lat
                    }

        # Show selected area
        if st.session_state.bbox:
            b = st.session_state.bbox
            st.success(f"**Selected:** ({b['min_lat']:.4f}, {b['min_lon']:.4f}) to ({b['max_lat']:.4f}, {b['max_lon']:.4f})")

            if st.button("Continue to Download", type="primary"):
                st.session_state.job_id = str(uuid.uuid4())
                st.session_state.step = 2
                st.rerun()
        else:
            st.warning("Draw a rectangle on the map to select an area.")

    # Step 2: Download Images
    elif st.session_state.step == 2:
        st.header("Step 2: Download Satellite Images")

        bbox = st.session_state.bbox
        st.info(f"Area: ({bbox['min_lat']:.4f}, {bbox['min_lon']:.4f}) to ({bbox['max_lat']:.4f}, {bbox['max_lon']:.4f})")

        # Show download configuration
        if use_custom_dates:
            date_info = f"**Date range:** {start_date} to {end_date} ({interval})"
        else:
            date_info = f"**Period:** Last {years_back} years ({interval})"
        st.markdown(date_info)

        if st.session_state.download_task_id is None:
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Start Download", type="primary"):
                    # Start the download task
                    task = download_images_task.delay(
                        job_id=st.session_state.job_id,
                        min_lon=bbox['min_lon'],
                        min_lat=bbox['min_lat'],
                        max_lon=bbox['max_lon'],
                        max_lat=bbox['max_lat'],
                        start_date=start_date.strftime("%Y-%m-%d") if start_date else None,
                        end_date=end_date.strftime("%Y-%m-%d") if end_date else None,
                        interval=interval,
                        years_back=years_back if years_back else 5,
                        resume=True
                    )
                    st.session_state.download_task_id = task.id
                    st.rerun()
            with col2:
                st.caption("Downloads will resume if interrupted. Cached images are skipped.")
        else:
            # Check task status
            task_result = AsyncResult(st.session_state.download_task_id, app=celery_app)

            if task_result.state == 'PENDING':
                st.warning("Task is pending... Waiting for worker.")
                st.info("Make sure the Celery worker is running: `make worker`")

            elif task_result.state == 'PROGRESS':
                meta = task_result.info
                progress_val = meta.get('current', 0) / meta.get('total', 1)
                st.progress(progress_val)
                st.info(meta.get('status', 'Processing...'))
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Downloaded", meta.get('downloaded', 0), help="New images saved")
                with col2:
                    st.metric("Cached", meta.get('cached', 0), help="Already had these")
                with col3:
                    st.metric("No Data", meta.get('no_data', 0), help="Cloudy or no satellite pass")
                with col4:
                    st.metric("Failed", meta.get('failed', 0), help="Errors")

            elif task_result.state == 'SUCCESS':
                result = task_result.result
                st.success(f"Download complete! {result.get('total_images', result['downloaded'])} images available.")

                if st.button("Continue to Training", type="primary"):
                    st.session_state.step = 3
                    st.rerun()

            elif task_result.state == 'FAILURE':
                st.error(f"Task failed: {task_result.info}")
                if st.button("Retry"):
                    st.session_state.download_task_id = None
                    st.rerun()

            # Auto-refresh
            if task_result.state in ['PENDING', 'PROGRESS']:
                import time
                time.sleep(2)
                st.rerun()

    # Step 3: Train Model
    elif st.session_state.step == 3:
        st.header("Step 3: Train Flood Risk Model")

        # Show download summary
        images_file = IMAGES_DIR / st.session_state.job_id / "images.json"
        if images_file.exists():
            with open(images_file) as f:
                images = json.load(f)
            st.info(f"Training with {len(images)} satellite images")

        if st.session_state.train_task_id is None:
            st.markdown("The model will learn patterns from historical satellite data to predict flood risk zones.")

            if st.button("Start Training", type="primary"):
                task = train_model_task.delay(
                    job_id=st.session_state.job_id,
                    epochs=epochs
                )
                st.session_state.train_task_id = task.id
                st.rerun()
        else:
            # Check task status
            task_result = AsyncResult(st.session_state.train_task_id, app=celery_app)

            if task_result.state == 'PENDING':
                st.warning("Task is pending... Waiting for worker.")

            elif task_result.state == 'PROGRESS':
                meta = task_result.info
                progress = meta.get('current', 0) / meta.get('total', 100)
                st.progress(progress)
                st.info(meta.get('status', 'Training...'))

            elif task_result.state == 'SUCCESS':
                result = task_result.result
                if result.get('status') == 'completed':
                    st.success("Model training complete!")

                    # Show training metrics
                    if result.get('metrics'):
                        metrics = result['metrics']
                        st.subheader("Training Metrics")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Final Loss", f"{metrics.get('final_loss', 0):.4f}")
                        with col2:
                            st.metric("Best Loss", f"{metrics.get('best_loss', 0):.4f}")
                        with col3:
                            st.metric("Epochs", metrics.get('epochs_trained', 0))

                        st.caption(f"Trained on {metrics.get('num_time_steps', 0)} time steps of satellite data")

                    if st.button("View Results", type="primary"):
                        st.session_state.step = 4
                        st.rerun()
                else:
                    st.error(result.get('message', 'Unknown error'))

            elif task_result.state == 'FAILURE':
                st.error(f"Training failed: {task_result.info}")
                if st.button("Retry"):
                    st.session_state.train_task_id = None
                    st.rerun()

            # Auto-refresh
            if task_result.state in ['PENDING', 'PROGRESS']:
                import time
                time.sleep(2)
                st.rerun()

    # Step 4: Results
    elif st.session_state.step == 4:
        st.header("Step 4: Flood Risk Analysis Results")

        model_dir = MODELS_DIR / st.session_state.job_id

        # Load and display training metrics
        metrics_path = model_dir / "metrics.json"
        if metrics_path.exists():
            with open(metrics_path) as f:
                metrics = json.load(f)

            st.subheader("Model Performance")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Final Loss", f"{metrics.get('final_loss', 0):.4f}")
            with col2:
                st.metric("Best Loss", f"{metrics.get('best_loss', 0):.4f}")
            with col3:
                st.metric("Epochs", metrics.get('epochs_trained', 0))
            with col4:
                st.metric("Time Steps", metrics.get('num_time_steps', 0))

            # Loss curve chart
            if metrics.get('losses'):
                import matplotlib
                matplotlib.use('Agg')
                import matplotlib.pyplot as plt

                fig, ax = plt.subplots(figsize=(8, 3))
                ax.plot(range(1, len(metrics['losses']) + 1), metrics['losses'], 'b-', linewidth=1.5)
                ax.set_xlabel('Epoch')
                ax.set_ylabel('Loss')
                ax.set_title('Training Loss Curve')
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
                plt.close()

            st.divider()

        # Load and display results
        map_path = model_dir / "flood_risk_map.png"
        predictions_path = model_dir / "predictions.npy"

        if map_path.exists():
            col1, col2 = st.columns([2, 1])

            with col1:
                st.subheader("Flood Risk Map (Model Prediction)")
                st.image(str(map_path), width="stretch")

            with col2:
                st.subheader("Legend")
                st.markdown("""
                **Risk Levels:**
                - **Low Risk** (0.0 - 0.3): Safe areas
                - **Medium Risk** (0.3 - 0.6): Attention needed
                - **High Risk** (0.6 - 1.0): Flood-prone
                """)

                if predictions_path.exists():
                    predictions = np.load(predictions_path)

                    high_risk = np.sum(predictions > 0.6) / predictions.size * 100
                    medium_risk = np.sum((predictions > 0.3) & (predictions <= 0.6)) / predictions.size * 100
                    low_risk = np.sum(predictions <= 0.3) / predictions.size * 100

                    st.subheader("Risk Distribution")
                    st.metric("High Risk Area", f"{high_risk:.1f}%")
                    st.metric("Medium Risk Area", f"{medium_risk:.1f}%")
                    st.metric("Low Risk Area", f"{low_risk:.1f}%")

        else:
            st.error("Results not found. Please complete the training step first.")
            if st.button("Back to Training"):
                st.session_state.step = 3
                st.rerun()
            return

        # Acquired Data Layers Section
        st.divider()
        st.header("Acquired Data Layers")
        st.markdown("All spectral and terrain data used in the analysis.")

        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        # Load NDVI and NDWI from training
        mean_ndvi_path = model_dir / "mean_ndvi.npy"
        mean_ndwi_path = model_dir / "mean_ndwi.npy"

        has_spectral = mean_ndvi_path.exists() and mean_ndwi_path.exists()

        if has_spectral:
            mean_ndvi = np.load(mean_ndvi_path)
            mean_ndwi = np.load(mean_ndwi_path)

            # Also load std if available
            std_ndvi_path = model_dir / "std_ndvi.npy"
            std_ndwi_path = model_dir / "std_ndwi.npy"
            has_std = std_ndvi_path.exists() and std_ndwi_path.exists()

            st.subheader("Spectral Indices (from Satellite Imagery)")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown("**Mean NDVI (Vegetation)**")
                fig, ax = plt.subplots(figsize=(6, 5))
                im = ax.imshow(mean_ndvi, cmap='RdYlGn', vmin=-0.2, vmax=0.8)
                plt.colorbar(im, ax=ax, label='NDVI', shrink=0.8)
                ax.axis('off')
                st.pyplot(fig)
                plt.close()

                # Stats
                st.caption(f"Range: {np.nanmin(mean_ndvi):.2f} to {np.nanmax(mean_ndvi):.2f}")
                st.caption(f"Mean: {np.nanmean(mean_ndvi):.3f}")

            with col2:
                st.markdown("**Mean NDWI (Water)**")
                fig, ax = plt.subplots(figsize=(6, 5))
                im = ax.imshow(mean_ndwi, cmap='RdYlBu', vmin=-0.5, vmax=0.5)
                plt.colorbar(im, ax=ax, label='NDWI', shrink=0.8)
                ax.axis('off')
                st.pyplot(fig)
                plt.close()

                # Stats
                water_pct = np.sum(mean_ndwi > 0.3) / mean_ndwi.size * 100
                st.caption(f"Water coverage: {water_pct:.1f}%")
                st.caption(f"Mean: {np.nanmean(mean_ndwi):.3f}")

            with col3:
                st.markdown("**Water Bodies Detected**")
                fig, ax = plt.subplots(figsize=(6, 5))
                water_mask = mean_ndwi > 0.3
                ax.imshow(water_mask, cmap='Blues')
                ax.axis('off')
                st.pyplot(fig)
                plt.close()

                st.caption(f"Threshold: NDWI > 0.3")
                st.caption(f"Water pixels: {np.sum(water_mask):,}")

            # Temporal variability
            if has_std:
                std_ndvi = np.load(std_ndvi_path)
                std_ndwi = np.load(std_ndwi_path)

                st.markdown("---")
                st.markdown("**Temporal Variability** (areas with high variability may indicate flooding events)")

                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("**NDVI Variability (Std Dev)**")
                    fig, ax = plt.subplots(figsize=(6, 5))
                    im = ax.imshow(std_ndvi, cmap='YlOrRd', vmin=0, vmax=np.nanpercentile(std_ndvi, 95))
                    plt.colorbar(im, ax=ax, label='Std Dev', shrink=0.8)
                    ax.axis('off')
                    st.pyplot(fig)
                    plt.close()

                with col2:
                    st.markdown("**NDWI Variability (Std Dev)**")
                    fig, ax = plt.subplots(figsize=(6, 5))
                    im = ax.imshow(std_ndwi, cmap='YlOrRd', vmin=0, vmax=np.nanpercentile(std_ndwi, 95))
                    plt.colorbar(im, ax=ax, label='Std Dev', shrink=0.8)
                    ax.axis('off')
                    st.pyplot(fig)
                    plt.close()

        else:
            st.info("Spectral data will be available after training. Re-run training to generate NDVI/NDWI visualizations.")

        # Show terrain data if available
        if st.session_state.get('proxy_dem') is not None:
            st.divider()
            st.subheader("Terrain Data (from DEM)")

            dem = st.session_state.proxy_dem
            slope = st.session_state.proxy_slope
            twi = st.session_state.proxy_twi
            distance = st.session_state.proxy_distance

            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown("**Elevation (DEM)**")
                fig, ax = plt.subplots(figsize=(6, 5))
                im = ax.imshow(dem, cmap='terrain')
                plt.colorbar(im, ax=ax, label='Meters', shrink=0.8)
                ax.axis('off')
                st.pyplot(fig)
                plt.close()
                st.caption(f"Range: {np.nanmin(dem):.0f}m - {np.nanmax(dem):.0f}m")

            with col2:
                st.markdown("**Slope**")
                fig, ax = plt.subplots(figsize=(6, 5))
                im = ax.imshow(slope, cmap='YlOrRd', vmin=0, vmax=30)
                plt.colorbar(im, ax=ax, label='Degrees', shrink=0.8)
                ax.axis('off')
                st.pyplot(fig)
                plt.close()
                flat_pct = np.sum(slope < 5) / slope.size * 100
                st.caption(f"Flat areas (<5°): {flat_pct:.1f}%")

            with col3:
                st.markdown("**TWI (Wetness Index)**")
                fig, ax = plt.subplots(figsize=(6, 5))
                im = ax.imshow(twi, cmap='Blues', vmin=np.nanpercentile(twi, 5), vmax=np.nanpercentile(twi, 95))
                plt.colorbar(im, ax=ax, label='TWI', shrink=0.8)
                ax.axis('off')
                st.pyplot(fig)
                plt.close()
                st.caption(f"Mean TWI: {np.nanmean(twi):.2f}")

            if distance is not None:
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.markdown("**Distance to Water**")
                    fig, ax = plt.subplots(figsize=(6, 5))
                    im = ax.imshow(distance, cmap='YlOrRd_r', vmax=np.nanpercentile(distance, 95))
                    plt.colorbar(im, ax=ax, label='Meters', shrink=0.8)
                    ax.axis('off')
                    st.pyplot(fig)
                    plt.close()
                    near_water = np.sum(distance < 500) / distance.size * 100
                    st.caption(f"Near water (<500m): {near_water:.1f}%")

                with col2:
                    # Composite susceptibility
                    from app.utils.satellite import calculate_composite_flood_susceptibility
                    composite = calculate_composite_flood_susceptibility(dem, slope, twi, distance)
                    st.markdown("**Composite Susceptibility**")
                    fig, ax = plt.subplots(figsize=(6, 5))
                    im = ax.imshow(composite, cmap='RdYlGn_r', vmin=0, vmax=1)
                    plt.colorbar(im, ax=ax, label='Susceptibility', shrink=0.8)
                    ax.axis('off')
                    st.pyplot(fig)
                    plt.close()
                    st.caption("Combined terrain-based risk")

                with col3:
                    st.markdown("**Model Prediction**")
                    predictions = np.load(predictions_path)
                    fig, ax = plt.subplots(figsize=(6, 5))
                    im = ax.imshow(predictions, cmap='RdYlGn_r', vmin=0, vmax=1)
                    plt.colorbar(im, ax=ax, label='Risk', shrink=0.8)
                    ax.axis('off')
                    st.pyplot(fig)
                    plt.close()
                    st.caption("CNN-based prediction")

        # Proxy Validation Section
        st.divider()
        st.header("Proxy Validation Analysis")
        st.markdown("""
        Validate model predictions against topographic and hydrological factors.
        This provides quantitative evidence that predictions correlate with known flood risk indicators.
        """)

        # Initialize proxy data in session state
        if 'proxy_dem' not in st.session_state:
            st.session_state.proxy_dem = None
        if 'proxy_slope' not in st.session_state:
            st.session_state.proxy_slope = None
        if 'proxy_twi' not in st.session_state:
            st.session_state.proxy_twi = None
        if 'proxy_distance' not in st.session_state:
            st.session_state.proxy_distance = None

        # Auto-fetch proxy data (no button needed)
        if st.session_state.proxy_dem is None:
            with st.spinner("Fetching DEM and calculating terrain metrics..."):
                try:
                    from app.utils.satellite import (
                        get_bbox_from_coords, fetch_dem_data, calculate_slope,
                        calculate_twi, calculate_distance_to_water, calculate_ndwi,
                        get_adjusted_resolution, fetch_satellite_image
                    )
                    from app.config import get_sentinel_config
                    from datetime import datetime, timedelta

                    bbox = st.session_state.bbox
                    sh_bbox = get_bbox_from_coords(
                        bbox['min_lon'], bbox['min_lat'],
                        bbox['max_lon'], bbox['max_lat']
                    )
                    config = get_sentinel_config()
                    resolution = get_adjusted_resolution(sh_bbox)

                    # Fetch DEM
                    dem_data = fetch_dem_data(sh_bbox, config)
                    if dem_data is not None:
                        st.session_state.proxy_dem = dem_data
                        st.session_state.proxy_slope = calculate_slope(dem_data, resolution)
                        st.session_state.proxy_twi = calculate_twi(dem_data, resolution)

                        # Fetch recent satellite image for distance to water
                        end_date = datetime.now()
                        start_date = end_date - timedelta(days=30)
                        time_interval = (start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))

                        image = fetch_satellite_image(sh_bbox, time_interval, config)
                        if image is not None:
                            ndwi = calculate_ndwi(image)
                            st.session_state.proxy_distance = calculate_distance_to_water(
                                ndwi, water_threshold=0.3, resolution=resolution
                            )

                        st.success("Terrain data loaded!")
                        st.rerun()
                    else:
                        st.error("Failed to fetch DEM data.")
                except Exception as e:
                    st.error(f"Error: {e}")

        # Display proxy validation results
        if st.session_state.proxy_dem is not None and predictions_path.exists():
            predictions = np.load(predictions_path)
            dem = st.session_state.proxy_dem
            slope = st.session_state.proxy_slope
            twi = st.session_state.proxy_twi
            distance = st.session_state.proxy_distance

            # Resize predictions to match DEM if needed
            if predictions.shape != dem.shape:
                from scipy.ndimage import zoom
                zoom_factors = (dem.shape[0] / predictions.shape[0], dem.shape[1] / predictions.shape[1])
                predictions_resized = zoom(predictions, zoom_factors, order=1)
            else:
                predictions_resized = predictions

            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt

            # Correlation Analysis
            st.subheader("Correlation Analysis")
            st.markdown("Pearson correlation between model predictions and terrain factors:")

            col1, col2, col3, col4 = st.columns(4)

            corr_elev = np.corrcoef(predictions_resized.flatten(), dem.flatten())[0, 1]
            corr_slope = np.corrcoef(predictions_resized.flatten(), slope.flatten())[0, 1]
            corr_twi = np.corrcoef(predictions_resized.flatten(), twi.flatten())[0, 1]

            with col1:
                st.metric("vs Elevation", f"{corr_elev:.3f}")
                st.caption("Negative = risk in low areas")
            with col2:
                st.metric("vs Slope", f"{corr_slope:.3f}")
                st.caption("Negative = risk in flat areas")
            with col3:
                st.metric("vs TWI", f"{corr_twi:.3f}")
                st.caption("Positive = risk where water accumulates")

            if distance is not None:
                corr_dist = np.corrcoef(predictions_resized.flatten(), distance.flatten())[0, 1]
                with col4:
                    st.metric("vs Water Distance", f"{corr_dist:.3f}")
                    st.caption("Negative = risk near water")

            st.divider()

            # Visual Comparison
            st.subheader("Visual Comparison: Model vs Terrain Factors")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown("**Model Prediction**")
                fig, ax = plt.subplots(figsize=(6, 5))
                im = ax.imshow(predictions_resized, cmap='RdYlGn_r', vmin=0, vmax=1)
                plt.colorbar(im, ax=ax, label='Risk', shrink=0.8)
                ax.axis('off')
                st.pyplot(fig)
                plt.close()

            with col2:
                st.markdown("**Elevation (DEM)**")
                fig, ax = plt.subplots(figsize=(6, 5))
                im = ax.imshow(dem, cmap='terrain')
                plt.colorbar(im, ax=ax, label='Meters', shrink=0.8)
                ax.axis('off')
                st.pyplot(fig)
                plt.close()

            with col3:
                st.markdown("**TWI (Wetness Index)**")
                fig, ax = plt.subplots(figsize=(6, 5))
                im = ax.imshow(twi, cmap='Blues', vmin=np.nanpercentile(twi, 5), vmax=np.nanpercentile(twi, 95))
                plt.colorbar(im, ax=ax, label='TWI', shrink=0.8)
                ax.axis('off')
                st.pyplot(fig)
                plt.close()

            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown("**Slope**")
                fig, ax = plt.subplots(figsize=(6, 5))
                im = ax.imshow(slope, cmap='YlOrRd', vmin=0, vmax=30)
                plt.colorbar(im, ax=ax, label='Degrees', shrink=0.8)
                ax.axis('off')
                st.pyplot(fig)
                plt.close()

            with col2:
                if distance is not None:
                    st.markdown("**Distance to Water**")
                    fig, ax = plt.subplots(figsize=(6, 5))
                    im = ax.imshow(distance, cmap='YlOrRd_r', vmax=np.nanpercentile(distance, 95))
                    plt.colorbar(im, ax=ax, label='Meters', shrink=0.8)
                    ax.axis('off')
                    st.pyplot(fig)
                    plt.close()

            with col3:
                # Composite proxy susceptibility
                from app.utils.satellite import calculate_composite_flood_susceptibility
                if distance is not None:
                    composite = calculate_composite_flood_susceptibility(dem, slope, twi, distance)
                    st.markdown("**Proxy Susceptibility**")
                    fig, ax = plt.subplots(figsize=(6, 5))
                    im = ax.imshow(composite, cmap='RdYlGn_r', vmin=0, vmax=1)
                    plt.colorbar(im, ax=ax, label='Susceptibility', shrink=0.8)
                    ax.axis('off')
                    st.pyplot(fig)
                    plt.close()

            st.divider()

            # Statistical Analysis
            st.subheader("Statistical Analysis by Risk Level")

            high_mask = predictions_resized > 0.6
            low_mask = predictions_resized <= 0.3

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Mean Values by Risk Level**")

                stats_data = {
                    "Factor": ["Elevation (m)", "Slope (°)", "TWI"],
                    "High Risk": [
                        f"{np.nanmean(dem[high_mask]):.1f}" if high_mask.any() else "N/A",
                        f"{np.nanmean(slope[high_mask]):.1f}" if high_mask.any() else "N/A",
                        f"{np.nanmean(twi[high_mask]):.2f}" if high_mask.any() else "N/A",
                    ],
                    "Low Risk": [
                        f"{np.nanmean(dem[low_mask]):.1f}" if low_mask.any() else "N/A",
                        f"{np.nanmean(slope[low_mask]):.1f}" if low_mask.any() else "N/A",
                        f"{np.nanmean(twi[low_mask]):.2f}" if low_mask.any() else "N/A",
                    ],
                }

                if distance is not None:
                    stats_data["Factor"].append("Distance to Water (m)")
                    stats_data["High Risk"].append(f"{np.nanmean(distance[high_mask]):.0f}" if high_mask.any() else "N/A")
                    stats_data["Low Risk"].append(f"{np.nanmean(distance[low_mask]):.0f}" if low_mask.any() else "N/A")

                import pandas as pd
                st.dataframe(pd.DataFrame(stats_data), hide_index=True)

            with col2:
                st.markdown("**Elevation Distribution by Risk**")
                fig, ax = plt.subplots(figsize=(8, 4))

                if high_mask.any():
                    ax.hist(dem[high_mask].flatten(), bins=30, alpha=0.7, label='High Risk', color='red')
                if low_mask.any():
                    ax.hist(dem[low_mask].flatten(), bins=30, alpha=0.7, label='Low Risk', color='green')

                ax.set_xlabel('Elevation (m)')
                ax.set_ylabel('Pixel Count')
                ax.legend()
                st.pyplot(fig)
                plt.close()

            # Validation Summary
            st.divider()
            st.subheader("Validation Summary")

            validation_passed = []
            if corr_elev < -0.1:
                validation_passed.append("Elevation: High-risk areas at lower elevations")
            if corr_slope < -0.1:
                validation_passed.append("Slope: High-risk areas in flatter terrain")
            if corr_twi > 0.1:
                validation_passed.append("TWI: High-risk areas where water accumulates")
            if distance is not None and corr_dist < -0.1:
                validation_passed.append("Distance: High-risk areas closer to water bodies")

            if len(validation_passed) >= 2:
                st.success(f"Model predictions align with {len(validation_passed)}/4 terrain-based flood indicators:")
                for v in validation_passed:
                    st.markdown(f"- {v}")
            elif len(validation_passed) == 1:
                st.warning("Model predictions partially align with terrain factors:")
                for v in validation_passed:
                    st.markdown(f"- {v}")
            else:
                st.error("Model predictions do not strongly correlate with terrain-based flood indicators.")

        st.divider()

        # Interactive Map
        st.subheader("Interactive Map")

        if st.session_state.bbox and predictions_path.exists():
            bbox = st.session_state.bbox
            predictions = np.load(predictions_path)

            center_lat = (bbox['min_lat'] + bbox['max_lat']) / 2
            center_lon = (bbox['min_lon'] + bbox['max_lon']) / 2

            m = folium.Map(location=[center_lat, center_lon], zoom_start=12)

            folium.Rectangle(
                bounds=[[bbox['min_lat'], bbox['min_lon']], [bbox['max_lat'], bbox['max_lon']]],
                color='blue',
                fill=False,
                weight=2
            ).add_to(m)

            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            from io import BytesIO
            import base64

            cmap = plt.cm.RdYlGn_r
            rgba_image = cmap(predictions)
            rgba_image[:, :, 3] = 0.6

            fig, ax = plt.subplots(figsize=(10, 10))
            ax.imshow(rgba_image)
            ax.axis('off')
            buf = BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, transparent=True)
            plt.close()
            buf.seek(0)
            img_data = base64.b64encode(buf.read()).decode()

            folium.raster_layers.ImageOverlay(
                image=f"data:image/png;base64,{img_data}",
                bounds=[[bbox['min_lat'], bbox['min_lon']], [bbox['max_lat'], bbox['max_lon']]],
                opacity=0.6
            ).add_to(m)

            st_folium(m, width=800, height=500)


if __name__ == "__main__":
    main()
