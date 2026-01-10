"""DEM (Digital Elevation Model) Test Page.

This page allows testing the DEM data fetching functionality
and all derived terrain metrics for flood risk validation.
"""
import sys
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import folium
from streamlit_folium import st_folium
import numpy as np

from app.config import get_sentinel_config
from app.utils.satellite import (
    get_bbox_from_coords,
    fetch_dem_data,
    fetch_satellite_image,
    calculate_slope,
    calculate_twi,
    calculate_distance_to_water,
    calculate_ndwi,
    calculate_composite_flood_susceptibility,
    get_adjusted_resolution
)

# Page configuration
st.set_page_config(
    page_title="DEM Test - Flood Risk Detector",
    page_icon="🏔️",
    layout="wide"
)

# Preset locations
PRESET_LOCATIONS = {
    "Custom (Draw on map)": None,
    "Valencia - Paiporta (DANA 2024)": {
        "center": [39.43, -0.42],
        "bbox": {"min_lon": -0.50, "min_lat": 39.38, "max_lon": -0.35, "max_lat": 39.48},
    },
    "Thessaly - Larissa (Storm Daniel 2023)": {
        "center": [39.55, 22.30],
        "bbox": {"min_lon": 22.20, "min_lat": 39.48, "max_lon": 22.40, "max_lat": 39.62},
    },
    "Piura, Peru (Flood-prone)": {
        "center": [-5.20, -80.63],
        "bbox": {"min_lon": -80.72, "min_lat": -5.28, "max_lon": -80.55, "max_lat": -5.12},
    },
}

# Initialize session state
if 'dem_bbox' not in st.session_state:
    st.session_state.dem_bbox = None
if 'dem_data' not in st.session_state:
    st.session_state.dem_data = None
if 'slope_data' not in st.session_state:
    st.session_state.slope_data = None
if 'twi_data' not in st.session_state:
    st.session_state.twi_data = None
if 'ndwi_data' not in st.session_state:
    st.session_state.ndwi_data = None
if 'distance_water' not in st.session_state:
    st.session_state.distance_water = None
if 'composite_susceptibility' not in st.session_state:
    st.session_state.composite_susceptibility = None
if 'resolution' not in st.session_state:
    st.session_state.resolution = 10.0


def main():
    st.title("Terrain & Water Analysis")
    st.markdown("""
    Test all proxy validation data sources for flood risk assessment:
    **DEM**, **Slope**, **TWI**, **NDWI**, and **Distance to Water**.
    """)

    # Area Selection
    st.header("1. Select Area")

    selected_preset = st.selectbox(
        "Choose a preset location or draw custom:",
        list(PRESET_LOCATIONS.keys())
    )

    preset_data = PRESET_LOCATIONS[selected_preset]
    if preset_data:
        center = preset_data["center"]
        zoom = 11
        if st.button("Use This Location"):
            st.session_state.dem_bbox = preset_data["bbox"]
            # Clear previous data
            st.session_state.dem_data = None
            st.session_state.slope_data = None
            st.session_state.twi_data = None
            st.session_state.ndwi_data = None
            st.session_state.distance_water = None
            st.session_state.composite_susceptibility = None
            st.rerun()
    else:
        center = [39.43, -0.42]
        zoom = 10

    # Create map
    m = folium.Map(location=center, zoom_start=zoom)

    if preset_data:
        bbox = preset_data["bbox"]
        folium.Rectangle(
            bounds=[[bbox['min_lat'], bbox['min_lon']], [bbox['max_lat'], bbox['max_lon']]],
            color='blue',
            fill=True,
            fill_opacity=0.2,
            weight=2
        ).add_to(m)

    if st.session_state.dem_bbox:
        bbox = st.session_state.dem_bbox
        folium.Rectangle(
            bounds=[[bbox['min_lat'], bbox['min_lon']], [bbox['max_lat'], bbox['max_lon']]],
            color='green',
            fill=True,
            fill_opacity=0.3,
            weight=3
        ).add_to(m)

    from folium.plugins import Draw
    draw = Draw(
        draw_options={
            'polyline': False, 'polygon': False, 'circle': False,
            'marker': False, 'circlemarker': False, 'rectangle': True
        },
        edit_options={'edit': False}
    )
    draw.add_to(m)

    output = st_folium(m, width=700, height=350, returned_objects=["all_drawings"])

    if output and output.get("all_drawings"):
        drawings = output["all_drawings"]
        if drawings:
            last_drawing = drawings[-1]
            if last_drawing.get("geometry", {}).get("type") == "Polygon":
                coords = last_drawing["geometry"]["coordinates"][0]
                st.session_state.dem_bbox = {
                    'min_lon': min(c[0] for c in coords),
                    'min_lat': min(c[1] for c in coords),
                    'max_lon': max(c[0] for c in coords),
                    'max_lat': max(c[1] for c in coords)
                }
                # Clear previous data
                st.session_state.dem_data = None
                st.session_state.slope_data = None
                st.session_state.twi_data = None
                st.session_state.ndwi_data = None
                st.session_state.distance_water = None
                st.session_state.composite_susceptibility = None

    if st.session_state.dem_bbox:
        bbox = st.session_state.dem_bbox
        st.success(f"Area: ({bbox['min_lat']:.4f}, {bbox['min_lon']:.4f}) to ({bbox['max_lat']:.4f}, {bbox['max_lon']:.4f})")

    st.divider()

    # Fetch Data
    st.header("2. Fetch Data")

    if st.session_state.dem_bbox is None:
        st.warning("Please select an area first.")
    else:
        col1, col2 = st.columns(2)

        with col1:
            if st.button("Fetch DEM + Terrain Data", type="primary"):
                with st.spinner("Fetching DEM data..."):
                    try:
                        bbox = st.session_state.dem_bbox
                        sh_bbox = get_bbox_from_coords(
                            bbox['min_lon'], bbox['min_lat'],
                            bbox['max_lon'], bbox['max_lat']
                        )
                        config = get_sentinel_config()
                        resolution = get_adjusted_resolution(sh_bbox)
                        st.session_state.resolution = resolution

                        dem_data = fetch_dem_data(sh_bbox, config)

                        if dem_data is not None:
                            st.session_state.dem_data = dem_data
                            with st.spinner("Calculating slope..."):
                                st.session_state.slope_data = calculate_slope(dem_data, resolution)
                            with st.spinner("Calculating TWI (this may take a moment)..."):
                                st.session_state.twi_data = calculate_twi(dem_data, resolution)
                            st.success("DEM + Terrain data ready!")
                            st.rerun()
                        else:
                            st.error("Failed to fetch DEM data.")
                    except Exception as e:
                        st.error(f"Error: {e}")

        with col2:
            if st.button("Fetch Satellite + Water Data", type="secondary"):
                with st.spinner("Fetching satellite image..."):
                    try:
                        bbox = st.session_state.dem_bbox
                        sh_bbox = get_bbox_from_coords(
                            bbox['min_lon'], bbox['min_lat'],
                            bbox['max_lon'], bbox['max_lat']
                        )
                        config = get_sentinel_config()
                        resolution = st.session_state.resolution

                        # Get recent image for NDWI
                        from datetime import datetime, timedelta
                        end_date = datetime.now()
                        start_date = end_date - timedelta(days=30)
                        time_interval = (
                            start_date.strftime("%Y-%m-%d"),
                            end_date.strftime("%Y-%m-%d")
                        )

                        image = fetch_satellite_image(sh_bbox, time_interval, config)

                        if image is not None:
                            ndwi = calculate_ndwi(image)
                            st.session_state.ndwi_data = ndwi
                            with st.spinner("Calculating distance to water..."):
                                st.session_state.distance_water = calculate_distance_to_water(
                                    ndwi, water_threshold=0.3, resolution=resolution
                                )
                            st.success("Satellite + Water data ready!")
                            st.rerun()
                        else:
                            st.error("Failed to fetch satellite image.")
                    except Exception as e:
                        st.error(f"Error: {e}")

        # Calculate composite if all data available
        if (st.session_state.dem_data is not None and
            st.session_state.slope_data is not None and
            st.session_state.twi_data is not None and
            st.session_state.distance_water is not None):

            if st.session_state.composite_susceptibility is None:
                st.session_state.composite_susceptibility = calculate_composite_flood_susceptibility(
                    st.session_state.dem_data,
                    st.session_state.slope_data,
                    st.session_state.twi_data,
                    st.session_state.distance_water
                )
                st.rerun()

    st.divider()

    # Results
    st.header("3. Results")

    # Check what data we have
    has_dem = st.session_state.dem_data is not None
    has_water = st.session_state.ndwi_data is not None
    has_composite = st.session_state.composite_susceptibility is not None

    if not has_dem and not has_water:
        st.info("Fetch data to see results.")
        return

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # Summary Statistics
    st.subheader("Summary Statistics")

    if has_dem:
        dem = st.session_state.dem_data
        slope = st.session_state.slope_data
        twi = st.session_state.twi_data

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Elevation Range", f"{np.nanmin(dem):.0f} - {np.nanmax(dem):.0f} m")
        with col2:
            st.metric("Mean Elevation", f"{np.nanmean(dem):.1f} m")
        with col3:
            st.metric("Mean Slope", f"{np.nanmean(slope):.1f}°")
        with col4:
            flat_pct = np.sum(slope < 5) / slope.size * 100
            st.metric("Flat Areas (<5°)", f"{flat_pct:.1f}%")

        if twi is not None:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("TWI Range", f"{np.nanmin(twi):.1f} - {np.nanmax(twi):.1f}")
            with col2:
                st.metric("Mean TWI", f"{np.nanmean(twi):.2f}")
            with col3:
                high_twi = np.sum(twi > np.nanpercentile(twi, 75)) / twi.size * 100
                st.metric("High TWI Areas", f"{high_twi:.1f}%")
            with col4:
                st.metric("Image Size", f"{dem.shape[0]} x {dem.shape[1]}")

    if has_water:
        ndwi = st.session_state.ndwi_data
        dist = st.session_state.distance_water

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            water_pct = np.sum(ndwi > 0.3) / ndwi.size * 100
            st.metric("Water Coverage", f"{water_pct:.1f}%")
        with col2:
            st.metric("Mean NDWI", f"{np.nanmean(ndwi):.3f}")
        with col3:
            near_water = np.sum(dist < 500) / dist.size * 100
            st.metric("Near Water (<500m)", f"{near_water:.1f}%")
        with col4:
            st.metric("Max Distance", f"{np.nanmax(dist):.0f} m")

    st.divider()

    # Visualizations
    st.subheader("Terrain Analysis")

    if has_dem:
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**Elevation (DEM)**")
            fig, ax = plt.subplots(figsize=(6, 5))
            im = ax.imshow(st.session_state.dem_data, cmap='terrain')
            plt.colorbar(im, ax=ax, label='Meters', shrink=0.8)
            ax.axis('off')
            st.pyplot(fig)
            plt.close()

        with col2:
            st.markdown("**Slope**")
            fig, ax = plt.subplots(figsize=(6, 5))
            im = ax.imshow(st.session_state.slope_data, cmap='YlOrRd', vmin=0, vmax=30)
            plt.colorbar(im, ax=ax, label='Degrees', shrink=0.8)
            ax.axis('off')
            st.pyplot(fig)
            plt.close()

        with col3:
            st.markdown("**TWI (Wetness Index)**")
            fig, ax = plt.subplots(figsize=(6, 5))
            twi = st.session_state.twi_data
            im = ax.imshow(twi, cmap='Blues', vmin=np.nanpercentile(twi, 5), vmax=np.nanpercentile(twi, 95))
            plt.colorbar(im, ax=ax, label='TWI', shrink=0.8)
            ax.axis('off')
            st.pyplot(fig)
            plt.close()

    st.divider()

    st.subheader("Water Analysis")

    if has_water:
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**NDWI (Water Index)**")
            fig, ax = plt.subplots(figsize=(6, 5))
            im = ax.imshow(st.session_state.ndwi_data, cmap='RdYlBu', vmin=-0.5, vmax=0.5)
            plt.colorbar(im, ax=ax, label='NDWI', shrink=0.8)
            ax.axis('off')
            st.pyplot(fig)
            plt.close()

        with col2:
            st.markdown("**Water Bodies**")
            fig, ax = plt.subplots(figsize=(6, 5))
            water_mask = st.session_state.ndwi_data > 0.3
            ax.imshow(water_mask, cmap='Blues')
            ax.axis('off')
            st.pyplot(fig)
            plt.close()

        with col3:
            st.markdown("**Distance to Water**")
            fig, ax = plt.subplots(figsize=(6, 5))
            dist = st.session_state.distance_water
            im = ax.imshow(dist, cmap='YlOrRd_r', vmax=np.nanpercentile(dist, 95))
            plt.colorbar(im, ax=ax, label='Meters', shrink=0.8)
            ax.axis('off')
            st.pyplot(fig)
            plt.close()
    else:
        st.info("Fetch satellite data to see water analysis.")

    st.divider()

    # Composite Flood Susceptibility
    st.subheader("Composite Flood Susceptibility (Proxy Validation)")

    if has_composite:
        st.markdown("""
        **Combined analysis using weighted factors:**
        - Elevation (25%) - lower = higher risk
        - Slope (25%) - flatter = higher risk
        - TWI (30%) - higher = higher risk
        - Distance to water (20%) - closer = higher risk
        """)

        col1, col2 = st.columns([2, 1])

        with col1:
            fig, ax = plt.subplots(figsize=(10, 8))
            im = ax.imshow(st.session_state.composite_susceptibility, cmap='RdYlGn_r', vmin=0, vmax=1)
            plt.colorbar(im, ax=ax, label='Flood Susceptibility', shrink=0.8)
            ax.set_title('Composite Flood Susceptibility Map')
            ax.axis('off')
            st.pyplot(fig)
            plt.close()

        with col2:
            composite = st.session_state.composite_susceptibility
            high_risk = np.sum(composite > 0.6) / composite.size * 100
            medium_risk = np.sum((composite > 0.3) & (composite <= 0.6)) / composite.size * 100
            low_risk = np.sum(composite <= 0.3) / composite.size * 100

            st.markdown("**Risk Distribution**")
            st.metric("High Risk (>0.6)", f"{high_risk:.1f}%")
            st.metric("Medium Risk (0.3-0.6)", f"{medium_risk:.1f}%")
            st.metric("Low Risk (<0.3)", f"{low_risk:.1f}%")

            st.divider()

            st.markdown("**Correlation with Elevation**")
            # Calculate correlation between composite and elevation
            corr = np.corrcoef(
                st.session_state.dem_data.flatten(),
                composite.flatten()
            )[0, 1]
            st.metric("Elevation Correlation", f"{corr:.3f}")
            st.caption("Negative = high risk in low areas")

        # Histogram comparison
        st.markdown("**Elevation Distribution by Risk Level**")
        fig, ax = plt.subplots(figsize=(10, 4))

        dem = st.session_state.dem_data
        composite = st.session_state.composite_susceptibility

        high_mask = composite > 0.6
        low_mask = composite <= 0.3

        if high_mask.any():
            ax.hist(dem[high_mask].flatten(), bins=30, alpha=0.7, label='High Risk', color='red')
        if low_mask.any():
            ax.hist(dem[low_mask].flatten(), bins=30, alpha=0.7, label='Low Risk', color='green')

        ax.set_xlabel('Elevation (m)')
        ax.set_ylabel('Pixel Count')
        ax.legend()
        ax.set_title('Elevation Distribution: High Risk vs Low Risk Areas')
        st.pyplot(fig)
        plt.close()

        st.success("All proxy validation data ready! This can be used to validate model predictions.")

    elif has_dem and not has_water:
        st.warning("Fetch satellite data to calculate composite susceptibility.")
    elif has_water and not has_dem:
        st.warning("Fetch DEM data to calculate composite susceptibility.")


if __name__ == "__main__":
    main()
