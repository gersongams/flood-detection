"""Deep learning model for flood risk prediction."""
import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
from torch.optim.lr_scheduler import OneCycleLR
import numpy as np
from pathlib import Path


def get_device() -> torch.device:
    """Get the best available device (CUDA > MPS > CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_amp_device_type(device: torch.device) -> str:
    """Get the device type string for AMP autocast."""
    if device.type == "cuda":
        return "cuda"
    elif device.type == "mps":
        return "cpu"  # MPS doesn't support AMP, fallback to CPU autocast (no-op)
    return "cpu"


class FloodRiskCNN(nn.Module):
    """CNN model for flood risk prediction based on spectral indices time series.

    Input: Stack of NDVI and NDWI images over time + optional DEM features
    Output: Flood risk probability map
    """

    def __init__(
        self,
        num_time_steps: int = 60,
        in_channels: int = 2,
        num_dem_channels: int = 0
    ):
        super().__init__()

        # Number of time steps (months of data)
        self.num_time_steps = num_time_steps
        # DEM channels (elevation, slope, TWI)
        self.num_dem_channels = num_dem_channels
        # 2 channels: NDVI and NDWI per time step + DEM features
        total_channels = in_channels * num_time_steps + num_dem_channels

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(total_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        # Output layer - flood risk probability
        self.output = nn.Sequential(
            nn.Conv2d(32, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, time_steps * 2, height, width)
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return self.output(decoded)


def normalize_array(arr: np.ndarray) -> np.ndarray:
    """Normalize array to 0-1 range."""
    min_val, max_val = np.nanmin(arr), np.nanmax(arr)
    if max_val - min_val == 0:
        return np.zeros_like(arr)
    return (arr - min_val) / (max_val - min_val)


def prepare_training_data(
    images: list[np.ndarray],
    dem: np.ndarray | None = None,
    slope: np.ndarray | None = None,
    twi: np.ndarray | None = None,
    ndvi_threshold: float = 0.2,
    ndwi_threshold: float = 0.3
) -> tuple[np.ndarray, np.ndarray, bool]:
    """Prepare training data from satellite images and DEM features.

    Creates pseudo-labels based on water presence indicators and terrain.

    Args:
        images: List of multi-band satellite images
        dem: Digital Elevation Model (optional)
        slope: Slope derived from DEM (optional)
        twi: Topographic Wetness Index (optional)
        ndvi_threshold: Threshold for low vegetation (potential flood)
        ndwi_threshold: Threshold for water presence

    Returns:
        X: Stacked features (NDVI, NDWI per timestep + DEM features)
        y: Pseudo-labels for flood risk
        has_dem: Whether DEM features were included
    """
    from app.utils.satellite import calculate_ndvi, calculate_ndwi

    ndvi_stack = []
    ndwi_stack = []

    for img in images:
        ndvi = calculate_ndvi(img)
        ndwi = calculate_ndwi(img)
        ndvi_stack.append(ndvi)
        ndwi_stack.append(ndwi)

    # Stack all indices
    ndvi_array = np.stack(ndvi_stack, axis=0)
    ndwi_array = np.stack(ndwi_stack, axis=0)

    # Check if DEM features are available and have matching dimensions
    has_dem = False
    if dem is not None and slope is not None and twi is not None:
        target_shape = ndvi_array.shape[1:]
        if dem.shape == target_shape and slope.shape == target_shape and twi.shape == target_shape:
            has_dem = True

    if has_dem:
        # Normalize DEM features to 0-1 range
        dem_norm = normalize_array(dem)
        slope_norm = normalize_array(slope)
        twi_norm = normalize_array(twi)

        # Create feature array: [NDVI_t1, NDWI_t1, ..., NDVI_tN, NDWI_tN, DEM, Slope, TWI]
        num_time_channels = len(images) * 2
        num_dem_channels = 3
        X = np.empty((num_time_channels + num_dem_channels, *ndvi_array.shape[1:]))

        # Interleave NDVI and NDWI for each time step
        X[0:num_time_channels:2] = ndvi_array
        X[1:num_time_channels:2] = ndwi_array

        # Add DEM features as additional channels
        X[num_time_channels] = dem_norm
        X[num_time_channels + 1] = slope_norm
        X[num_time_channels + 2] = twi_norm
    else:
        # Fallback: only spectral indices
        X = np.empty((len(images) * 2, *ndvi_array.shape[1:]))
        X[0::2] = ndvi_array
        X[1::2] = ndwi_array

    # Create pseudo-labels based on historical water presence, vegetation loss, and terrain
    flood_indicators = np.zeros_like(ndvi_array[0])

    for i in range(len(images)):
        water_mask = ndwi_array[i] > ndwi_threshold
        low_veg_mask = ndvi_array[i] < ndvi_threshold
        flood_indicators += (water_mask & low_veg_mask).astype(float)

    # Normalize spectral-based indicators
    spectral_risk = flood_indicators / len(images)

    if has_dem:
        # Combine spectral risk with terrain susceptibility
        # High TWI = water accumulates, Low elevation = water flows there, Low slope = water stays
        terrain_risk = (
            0.4 * twi_norm +  # TWI is strong indicator
            0.3 * (1 - normalize_array(dem)) +  # Lower elevation = higher risk
            0.3 * (1 - slope_norm)  # Flatter = higher risk
        )

        # Weighted combination: 60% historical patterns, 40% terrain
        y = 0.6 * spectral_risk + 0.4 * terrain_risk
    else:
        y = spectral_risk

    y = np.clip(y, 0, 1)

    return X.astype(np.float32), y.astype(np.float32), has_dem


def train_model(
    X: np.ndarray,
    y: np.ndarray,
    epochs: int = 50,
    batch_size: int = 16,
    learning_rate: float = 0.001,
    save_path: Path | None = None,
    progress_callback=None,
    use_amp: bool = True,
    use_compile: bool = True,
    early_stopping_patience: int = 5,
    num_dem_channels: int = 0,
) -> tuple[FloodRiskCNN, dict]:
    """Train the flood risk prediction model.

    Args:
        X: Training features (stacked NDVI, NDWI, and optionally DEM features)
        y: Training labels (flood risk probability)
        epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate
        save_path: Path to save the trained model
        progress_callback: Callback function for progress updates
        use_amp: Use automatic mixed precision for faster training
        use_compile: Use torch.compile() for model optimization (PyTorch 2.0+)
        early_stopping_patience: Stop training if loss doesn't improve for N epochs
        num_dem_channels: Number of DEM feature channels (0 if no DEM data)

    Returns:
        Tuple of (trained model, training metrics dict)
    """
    device = get_device()
    amp_device_type = get_amp_device_type(device)
    use_amp = use_amp and device.type == "cuda"  # AMP only benefits CUDA

    # Get number of time steps from data (subtract DEM channels, divide by 2 for NDVI+NDWI)
    num_time_steps = (X.shape[0] - num_dem_channels) // 2

    model = FloodRiskCNN(
        num_time_steps=num_time_steps,
        num_dem_channels=num_dem_channels
    ).to(device)

    # Compile model for faster execution (PyTorch 2.0+)
    # Note: torch.compile is unstable on macOS/MPS and can cause SIGABRT
    can_compile = (
        use_compile
        and hasattr(torch, 'compile')
        and device.type == 'cuda'  # Only compile on CUDA, not MPS/CPU
    )
    if can_compile:
        try:
            model = torch.compile(model)
        except Exception:
            pass  # Compilation not supported on this platform

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Prepare data - reshape for batch processing
    # For simplicity, we'll use the entire image as one sample and
    # create patches for training
    patch_size = 128
    patches_X = []
    patches_y = []

    h, w = y.shape
    for i in range(0, h - patch_size + 1, patch_size // 2):
        for j in range(0, w - patch_size + 1, patch_size // 2):
            patch_X = X[:, i:i+patch_size, j:j+patch_size]
            patch_y = y[i:i+patch_size, j:j+patch_size]
            patches_X.append(patch_X)
            patches_y.append(patch_y)

    if not patches_X:
        # If image is too small, use the whole image
        patches_X = [X]
        patches_y = [y]

    X_tensor = torch.FloatTensor(np.stack(patches_X)).to(device)
    y_tensor = torch.FloatTensor(np.stack(patches_y)).unsqueeze(1).to(device)

    num_batches = (len(X_tensor) + batch_size - 1) // batch_size

    # OneCycleLR scheduler for faster convergence
    scheduler = OneCycleLR(
        optimizer,
        max_lr=learning_rate * 10,
        epochs=epochs,
        steps_per_epoch=num_batches,
    )

    # Mixed precision training scaler
    scaler = GradScaler(device.type) if use_amp else None

    # Training metrics
    metrics = {
        'losses': [],
        'num_patches': len(patches_X),
        'num_time_steps': num_time_steps,
        'num_dem_channels': num_dem_channels,
        'device': str(device),
        'use_amp': use_amp,
        'use_compile': use_compile and hasattr(torch, 'compile'),
    }

    # Early stopping state
    best_loss = float('inf')
    epochs_without_improvement = 0
    best_model_state = None

    # Training loop
    model.train()
    for epoch in range(epochs):
        total_loss = 0

        for i in range(0, len(X_tensor), batch_size):
            batch_X = X_tensor[i:i+batch_size]
            batch_y = y_tensor[i:i+batch_size]

            optimizer.zero_grad(set_to_none=True)  # Faster than zero_grad()

            if use_amp and scaler is not None:
                # Mixed precision forward pass
                with autocast(amp_device_type):
                    outputs = model(batch_X)
                    if outputs.shape != batch_y.shape:
                        outputs = nn.functional.interpolate(
                            outputs, size=batch_y.shape[2:], mode='bilinear', align_corners=False
                        )
                    loss = criterion(outputs, batch_y)

                # Mixed precision backward pass
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                # Standard forward pass
                outputs = model(batch_X)
                if outputs.shape != batch_y.shape:
                    outputs = nn.functional.interpolate(
                        outputs, size=batch_y.shape[2:], mode='bilinear', align_corners=False
                    )
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

            scheduler.step()
            total_loss += loss.item()

        avg_loss = total_loss / num_batches
        metrics['losses'].append(avg_loss)

        # Early stopping check
        if avg_loss < best_loss:
            best_loss = avg_loss
            epochs_without_improvement = 0
            best_model_state = model.state_dict().copy()
        else:
            epochs_without_improvement += 1

        if progress_callback:
            progress_callback(epoch + 1, epochs, avg_loss)

        # Early stopping
        if epochs_without_improvement >= early_stopping_patience:
            if best_model_state is not None:
                model.load_state_dict(best_model_state)
            break

    # Calculate final metrics
    metrics['final_loss'] = metrics['losses'][-1] if metrics['losses'] else 0
    metrics['best_loss'] = min(metrics['losses']) if metrics['losses'] else 0
    metrics['epochs_trained'] = len(metrics['losses'])
    metrics['early_stopped'] = epochs_without_improvement >= early_stopping_patience

    # Save model if path provided
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        # Get state dict from compiled model if needed
        state_dict = model._orig_mod.state_dict() if hasattr(model, '_orig_mod') else model.state_dict()
        torch.save({
            'model_state_dict': state_dict,
            'num_time_steps': num_time_steps,
            'num_dem_channels': num_dem_channels,
            'metrics': metrics,
        }, save_path)

    return model, metrics


def predict_flood_risk(
    model: FloodRiskCNN,
    images: list[np.ndarray],
    dem: np.ndarray | None = None,
    slope: np.ndarray | None = None,
    twi: np.ndarray | None = None,
    device: torch.device | None = None
) -> np.ndarray:
    """Predict flood risk from satellite images and optional DEM data.

    Args:
        model: Trained FloodRiskCNN model
        images: List of multi-band satellite images
        dem: Digital Elevation Model (optional, required if model was trained with DEM)
        slope: Slope derived from DEM (optional)
        twi: Topographic Wetness Index (optional)
        device: Device to run inference on (auto-detected if None)

    Returns:
        Flood risk probability map
    """
    from app.utils.satellite import calculate_ndvi, calculate_ndwi

    if device is None:
        device = get_device()

    model = model.to(device)
    model.eval()

    # Prepare input data
    ndvi_stack = []
    ndwi_stack = []

    for img in images:
        ndvi = calculate_ndvi(img)
        ndwi = calculate_ndwi(img)
        ndvi_stack.append(ndvi)
        ndwi_stack.append(ndwi)

    ndvi_array = np.stack(ndvi_stack, axis=0)
    ndwi_array = np.stack(ndwi_stack, axis=0)

    # Check if DEM features should be included
    has_dem = model.num_dem_channels > 0 and dem is not None and slope is not None and twi is not None

    if has_dem:
        # Normalize DEM features
        dem_norm = normalize_array(dem)
        slope_norm = normalize_array(slope)
        twi_norm = normalize_array(twi)

        num_time_channels = len(images) * 2
        X = np.empty((num_time_channels + 3, *ndvi_array.shape[1:]))
        X[0:num_time_channels:2] = ndvi_array
        X[1:num_time_channels:2] = ndwi_array
        X[num_time_channels] = dem_norm
        X[num_time_channels + 1] = slope_norm
        X[num_time_channels + 2] = twi_norm
    else:
        X = np.empty((len(images) * 2, *ndvi_array.shape[1:]))
        X[0::2] = ndvi_array
        X[1::2] = ndwi_array

    # Predict in patches and stitch together
    patch_size = 128
    h, w = X.shape[1], X.shape[2]
    result = np.zeros((h, w))
    counts = np.zeros((h, w))

    with torch.no_grad():
        for i in range(0, h - patch_size + 1, patch_size // 2):
            for j in range(0, w - patch_size + 1, patch_size // 2):
                patch = X[:, i:i+patch_size, j:j+patch_size]
                patch_tensor = torch.FloatTensor(patch).unsqueeze(0).to(device)

                output = model(patch_tensor)
                output = nn.functional.interpolate(
                    output, size=(patch_size, patch_size),
                    mode='bilinear', align_corners=False
                )
                output = output.squeeze().cpu().numpy()

                result[i:i+patch_size, j:j+patch_size] += output
                counts[i:i+patch_size, j:j+patch_size] += 1

    # Average overlapping predictions
    counts[counts == 0] = 1
    result = result / counts

    return result


def load_model(model_path: Path) -> FloodRiskCNN:
    """Load a trained model from disk."""
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    num_dem_channels = checkpoint.get('num_dem_channels', 0)
    model = FloodRiskCNN(
        num_time_steps=checkpoint['num_time_steps'],
        num_dem_channels=num_dem_channels
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    return model
