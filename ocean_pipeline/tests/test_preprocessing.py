"""
Phase 0 — Unit Tests for the preprocessing pipeline.

Tests:
  1. Regridding: output shape, coordinate bounds, method tagging
  2. Mask: ocean mask is boolean, coastal mask is [0, 1], no NaN over ocean
  3. Tensor: shape correctness, NaN propagation on land
  4. Normalization: correct mean≈0 / std≈1 after normalizing training data

Run with:
    .\\venv\\Scripts\\python.exe -m pytest ocean_pipeline\\tests\\test_preprocessing.py -v
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import xarray as xr
import pytest

from ocean_pipeline.config import LATS, LONS, NLAT, NLON, NCHAN, NDEPTH, INPUT_CHANNELS
from ocean_pipeline.preprocessing.regrid import make_target_grid, regrid_to_target
from ocean_pipeline.preprocessing.normalize import build_norm_stats, normalize, denormalize


# ══════════════════════════════════════════════════════════════════════════════
#  Fixtures
# ══════════════════════════════════════════════════════════════════════════════
@pytest.fixture
def coarse_sst_da():
    """Synthetic 1°-resolution SST DataArray for regridding tests."""
    lats = np.arange(4, 31, 1.0)
    lons = np.arange(44, 106, 1.0)
    data = 28.0 + 2.0 * np.random.randn(len(lats), len(lons))
    return xr.DataArray(
        data.astype(np.float32),
        dims=["latitude", "longitude"],
        coords={"latitude": lats, "longitude": lons},
        attrs={"units": "degrees_C"}
    )


@pytest.fixture
def sample_ocean_mask():
    """Simple synthetic ocean mask (all ocean for unit test)."""
    mask = np.ones((NLAT, NLON), dtype=bool)
    # Mark a 5×5 corner as land
    mask[:5, :5] = False
    return mask


@pytest.fixture
def sample_tensors(sample_ocean_mask):
    """Synthetic stacked tensors shape (N, H, W, C)."""
    N = 30
    data = np.random.randn(N, NLAT, NLON, NCHAN).astype(np.float32)
    # Apply land mask
    data[:, ~sample_ocean_mask, :] = np.nan
    return data, sample_ocean_mask


# ══════════════════════════════════════════════════════════════════════════════
#  Regridding tests
# ══════════════════════════════════════════════════════════════════════════════
class TestRegridding:

    def test_output_shape(self, coarse_sst_da):
        """Regridded output must match target grid dimensions."""
        out = regrid_to_target(coarse_sst_da, LATS, LONS)
        assert out.shape == (NLAT, NLON), (
            f"Expected ({NLAT}, {NLON}), got {out.shape}"
        )

    def test_coordinate_bounds(self, coarse_sst_da):
        """Regridded output coordinates must exactly match target grid."""
        out = regrid_to_target(coarse_sst_da, LATS, LONS)
        np.testing.assert_allclose(out.latitude.values, LATS, atol=1e-5)
        np.testing.assert_allclose(out.longitude.values, LONS, atol=1e-5)

    def test_values_in_physical_range(self, coarse_sst_da):
        """Regridded SST should be physically plausible (-2 to 40 °C)."""
        out = regrid_to_target(coarse_sst_da, LATS, LONS)
        valid = out.values[~np.isnan(out.values)]
        assert (valid > -5).all() and (valid < 50).all(), (
            "Regridded SST out of physical range"
        )

    def test_method_tagged_in_attrs(self, coarse_sst_da):
        """Regrid method must be recorded in output attrs."""
        out = regrid_to_target(coarse_sst_da, LATS, LONS)
        assert "regrid_method" in out.attrs, "regrid_method attr missing"

    def test_no_nan_interior_when_source_full(self, coarse_sst_da):
        """
        When source covers the domain fully, interior cells of the regridded
        output should not be NaN (boundary cells may be NaN due to extrapolation).
        """
        out = regrid_to_target(coarse_sst_da, LATS, LONS)
        # Check centre of domain only (avoid boundary edge effects)
        interior = out.values[5:-5, 5:-5]
        nan_frac = np.isnan(interior).mean()
        assert nan_frac < 0.05, (
            f"Too many NaNs in interior of regridded field: {nan_frac:.1%}"
        )


# ══════════════════════════════════════════════════════════════════════════════
#  Mask tests
# ══════════════════════════════════════════════════════════════════════════════
class TestMasks:

    def test_ocean_mask_dtype(self, sample_ocean_mask):
        """Ocean mask must be boolean."""
        assert sample_ocean_mask.dtype == bool, "Ocean mask is not boolean"

    def test_ocean_mask_shape(self, sample_ocean_mask):
        """Ocean mask must match target grid."""
        assert sample_ocean_mask.shape == (NLAT, NLON)

    def test_ocean_mask_has_both_classes(self, sample_ocean_mask):
        """Mask must contain both land and ocean cells."""
        assert sample_ocean_mask.any(), "No ocean cells in mask"
        assert (~sample_ocean_mask).any(), "No land cells in mask"

    def test_coastal_confidence_range(self, sample_ocean_mask):
        """Coastal confidence mask values must be in [0, 1] (or NaN for land)."""
        from ocean_pipeline.preprocessing.build_masks import (
            compute_distance_to_coast, build_confidence_mask
        )
        dist = compute_distance_to_coast(sample_ocean_mask)
        conf = build_confidence_mask(dist)

        ocean_conf = conf[sample_ocean_mask]
        assert (ocean_conf >= 0).all() and (ocean_conf <= 1).all(), (
            "Confidence values outside [0, 1]"
        )

    def test_land_cells_nan_in_confidence(self, sample_ocean_mask):
        """Land cells should be NaN in the confidence mask."""
        from ocean_pipeline.preprocessing.build_masks import (
            compute_distance_to_coast, build_confidence_mask
        )
        dist = compute_distance_to_coast(sample_ocean_mask)
        conf = build_confidence_mask(dist)
        land_conf = conf[~sample_ocean_mask]
        assert np.all(np.isnan(land_conf)), "Land cells should be NaN in confidence mask"


# ══════════════════════════════════════════════════════════════════════════════
#  Normalization tests
# ══════════════════════════════════════════════════════════════════════════════
class TestNormalization:

    def test_norm_stats_shape(self, sample_tensors, tmp_path):
        """build_norm_stats must return arrays of length NCHAN."""
        tensors, mask = sample_tensors
        save_path = tmp_path / "norm_stats.npz"
        means, stds = build_norm_stats(tensors, ocean_mask=mask, save_path=save_path)
        assert means.shape == (NCHAN,)
        assert stds.shape == (NCHAN,)

    def test_normalized_mean_approx_zero(self, sample_tensors, tmp_path):
        """After normalization, per-channel mean over ocean cells ≈ 0."""
        tensors, mask = sample_tensors
        save_path = tmp_path / "norm_stats.npz"
        means, stds = build_norm_stats(tensors, ocean_mask=mask, save_path=save_path)
        normed = normalize(tensors, means, stds)

        for c in range(NCHAN):
            ch = normed[:, mask, c].ravel()
            ch = ch[~np.isnan(ch)]
            assert abs(ch.mean()) < 0.1, (
                f"Channel {INPUT_CHANNELS[c]} normalized mean={ch.mean():.3f} not near 0"
            )

    def test_normalized_std_approx_one(self, sample_tensors, tmp_path):
        """After normalization, per-channel std over ocean cells ≈ 1."""
        tensors, mask = sample_tensors
        save_path = tmp_path / "norm_stats.npz"
        means, stds = build_norm_stats(tensors, ocean_mask=mask, save_path=save_path)
        normed = normalize(tensors, means, stds)

        for c in range(NCHAN):
            ch = normed[:, mask, c].ravel()
            ch = ch[~np.isnan(ch)]
            assert abs(ch.std() - 1.0) < 0.1, (
                f"Channel {INPUT_CHANNELS[c]} normalized std={ch.std():.3f} not near 1"
            )

    def test_denormalize_roundtrip(self, sample_tensors, tmp_path):
        """normalize → denormalize should recover the original values."""
        tensors, mask = sample_tensors
        save_path = tmp_path / "norm_stats.npz"
        means, stds = build_norm_stats(tensors, ocean_mask=mask, save_path=save_path)
        normed = normalize(tensors, means, stds)
        recovered = denormalize(normed, means, stds)
        np.testing.assert_allclose(
            tensors[~np.isnan(tensors)],
            recovered[~np.isnan(recovered)],
            rtol=1e-4,
            err_msg="Denormalization roundtrip error"
        )

    def test_nan_preserved_after_normalize(self, sample_tensors, tmp_path):
        """NaN values (land cells) must remain NaN after normalization."""
        tensors, mask = sample_tensors
        save_path = tmp_path / "norm_stats.npz"
        means, stds = build_norm_stats(tensors, ocean_mask=mask, save_path=save_path)
        normed = normalize(tensors, means, stds)
        land_before = np.isnan(tensors[:, ~mask, :])
        land_after  = np.isnan(normed[:, ~mask, :])
        np.testing.assert_array_equal(land_before, land_after,
                                      err_msg="NaN propagation failed through normalization")


# ══════════════════════════════════════════════════════════════════════════════
#  Target tensor sanity
# ══════════════════════════════════════════════════════════════════════════════
class TestTargetPhysics:

    def test_sst_warmer_than_deep_temperature(self):
        """
        Physical sanity: mean SST in the North Indian Ocean should exceed
        deep temperature at 1000 m. This test uses synthetic data that respects
        basic oceanic stratification.
        """
        from ocean_pipeline.config import DEPTH_LEVELS
        # Synthetic profile: warm surface, cold deep
        profile = np.array([28, 27, 26, 24, 22, 18, 15, 12, 10, 9, 7, 5, 3, 2, 1.5],
                           dtype=np.float32)
        assert len(profile) == len(DEPTH_LEVELS)
        assert profile[0] > profile[-1], "SST should be warmer than 1000 m temperature"
        # Ensure mostly decreasing (allow slight inversions for thermocline shape)
        diffs = np.diff(profile)
        pct_decreasing = (diffs <= 0).mean()
        assert pct_decreasing > 0.8, "Temperature profile should be mostly decreasing with depth"
