"""
CMEMS (Copernicus Marine) API data fetcher for Ocean Embedding pipeline.

Downloads multi-year time series for the North Indian Ocean (5-30N, 45-105E):
    - SST  : OSTIA L4 daily (analysed_sst)
    - SSH  : DUACS L4 daily (adt, sla, ugos, vgos)
    - SSS  : Multi-obs daily (sos)
    - GLORYS: Reanalysis daily (thetao, so — 3D temperature & salinity targets)
    - Wind : CERSAT blended daily (eastward_wind, northward_wind)

Prerequisites:
    pip install copernicusmarine
    copernicusmarine login       # one-time interactive login (creates ~/.copernicusmarine)

Usage:
    from ocean_pipeline.ingestion.cmems_fetcher import CMECSFetcher
    fetcher = CMECSFetcher()
    fetcher.fetch_sst(year=2020)
    fetcher.fetch_all(years=range(2015, 2024))
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import warnings
from datetime import datetime
from typing import Optional

from ocean_pipeline.config import (
    LAT_MIN, LAT_MAX, LON_MIN, LON_MAX,
    DATA_RAW, DEPTH_LEVELS, TRAIN_YEARS, VAL_YEARS, TEST_YEARS
)


# ---------------------------------------------------------------------------
#  CMEMS Dataset Catalogue — IDs and variable names
# ---------------------------------------------------------------------------
CMEMS_DATASETS = {
    "sst": {
        "dataset_id": "METOFFICE-GLO-SST-L4-REP-OBS-SST",
        "variables": ["analysed_sst"],
        "description": "OSTIA SST L4 reprocessed (0.05deg daily)",
    },
    "ssh": {
        "dataset_id": "c3s_obs-sl_glo_phy-ssh_my_twosat-l4-duacs-0.25deg_P1D",
        "variables": ["adt", "sla", "ugos", "vgos"],
        "description": "DUACS SSH L4 multi-year (0.25deg daily)",
    },
    "sss": {
        "dataset_id": "cmems_obs-mob_glo_phy-sss_my_multi_P1D",
        "variables": ["sos"],
        "description": "Multi-obs SSS (0.125deg daily)",
    },
    "glorys": {
        "dataset_id": "cmems_mod_glo_phy_my_0.083deg_P1D-m",
        "variables": ["thetao", "so"],
        "description": "GLORYS12V4 reanalysis (1/12deg daily, 3D T+S)",
    },
    "wind": {
        "dataset_id": "cmems_obs-wind_glo_phy_my_l4_0.125deg_PT1H",
        "variables": ["eastward_wind", "northward_wind"],
        "description": "CERSAT blended wind L4 (0.125deg)",
    },
}

# Spatial buffer (degrees) around NIO domain for subsetting
SPATIAL_BUFFER = 0.5

# Maximum depth for GLORYS subset (metres)
MAX_DEPTH = 1100.0  # slightly above deepest target level (1000m)


class CMECSFetcher:
    """
    Fetch CMEMS datasets for the Ocean Embedding pipeline.

    Data is downloaded to data/raw/cmems/<variable>/ as NetCDF files,
    one file per year (surface obs) or per month (GLORYS).

    Existing files are skipped unless force=True.
    """

    def __init__(self, output_root: Optional[Path] = None):
        self.output_root = output_root or (DATA_RAW / "cmems")
        self.output_root.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        for key in CMEMS_DATASETS:
            (self.output_root / key).mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    #  Core download via copernicusmarine.subset()
    # ------------------------------------------------------------------
    def _download_subset(
        self,
        dataset_key: str,
        start_date: str,
        end_date: str,
        output_filename: str,
        variables: Optional[list] = None,
        min_depth: Optional[float] = None,
        max_depth: Optional[float] = None,
        force: bool = False,
    ) -> Path:
        """
        Download a spatial+temporal subset from CMEMS.

        Parameters
        ----------
        dataset_key : one of 'sst', 'ssh', 'sss', 'glorys', 'wind'
        start_date  : ISO date string 'YYYY-MM-DD'
        end_date    : ISO date string 'YYYY-MM-DD'
        output_filename : filename for the saved NetCDF
        variables   : override variable list (default: from CMEMS_DATASETS)
        min_depth, max_depth : depth bounds for 3D datasets
        force       : re-download even if file exists

        Returns
        -------
        Path to the downloaded file.
        """
        import os
        import copernicusmarine

        cfg = CMEMS_DATASETS[dataset_key]
        out_dir = self.output_root / dataset_key
        out_path = out_dir / output_filename

        if out_path.exists() and not force:
            print(f"  [CMEMS] SKIP (exists): {out_path.name}")
            return out_path

        vars_to_fetch = variables or cfg["variables"]

        print(f"  [CMEMS] Downloading {dataset_key} "
              f"({start_date} to {end_date}) -> {output_filename}")

        kwargs = dict(
            dataset_id=cfg["dataset_id"],
            variables=vars_to_fetch,
            minimum_longitude=LON_MIN - SPATIAL_BUFFER,
            maximum_longitude=LON_MAX + SPATIAL_BUFFER,
            minimum_latitude=LAT_MIN - SPATIAL_BUFFER,
            maximum_latitude=LAT_MAX + SPATIAL_BUFFER,
            start_datetime=f"{start_date}T00:00:00",
            end_datetime=f"{end_date}T23:59:59",
            output_directory=str(out_dir),
            output_filename=output_filename,
            overwrite=force,
        )

        # Inject credentials if available to avoid interactive prompts
        cmems_user = os.environ.get("COPERNICUSMARINE_SERVICE_USERNAME")
        cmems_pass = os.environ.get("COPERNICUSMARINE_SERVICE_PASSWORD")
        if cmems_user and cmems_pass:
            kwargs["username"] = cmems_user
            kwargs["password"] = cmems_pass

        if min_depth is not None:
            kwargs["minimum_depth"] = min_depth
        if max_depth is not None:
            kwargs["maximum_depth"] = max_depth

        try:
            copernicusmarine.subset(**kwargs)
            print(f"  [CMEMS] OK: {out_path.name} "
                  f"({out_path.stat().st_size / 1e6:.1f} MB)")
        except Exception as e:
            print(f"  [CMEMS] FAILED: {dataset_key} {start_date}-{end_date}: {e}")
            # Try with fewer variables as fallback
            if len(vars_to_fetch) > 1:
                print(f"  [CMEMS] Retrying with primary variable only...")
                kwargs["variables"] = [vars_to_fetch[0]]
                try:
                    copernicusmarine.subset(**kwargs)
                    print(f"  [CMEMS] OK (partial): {out_path.name}")
                except Exception as e2:
                    print(f"  [CMEMS] FAILED again: {e2}")
                    raise

        return out_path

    # ------------------------------------------------------------------
    #  Surface observation fetchers (yearly files)
    # ------------------------------------------------------------------
    def fetch_sst(self, year: int, force: bool = False) -> Path:
        """Download OSTIA SST for one year."""
        return self._download_subset(
            dataset_key="sst",
            start_date=f"{year}-01-01",
            end_date=f"{year}-12-31",
            output_filename=f"sst_{year}.nc",
            force=force,
        )

    def fetch_ssh(self, year: int, force: bool = False) -> Path:
        """Download DUACS SSH (+ ugos/vgos geostrophic currents) for one year."""
        return self._download_subset(
            dataset_key="ssh",
            start_date=f"{year}-01-01",
            end_date=f"{year}-12-31",
            output_filename=f"ssh_{year}.nc",
            force=force,
        )

    def fetch_sss(self, year: int, force: bool = False) -> Path:
        """Download multi-obs SSS for one year."""
        return self._download_subset(
            dataset_key="sss",
            start_date=f"{year}-01-01",
            end_date=f"{year}-12-31",
            output_filename=f"sss_{year}.nc",
            force=force,
        )

    def fetch_wind(self, year: int, force: bool = False) -> Path:
        """Download CERSAT blended wind for one year."""
        return self._download_subset(
            dataset_key="wind",
            start_date=f"{year}-01-01",
            end_date=f"{year}-12-31",
            output_filename=f"wind_{year}.nc",
            force=force,
        )

    # ------------------------------------------------------------------
    #  GLORYS fetcher (monthly files — too large for yearly)
    # ------------------------------------------------------------------
    def fetch_glorys(self, year: int, month: Optional[int] = None,
                     force: bool = False) -> list[Path]:
        """
        Download GLORYS T+S for one year (month-by-month).

        If month is specified, download only that month.
        Returns list of Paths to downloaded files.
        """
        import calendar

        months = [month] if month else range(1, 13)
        paths = []

        for m in months:
            last_day = calendar.monthrange(year, m)[1]
            start = f"{year}-{m:02d}-01"
            end = f"{year}-{m:02d}-{last_day:02d}"
            fname = f"glorys_{year}_{m:02d}.nc"

            try:
                p = self._download_subset(
                    dataset_key="glorys",
                    start_date=start,
                    end_date=end,
                    output_filename=fname,
                    min_depth=0.0,
                    max_depth=MAX_DEPTH,
                    force=force,
                )
                paths.append(p)
            except Exception as e:
                print(f"  [GLORYS] Failed for {year}-{m:02d}: {e}")

        return paths

    # ------------------------------------------------------------------
    #  Fetch all datasets for a range of years
    # ------------------------------------------------------------------
    def fetch_all(self, years: Optional[range] = None,
                  skip_wind: bool = False,
                  force: bool = False) -> dict:
        """
        Download all required datasets for the given years.

        Parameters
        ----------
        years : range of years (default: TRAIN_YEARS + VAL_YEARS + TEST_YEARS)
        skip_wind : skip wind download (winds are least critical for subsurface T)
        force : re-download even if files exist

        Returns
        -------
        dict with keys 'sst', 'ssh', 'sss', 'glorys', 'wind', each
        mapping to a list of downloaded file Paths.
        """
        if years is None:
            all_years = sorted(set(TRAIN_YEARS + VAL_YEARS + TEST_YEARS))
            years = range(min(all_years), max(all_years) + 1)

        results = {"sst": [], "ssh": [], "sss": [], "glorys": [], "wind": []}

        total_years = len(list(years))
        for i, year in enumerate(years):
            print(f"\n{'='*60}")
            print(f"  YEAR {year}  ({i+1}/{total_years})")
            print(f"{'='*60}")

            # Surface observations (yearly files)
            for name, fetcher in [("sst", self.fetch_sst),
                                   ("ssh", self.fetch_ssh),
                                   ("sss", self.fetch_sss)]:
                try:
                    p = fetcher(year, force=force)
                    results[name].append(p)
                except Exception as e:
                    print(f"  [{name.upper()}] FAILED for {year}: {e}")

            # Wind (optional)
            if not skip_wind:
                try:
                    p = self.fetch_wind(year, force=force)
                    results["wind"].append(p)
                except Exception as e:
                    print(f"  [WIND] FAILED for {year}: {e}")

            # GLORYS (monthly — largest download)
            try:
                paths = self.fetch_glorys(year, force=force)
                results["glorys"].extend(paths)
            except Exception as e:
                print(f"  [GLORYS] FAILED for {year}: {e}")

        # Summary
        print(f"\n{'='*60}")
        print("  DOWNLOAD SUMMARY")
        print(f"{'='*60}")
        for key, paths in results.items():
            print(f"  {key:8s}: {len(paths)} files downloaded")

        return results

    # ------------------------------------------------------------------
    #  Status check — what has been downloaded already
    # ------------------------------------------------------------------
    def status(self, years: Optional[range] = None) -> dict:
        """Check which files exist for each dataset and year."""
        if years is None:
            all_years = sorted(set(TRAIN_YEARS + VAL_YEARS + TEST_YEARS))
            years = range(min(all_years), max(all_years) + 1)

        status = {}
        for year in years:
            yr_status = {}
            for key in ["sst", "ssh", "sss"]:
                f = self.output_root / key / f"{key}_{year}.nc"
                yr_status[key] = "OK" if f.exists() else "MISSING"

            # GLORYS monthly
            glorys_count = sum(
                1 for m in range(1, 13)
                if (self.output_root / "glorys" / f"glorys_{year}_{m:02d}.nc").exists()
            )
            yr_status["glorys"] = f"{glorys_count}/12"

            # Wind
            f = self.output_root / "wind" / f"wind_{year}.nc"
            yr_status["wind"] = "OK" if f.exists() else "MISSING"

            status[year] = yr_status

        # Print table
        print(f"\n{'Year':<6} {'SST':<8} {'SSH':<8} {'SSS':<8} {'GLORYS':<10} {'Wind':<8}")
        print("-" * 52)
        for year, st in status.items():
            print(f"{year:<6} {st['sst']:<8} {st['ssh']:<8} {st['sss']:<8} "
                  f"{st['glorys']:<10} {st['wind']:<8}")

        return status


# ---------------------------------------------------------------------------
#  CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Fetch CMECS data for Ocean Embedding")
    parser.add_argument("--years", type=str, default="2015-2023",
                        help="Year range, e.g. '2015-2023' or '2020'")
    parser.add_argument("--dataset", type=str, default="all",
                        choices=["all", "sst", "ssh", "sss", "glorys", "wind"],
                        help="Which dataset to fetch")
    parser.add_argument("--status", action="store_true",
                        help="Only show download status, don't download")
    parser.add_argument("--skip-wind", action="store_true",
                        help="Skip wind downloads")
    parser.add_argument("--force", action="store_true",
                        help="Re-download even if files exist")
    args = parser.parse_args()

    # Parse year range
    if "-" in args.years:
        start, end = args.years.split("-")
        years = range(int(start), int(end) + 1)
    else:
        years = range(int(args.years), int(args.years) + 1)

    fetcher = CMECSFetcher()

    if args.status:
        fetcher.status(years)
    elif args.dataset == "all":
        fetcher.fetch_all(years, skip_wind=args.skip_wind, force=args.force)
    else:
        for year in years:
            if args.dataset == "glorys":
                fetcher.fetch_glorys(year, force=args.force)
            else:
                getattr(fetcher, f"fetch_{args.dataset}")(year, force=args.force)
