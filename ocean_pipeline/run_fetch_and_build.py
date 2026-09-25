"""
Master data pipeline: fetch all datasets via CMEMS API and build training tensors.

This is the one-stop script to go from zero data to training-ready tensors.
Run it after setting up CMEMS credentials (one-time):

    copernicusmarine login

Then:
    python ocean_pipeline/run_fetch_and_build.py --years 2015-2023

Steps executed:
    1. Fetch SST, SSH, SSS from CMEMS (yearly files)
    2. Fetch GLORYS T, S (monthly files)
    3. Fetch wind if available (yearly files)
    4. Fetch ARGO validation profiles
    5. Regrid, align, and build daily .npz training tensors
    6. Compute normalization statistics (training years only)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import time


def main():
    parser = argparse.ArgumentParser(
        description="Fetch CMECS data and build training tensors"
    )
    parser.add_argument(
        "--years", type=str, default="2015-2023",
        help="Year range (e.g. '2015-2023' or '2020')"
    )
    parser.add_argument(
        "--fetch-only", action="store_true",
        help="Only download data, don't build tensors"
    )
    parser.add_argument(
        "--build-only", action="store_true",
        help="Only build tensors from already-downloaded data"
    )
    parser.add_argument(
        "--status", action="store_true",
        help="Show download status and exit"
    )
    parser.add_argument(
        "--skip-wind", action="store_true",
        help="Skip wind download (least critical for subsurface T)"
    )
    parser.add_argument(
        "--skip-argo", action="store_true",
        help="Skip ARGO validation download"
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-download / re-build even if files exist"
    )
    args = parser.parse_args()

    # Parse year range
    if "-" in args.years:
        start, end = args.years.split("-")
        years = range(int(start), int(end) + 1)
    else:
        years = range(int(args.years), int(args.years) + 1)

    print("=" * 64)
    print("  Ocean Embedding - Data Pipeline")
    print(f"  Years: {list(years)}")
    print("=" * 64)

    # --- Step 0: Status check ---
    if args.status:
        from ocean_pipeline.ingestion.cmems_fetcher import CMECSFetcher
        fetcher = CMECSFetcher()
        fetcher.status(years)
        return

    t0 = time.time()

    # --- Step 1: Fetch from CMECS ---
    if not args.build_only:
        print("\n" + "=" * 64)
        print("  STEP 1: Fetching data from CMECS API")
        print("=" * 64)

        from ocean_pipeline.ingestion.cmems_fetcher import CMECSFetcher
        fetcher = CMECSFetcher()

        try:
            fetcher.fetch_all(
                years=years,
                skip_wind=args.skip_wind,
                force=args.force,
            )
        except Exception as e:
            print(f"\n  [ERROR] CMECS fetch failed: {e}")
            print("  Make sure you have logged in:")
            print("    copernicusmarine login")
            print("  Or set environment variables:")
            print("    COPERNICUSMARINE_SERVICE_USERNAME")
            print("    COPERNICUSMARINE_SERVICE_PASSWORD")
            if not args.build_only:
                print("\n  Continuing to build tensors from whatever data is available...")

        # --- Step 2: Fetch ARGO validation ---
        if not args.skip_argo:
            print("\n" + "-" * 64)
            print("  STEP 2: Fetching ARGO validation data")
            print("-" * 64)

            try:
                from ocean_pipeline.ingestion.argo_fetcher import fetch_argo_all
                fetch_argo_all(years, force=args.force)
            except Exception as e:
                print(f"  [ARGO] Failed: {e}")
                print("  ARGO can be downloaded manually later.")

    # --- Step 3: Build training tensors ---
    if not args.fetch_only:
        print("\n" + "=" * 64)
        print("  STEP 3: Building aligned training tensors")
        print("=" * 64)

        from ocean_pipeline.preprocessing.build_training_data import TrainingDataBuilder
        builder = TrainingDataBuilder()
        builder.build_all(years=years, force=args.force)

    # Summary
    elapsed = time.time() - t0
    print("\n" + "=" * 64)
    print(f"  PIPELINE COMPLETE  ({elapsed/60:.1f} minutes)")
    print("=" * 64)

    # Show final status
    from ocean_pipeline.ingestion.cmems_fetcher import CMECSFetcher
    CMECSFetcher().status(years)


if __name__ == "__main__":
    main()
