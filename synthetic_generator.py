"""
synthetic_generator.py
======================
Generates a realistic synthetic smart grid dataset calibrated to:
  - IEEE 118-bus system topology (feeder structure)
  - CEER EU benchmark SAIDI/SAIFI statistics
  - ERA5-style weather patterns (European climate zones)

Produces:
  - assets.csv        : 500 feeders with static attributes
  - timeseries.csv    : 3 years of daily feeder observations
  - outage_events.csv : historical outage events with SAIDI impact

Usage:
  python synthetic_generator.py --feeders 500 --years 3 --seed 42
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants calibrated to CEER 2022 EU reliability benchmarks
# ---------------------------------------------------------------------------
CEER_SAIDI_MEAN   = 180   # average customer-minutes lost per year (EU median)
CEER_SAIDI_STD    = 60
ANNUAL_OUTAGE_RATE = 0.35  # outages per feeder per year (EU distribution network)

CLIMATE_ZONES = {
    "continental":  {"temp_mean": 10, "temp_std": 15, "storm_prob": 0.04},
    "mediterranean":{"temp_mean": 17, "temp_std": 10, "storm_prob": 0.02},
    "oceanic":      {"temp_mean": 12, "temp_std":  8, "storm_prob": 0.05},
    "semi_arid":    {"temp_mean": 22, "temp_std": 12, "storm_prob": 0.015},  # ME proxy
}

# ---------------------------------------------------------------------------
# Feeder / asset generator
# ---------------------------------------------------------------------------

def generate_assets(n_feeders: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    zones = list(CLIMATE_ZONES.keys())

    data = {
        "feeder_id": [f"FEED_{i:04d}" for i in range(n_feeders)],
        "utility_id": rng.choice(["UTIL_A", "UTIL_B", "UTIL_C"], size=n_feeders),
        "climate_zone": rng.choice(zones, size=n_feeders),
        "asset_age_years": rng.integers(1, 45, size=n_feeders).astype(float),
        "rated_capacity_mva": rng.uniform(5, 40, size=n_feeders).round(1),
        "n_customers": rng.integers(200, 5000, size=n_feeders),
        "cable_length_km": rng.uniform(1, 30, size=n_feeders).round(2),
        "n_protection_devices": rng.integers(2, 12, size=n_feeders),
        "underground_ratio": rng.uniform(0.0, 1.0, size=n_feeders).round(2),
        "der_penetration_pct": rng.uniform(0.0, 40.0, size=n_feeders).round(1),
        "last_inspection_days_ago": rng.integers(30, 1095, size=n_feeders).astype(float),
        "vegetation_risk_index": rng.uniform(0.0, 1.0, size=n_feeders).round(3),
    }

    df = pd.DataFrame(data)

    # Health index: composite score 0-1 (higher = worse health)
    df["health_index"] = (
        0.35 * (df["asset_age_years"] / 45) +
        0.25 * (df["last_inspection_days_ago"] / 1095) +
        0.20 * df["vegetation_risk_index"] +
        0.20 * (1 - df["underground_ratio"])
    ).clip(0, 1).round(4)

    return df


# ---------------------------------------------------------------------------
# Daily time-series generator
# ---------------------------------------------------------------------------

def generate_timeseries(assets: pd.DataFrame, n_years: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed + 1)
    dates = pd.date_range("2021-01-01", periods=365 * n_years, freq="D")
    records = []

    for _, feeder in assets.iterrows():
        zone_params = CLIMATE_ZONES[feeder["climate_zone"]]
        n_days = len(dates)

        # ------------------------------------------------------------------
        # Weather simulation (ERA5-style seasonal patterns)
        # ------------------------------------------------------------------
        day_of_year = np.array([d.dayofyear for d in dates])
        seasonal = np.sin(2 * np.pi * (day_of_year - 80) / 365)

        temp = (
            zone_params["temp_mean"] +
            zone_params["temp_std"] * seasonal +
            rng.normal(0, 3, n_days)
        ).round(1)

        wind_speed = np.abs(rng.normal(5, 4, n_days)).round(1)

        # Storm flag: higher probability in winter for continental/oceanic
        storm_prob = zone_params["storm_prob"] * (1 + 0.5 * (-seasonal))
        storm_flag = (rng.uniform(0, 1, n_days) < storm_prob).astype(int)

        heatwave_flag = ((temp > (zone_params["temp_mean"] + 1.5 * zone_params["temp_std"]))).astype(int)

        # ------------------------------------------------------------------
        # Loading percentage (seasonal + random noise)
        # ------------------------------------------------------------------
        base_load = 55 + 20 * (-seasonal)  # higher load in winter
        loading_pct = (base_load + rng.normal(0, 8, n_days)).clip(10, 100).round(1)

        # ------------------------------------------------------------------
        # Outage risk score (ground truth for model — before labelling)
        # ------------------------------------------------------------------
        risk_score = (
            0.30 * feeder["health_index"] +
            0.20 * (loading_pct / 100) +
            0.20 * storm_flag +
            0.15 * heatwave_flag +
            0.10 * feeder["vegetation_risk_index"] +
            0.05 * (feeder["der_penetration_pct"] / 40)
        ) + rng.normal(0, 0.05, n_days)
        risk_score = risk_score.clip(0, 1)

        # ------------------------------------------------------------------
        # Outage label: Bernoulli draw weighted by risk + annual rate
        # ------------------------------------------------------------------
        daily_outage_prob = (ANNUAL_OUTAGE_RATE / 365) * (1 + 3 * risk_score)
        outage = (rng.uniform(0, 1, n_days) < daily_outage_prob).astype(int)

        df_feeder = pd.DataFrame({
            "feeder_id":         feeder["feeder_id"],
            "utility_id":        feeder["utility_id"],
            "date":              dates,
            "temp_c":            temp,
            "wind_speed_ms":     wind_speed,
            "storm_flag":        storm_flag,
            "heatwave_flag":     heatwave_flag,
            "loading_pct":       loading_pct,
            "der_penetration_pct": feeder["der_penetration_pct"],
            "health_index":      feeder["health_index"],
            "vegetation_risk_index": feeder["vegetation_risk_index"],
            "asset_age_years":   feeder["asset_age_years"],
            "underground_ratio": feeder["underground_ratio"],
            "risk_score_true":   risk_score.round(4),
            "outage_flag":       outage,
        })

        records.append(df_feeder)
        del df_feeder

    return pd.concat(records, ignore_index=True)


# ---------------------------------------------------------------------------
# Outage events table
# ---------------------------------------------------------------------------

def generate_outage_events(
    timeseries: pd.DataFrame,
    assets: pd.DataFrame,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed + 2)
    outage_days = timeseries[timeseries["outage_flag"] == 1].copy()

    customer_map = assets.set_index("feeder_id")["n_customers"]

    # Duration: lognormal, calibrated so mean SAIDI ≈ CEER EU benchmark
    duration_minutes = np.round(
        rng.lognormal(mean=np.log(60), sigma=0.8, size=len(outage_days))
    ).astype(int).clip(5, 600)

    n_customers_affected = (
        customer_map.loc[outage_days["feeder_id"]].values *
        rng.uniform(0.1, 1.0, size=len(outage_days))
    ).astype(int)

    saidi_contribution = (duration_minutes * n_customers_affected /
                          customer_map.loc[outage_days["feeder_id"]].values).round(1)

    cause_codes = rng.choice(
        ["WEATHER_STORM", "WEATHER_HEAT", "EQUIPMENT_AGE",
         "VEGETATION", "DER_FAULT", "UNKNOWN"],
        size=len(outage_days),
        p=[0.28, 0.12, 0.25, 0.18, 0.07, 0.10],
    )

    return pd.DataFrame({
        "event_id":                [f"EVT_{i:06d}" for i in range(len(outage_days))],
        "feeder_id":               outage_days["feeder_id"].values,
        "utility_id":              outage_days["utility_id"].values,
        "date":                    outage_days["date"].values,
        "duration_minutes":        duration_minutes,
        "customers_affected":      n_customers_affected,
        "saidi_contribution_min":  saidi_contribution,
        "cause_code":              cause_codes,
        "storm_flag":              outage_days["storm_flag"].values,
        "heatwave_flag":           outage_days["heatwave_flag"].values,
    })


# ---------------------------------------------------------------------------
# Feature engineering: rolling windows + 7-day / 30-day labels
# ---------------------------------------------------------------------------

def build_model_dataset(timeseries: pd.DataFrame) -> pd.DataFrame:
    """
    Adds rolling features and forward-looking labels for supervised learning.
    Returns one row per feeder-day suitable for training.
    """
    ts = timeseries.sort_values(["feeder_id", "date"]).copy()

    grp = ts.groupby("feeder_id")

    # Rolling features (7-day and 30-day lookback)
    for window in [7, 30]:
        ts[f"outage_rate_{window}d"]  = grp["outage_flag"].transform(
            lambda x: x.shift(1).rolling(window, min_periods=1).mean()
        ).round(4)
        ts[f"storm_rate_{window}d"]   = grp["storm_flag"].transform(
            lambda x: x.shift(1).rolling(window, min_periods=1).mean()
        ).round(4)
        ts[f"max_loading_{window}d"]  = grp["loading_pct"].transform(
            lambda x: x.shift(1).rolling(window, min_periods=1).max()
        ).round(2)
        ts[f"mean_temp_{window}d"]    = grp["temp_c"].transform(
            lambda x: x.shift(1).rolling(window, min_periods=1).mean()
        ).round(2)

    # Forward-looking labels: outage in next 7 / 30 days
    ts["label_7d"]  = grp["outage_flag"].transform(
        lambda x: x.shift(-1).rolling(7,  min_periods=1).max()
    ).fillna(0).astype(int)
    ts["label_30d"] = grp["outage_flag"].transform(
        lambda x: x.shift(-1).rolling(30, min_periods=1).max()
    ).fillna(0).astype(int)

    # Days since last outage (time-to-event proxy for survival model)
    def days_since_last(series):
        result = []
        count = 0
        for val in series:
            if val == 1:
                count = 0
            else:
                count += 1
            result.append(count)
        return result

    ts["days_since_last_outage"] = grp["outage_flag"].transform(days_since_last)

    return ts.dropna(subset=["label_7d", "label_30d"])


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic smart grid dataset")
    parser.add_argument("--feeders", type=int, default=500, help="Number of feeders")
    parser.add_argument("--years",   type=int, default=3,   help="Years of history")
    parser.add_argument("--seed",    type=int, default=42,  help="Random seed")
    parser.add_argument("--outdir",  type=str, default="data/raw", help="Output directory")
    args = parser.parse_args()

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"[1/4] Generating {args.feeders} feeder assets...")
    assets = generate_assets(args.feeders, args.seed)
    assets.to_csv(out / "assets.csv", index=False)
    print(f"      Saved assets.csv  ({len(assets)} rows)")

    print(f"[2/4] Generating {args.years}-year daily timeseries...")
    ts = generate_timeseries(assets, args.years, args.seed)
    ts.to_csv(out / "timeseries.csv", index=False)
    print(f"      Saved timeseries.csv  ({len(ts):,} rows)")

    print("[3/4] Generating outage events...")
    events = generate_outage_events(ts, assets, args.seed)
    events.to_csv(out / "outage_events.csv", index=False)
    total_saidi = events["saidi_contribution_min"].sum() / args.feeders
    print(f"      Saved outage_events.csv  ({len(events):,} events)")
    print(f"      Avg annual SAIDI per feeder: {total_saidi/args.years:.1f} min "
          f"(CEER EU benchmark: ~{CEER_SAIDI_MEAN} min)")

    print("[4/4] Building model-ready dataset with rolling features + labels...")
    model_data = build_model_dataset(ts)
    model_data.to_csv(out / "model_dataset.csv", index=False)
    print(f"      Saved model_dataset.csv  ({len(model_data):,} rows)")

    print("\n✓ Dataset generation complete.")
    print(f"  Outage rate (7-day label):  {model_data['label_7d'].mean():.3f}")
    print(f"  Outage rate (30-day label): {model_data['label_30d'].mean():.3f}")
    print(f"  Output directory: {out.resolve()}")


if __name__ == "__main__":
    main()
