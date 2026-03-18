---
name: mobility-trajectories
description: >
  Use this Skill for urban mobility analysis: GPS trajectory processing,
  stop detection, OD matrix construction, and mobility entropy metrics.
tags:
  - urban-science
  - mobility
  - gps
  - trajectories
  - transportation
version: "1.0.0"
authors:
  - name: Rosetta Skills Contributors
    github: "@xjtulyc"
license: "MIT"
platforms:
  - claude-code
  - codex
  - gemini-cli
  - cursor
dependencies:
  python:
    - pandas>=2.0
    - numpy>=1.24
    - scipy>=1.11
    - scikit-learn>=1.3
    - matplotlib>=3.7
    - geopandas>=0.14
last_updated: "2026-03-17"
status: "stable"
---

# Urban Mobility Trajectory Analysis

> **One-line summary**: Process GPS mobility traces to detect activity stops, build origin-destination matrices, measure individual mobility entropy, and model urban travel demand patterns.

---

## When to Use This Skill

- When processing GPS/CDR traces to identify stops and trips
- When building origin-destination (OD) matrices from mobility data
- When measuring individual mobility entropy and radius of gyration
- When detecting home/work locations from trajectory data
- When analyzing commuting patterns and modal split
- When computing flow maps and desire lines

**Trigger keywords**: GPS trajectories, mobility data, origin-destination matrix, stop detection, radius of gyration, mobility entropy, CDR data, travel behavior, commuting, flow map, home detection, individual mobility, urban mobility, transportation

---

## Background & Key Concepts

### Radius of Gyration

$$
r_g = \sqrt{\frac{1}{N}\sum_{i=1}^N (r_i - r_{cm})^2}
$$

where $r_{cm}$ is the center of mass of all visited locations. Measures spatial extent of individual mobility.

### Mobility Entropy (Diversity of Locations)

$$
S = -\sum_{i} p_i \ln p_i
$$

where $p_i$ = fraction of time spent at location $i$. High S = diverse, low S = routine/concentrated.

### Stop Detection

A stop = sequence of GPS points where speed < threshold AND dwell time > minimum duration. Between stops = trips.

---

## Environment Setup

### Install Dependencies

```bash
pip install pandas>=2.0 numpy>=1.24 scipy>=1.11 scikit-learn>=1.3 \
            matplotlib>=3.7 geopandas>=0.14
```

### Verify Installation

```python
import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist

# Haversine distance test
def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Earth radius km
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1; dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))

dist = haversine(40.7128, -74.0060, 40.7589, -73.9851)  # NYC points
print(f"Distance: {dist:.3f} km  (expected ~5.7 km)")
```

---

## Core Workflow

### Step 1: GPS Trajectory Simulation and Stop Detection

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# ------------------------------------------------------------------ #
# Simulate GPS traces for 10 individuals over one week
# ------------------------------------------------------------------ #

def haversine(lat1, lon1, lat2, lon2):
    """Vectorized haversine distance in km."""
    R = 6371.0
    lat1, lon1, lat2, lon2 = (np.radians(x) for x in [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1; dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return 2 * R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))

np.random.seed(42)
city_center = (40.7128, -74.0060)  # NYC

def generate_individual_trace(uid, n_days=7, city_lat=40.7128, city_lon=-74.0060):
    """
    Generate synthetic GPS trace for one individual.
    Activity pattern: home(night) → commute → work(day) → commute → home(evening)
    """
    records = []
    t = datetime(2023, 3, 6, 7, 0, 0)  # Start Monday

    # Fixed locations for this individual
    home_lat = city_lat + np.random.uniform(-0.05, 0.05)
    home_lon = city_lon + np.random.uniform(-0.05, 0.05)
    work_lat = city_lat + np.random.uniform(-0.03, 0.03)
    work_lon = city_lon + np.random.uniform(-0.03, 0.03)
    # Occasional visit locations
    pois = [(city_lat + np.random.uniform(-0.04, 0.04),
             city_lon + np.random.uniform(-0.04, 0.04)) for _ in range(5)]

    for day in range(n_days):
        # Morning: at home, GPS ping every 10min
        for h in range(0, 7):
            for m in range(0, 60, 10):
                records.append({'uid': uid, 'ts': t,
                                'lat': home_lat + np.random.normal(0, 0.0005),
                                'lon': home_lon + np.random.normal(0, 0.0005)})
                t += timedelta(minutes=10)

        # Commute (30 min)
        for step in range(6):
            frac = step / 5
            records.append({'uid': uid, 'ts': t,
                            'lat': home_lat + frac*(work_lat-home_lat) + np.random.normal(0, 0.001),
                            'lon': home_lon + frac*(work_lon-home_lon) + np.random.normal(0, 0.001)})
            t += timedelta(minutes=5)

        # At work (8 hours, weekdays only; walk around on weekends)
        if day < 5:
            for h in range(48):
                records.append({'uid': uid, 'ts': t,
                                'lat': work_lat + np.random.normal(0, 0.0005),
                                'lon': work_lon + np.random.normal(0, 0.0005)})
                t += timedelta(minutes=10)
        else:
            # Weekend: visit 1-2 POIs
            poi = pois[np.random.randint(len(pois))]
            for h in range(48):
                records.append({'uid': uid, 'ts': t,
                                'lat': poi[0] + np.random.normal(0, 0.001),
                                'lon': poi[1] + np.random.normal(0, 0.001)})
                t += timedelta(minutes=10)

        # Commute back
        for step in range(6):
            frac = step / 5
            records.append({'uid': uid, 'ts': t,
                            'lat': work_lat + frac*(home_lat-work_lat) + np.random.normal(0, 0.001),
                            'lon': work_lon + frac*(home_lon-work_lon) + np.random.normal(0, 0.001)})
            t += timedelta(minutes=5)

        # Evening at home
        for h in range(24):
            records.append({'uid': uid, 'ts': t,
                            'lat': home_lat + np.random.normal(0, 0.0005),
                            'lon': home_lon + np.random.normal(0, 0.0005)})
            t += timedelta(minutes=10)

    return records

# Generate traces for 10 individuals
all_records = []
for uid in range(10):
    all_records.extend(generate_individual_trace(uid))

df_gps = pd.DataFrame(all_records)
df_gps['ts'] = pd.to_datetime(df_gps['ts'])
df_gps = df_gps.sort_values(['uid', 'ts']).reset_index(drop=True)
print(f"GPS dataset: {len(df_gps):,} records, {df_gps['uid'].nunique()} individuals")

# ---- Stop Detection -------------------------------------------- #
def detect_stops(trace, speed_threshold_kmh=1.5, min_duration_min=15, radius_m=50):
    """
    Detect activity stops from GPS trace.

    Parameters
    ----------
    trace : DataFrame — sorted by 'ts', with 'lat', 'lon' columns
    speed_threshold_kmh : float — max speed to be considered stationary
    min_duration_min : float — minimum stop duration

    Returns
    -------
    DataFrame with stop records: lat, lon, arrive_ts, leave_ts, duration_min
    """
    trace = trace.copy().reset_index(drop=True)

    # Compute speed between consecutive points
    speeds = [0.0]
    for i in range(1, len(trace)):
        dt = (trace.loc[i,'ts'] - trace.loc[i-1,'ts']).total_seconds() / 3600  # hours
        d  = haversine(trace.loc[i-1,'lat'], trace.loc[i-1,'lon'],
                       trace.loc[i,'lat'],   trace.loc[i,'lon'])
        speeds.append(d/dt if dt > 0 else 0)
    trace['speed_kmh'] = speeds

    # Mark stationary points
    trace['stationary'] = trace['speed_kmh'] < speed_threshold_kmh

    # Cluster consecutive stationary points
    stops = []
    in_stop = False
    stop_pts = []

    for _, row in trace.iterrows():
        if row['stationary']:
            stop_pts.append(row)
            in_stop = True
        else:
            if in_stop and len(stop_pts) > 0:
                arrive = stop_pts[0]['ts']
                leave  = stop_pts[-1]['ts']
                dur_min = (leave - arrive).total_seconds() / 60
                if dur_min >= min_duration_min:
                    stops.append({
                        'lat': np.mean([p['lat'] for p in stop_pts]),
                        'lon': np.mean([p['lon'] for p in stop_pts]),
                        'arrive_ts': arrive,
                        'leave_ts': leave,
                        'duration_min': dur_min,
                        'n_points': len(stop_pts),
                    })
            stop_pts = []
            in_stop = False

    return pd.DataFrame(stops)

# Apply to all individuals
stops_all = []
for uid, group in df_gps.groupby('uid'):
    group_sorted = group.sort_values('ts')
    stops = detect_stops(group_sorted)
    if len(stops) > 0:
        stops['uid'] = uid
        stops_all.append(stops)

df_stops = pd.concat(stops_all, ignore_index=True)
print(f"\nDetected {len(df_stops)} stops across {df_stops['uid'].nunique()} individuals")
print(f"Mean stop duration: {df_stops['duration_min'].mean():.1f} min")
print(f"Median stops per person: {df_stops.groupby('uid').size().median():.0f}")
```

### Step 2: Origin-Destination Matrix

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster

# ------------------------------------------------------------------ #
# Build OD matrix by clustering stop locations into zones
# ------------------------------------------------------------------ #

# Cluster stops into geographic zones (k-means on lat/lon)
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Use all unique stop locations
stop_coords = df_stops[['lat', 'lon']].values
scaler = StandardScaler()
coords_scaled = scaler.fit_transform(stop_coords)

n_zones = 8  # Number of geographic zones
kmeans = KMeans(n_clusters=n_zones, random_state=42, n_init=10)
df_stops['zone'] = kmeans.fit_predict(coords_scaled)

zone_centers = scaler.inverse_transform(kmeans.cluster_centers_)
zone_centers_df = pd.DataFrame(zone_centers, columns=['lat', 'lon'])

# Assign zone names (simplified)
zone_names = [f"Zone {i+1}" for i in range(n_zones)]

# ---- Build OD matrix ------------------------------------------- #
def build_od_matrix(stops_df, n_zones):
    """
    Build OD matrix from consecutive stops (same individual).
    Each trip = from departure zone to arrival zone.
    """
    od = np.zeros((n_zones, n_zones), dtype=int)
    for uid, group in stops_df.groupby('uid'):
        group_sorted = group.sort_values('arrive_ts').reset_index(drop=True)
        for i in range(len(group_sorted) - 1):
            origin = int(group_sorted.loc[i, 'zone'])
            dest   = int(group_sorted.loc[i+1, 'zone'])
            if origin != dest:
                od[origin, dest] += 1
    return od

od_matrix = build_od_matrix(df_stops, n_zones)

# Print OD matrix
od_df = pd.DataFrame(od_matrix, index=zone_names, columns=zone_names)
print("\nOrigin-Destination Matrix (trips):")
print(od_df)

# ---- Visualize OD matrix and flow map -------------------------- #
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# OD matrix heatmap
im = axes[0].imshow(od_matrix, cmap='YlOrRd', aspect='auto')
plt.colorbar(im, ax=axes[0], label='Number of trips')
axes[0].set_xticks(range(n_zones)); axes[0].set_xticklabels(zone_names, rotation=30, ha='right', fontsize=8)
axes[0].set_yticks(range(n_zones)); axes[0].set_yticklabels(zone_names, fontsize=8)
axes[0].set_xlabel("Destination"); axes[0].set_ylabel("Origin")
axes[0].set_title("Origin-Destination Matrix")
for i in range(n_zones):
    for j in range(n_zones):
        if od_matrix[i,j] > 0:
            axes[0].text(j, i, str(od_matrix[i,j]), ha='center', va='center',
                          fontsize=8, color='white' if od_matrix[i,j] > od_matrix.max()/2 else 'black')

# Flow map (desire lines)
axes[1].scatter(zone_centers_df['lon'], zone_centers_df['lat'],
                c='red', s=200, zorder=5, edgecolors='black', linewidths=0.7)
for i, name in enumerate(zone_names):
    axes[1].annotate(name, (zone_centers_df.loc[i,'lon'], zone_centers_df.loc[i,'lat']),
                      fontsize=8, xytext=(5,5), textcoords='offset points')

max_flow = od_matrix.max()
for i in range(n_zones):
    for j in range(i+1, n_zones):
        flow = od_matrix[i,j] + od_matrix[j,i]
        if flow > 0:
            x = [zone_centers_df.loc[i,'lon'], zone_centers_df.loc[j,'lon']]
            y = [zone_centers_df.loc[i,'lat'],  zone_centers_df.loc[j,'lat']]
            lw = flow / max_flow * 5 + 0.5
            axes[1].plot(x, y, 'b-', linewidth=lw, alpha=0.5)

axes[1].set_xlabel("Longitude"); axes[1].set_ylabel("Latitude")
axes[1].set_title("Mobility Flow Map (Desire Lines)")
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("od_matrix.png", dpi=150)
plt.show()
```

### Step 3: Mobility Metrics (Entropy, Radius of Gyration)

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ------------------------------------------------------------------ #
# Compute individual mobility metrics
# ------------------------------------------------------------------ #

def radius_of_gyration(stops_df, uid):
    """Radius of gyration for one individual (in km)."""
    user_stops = stops_df[stops_df['uid'] == uid][['lat','lon','duration_min']]
    if len(user_stops) == 0:
        return np.nan
    weights = user_stops['duration_min'].values
    weights = weights / weights.sum()
    lat_cm = np.average(user_stops['lat'], weights=weights)
    lon_cm = np.average(user_stops['lon'], weights=weights)
    dists = haversine(user_stops['lat'].values, user_stops['lon'].values,
                      np.full(len(user_stops), lat_cm),
                      np.full(len(user_stops), lon_cm))
    return np.sqrt(np.average(dists**2, weights=weights))

def mobility_entropy(stops_df, uid, zone_col='zone'):
    """Shannon entropy over zone visit time fractions."""
    user_stops = stops_df[stops_df['uid'] == uid]
    if len(user_stops) == 0:
        return np.nan
    zone_time = user_stops.groupby(zone_col)['duration_min'].sum()
    p = zone_time / zone_time.sum()
    p = p[p > 0]
    return -np.sum(p * np.log(p))

def detect_home(stops_df, uid, night_hours=(22, 6)):
    """Detect home location: most frequent stop during night hours."""
    user_stops = stops_df[stops_df['uid'] == uid].copy()
    hour = user_stops['arrive_ts'].dt.hour
    night_mask = (hour >= night_hours[0]) | (hour < night_hours[1])
    night_stops = user_stops[night_mask]
    if len(night_stops) == 0:
        return np.nan, np.nan
    home_zone = night_stops.groupby('zone')['duration_min'].sum().idxmax()
    home_loc = night_stops[night_stops['zone'] == home_zone][['lat','lon']].mean()
    return home_loc['lat'], home_loc['lon']

# Compute metrics for all individuals
mobility_metrics = []
for uid in df_stops['uid'].unique():
    rg  = radius_of_gyration(df_stops, uid)
    S   = mobility_entropy(df_stops, uid)
    h_lat, h_lon = detect_home(df_stops, uid)
    n_stops_per_day = df_stops[df_stops['uid']==uid]['arrive_ts'].apply(lambda x: x.date()).nunique()
    mobility_metrics.append({
        'uid': uid,
        'n_stops': len(df_stops[df_stops['uid']==uid]),
        'radius_of_gyration_km': rg,
        'mobility_entropy': S,
        'n_zones_visited': df_stops[df_stops['uid']==uid]['zone'].nunique(),
        'home_lat': h_lat, 'home_lon': h_lon,
    })

df_mobility = pd.DataFrame(mobility_metrics)
print("Individual Mobility Metrics:")
print(df_mobility[['uid','n_stops','radius_of_gyration_km','mobility_entropy','n_zones_visited']].round(3).to_string(index=False))

# ---- Visualization --------------------------------------------- #
fig, axes = plt.subplots(2, 2, figsize=(12, 9))

# Radius of gyration distribution
axes[0][0].hist(df_mobility['radius_of_gyration_km'].dropna(), bins=8,
                color='steelblue', edgecolor='black', linewidth=0.7)
axes[0][0].set_xlabel("Radius of Gyration (km)"); axes[0][0].set_ylabel("Count")
axes[0][0].set_title("Radius of Gyration Distribution"); axes[0][0].grid(axis='y', alpha=0.3)

# Mobility entropy distribution
axes[0][1].hist(df_mobility['mobility_entropy'].dropna(), bins=8,
                color='coral', edgecolor='black', linewidth=0.7)
axes[0][1].set_xlabel("Mobility Entropy (nats)"); axes[0][1].set_ylabel("Count")
axes[0][1].set_title("Mobility Entropy Distribution"); axes[0][1].grid(axis='y', alpha=0.3)

# Rg vs. entropy scatter
axes[1][0].scatter(df_mobility['radius_of_gyration_km'],
                    df_mobility['mobility_entropy'],
                    c='purple', s=80, edgecolors='black', linewidths=0.5)
for _, row in df_mobility.iterrows():
    axes[1][0].annotate(f"U{row['uid']}", (row['radius_of_gyration_km'], row['mobility_entropy']),
                         fontsize=8, xytext=(3,3), textcoords='offset points')
axes[1][0].set_xlabel("Radius of Gyration (km)"); axes[1][0].set_ylabel("Mobility Entropy")
axes[1][0].set_title("Mobility Extent vs. Diversity"); axes[1][0].grid(True, alpha=0.3)

# Stops per day by zone (heatmap)
zone_visit_matrix = df_stops.groupby(['uid','zone'])['duration_min'].sum().unstack(fill_value=0)
im = axes[1][1].imshow(zone_visit_matrix.values, cmap='YlOrRd', aspect='auto')
plt.colorbar(im, ax=axes[1][1], label='Total time (min)')
axes[1][1].set_xticks(range(n_zones)); axes[1][1].set_xticklabels(zone_names, rotation=30, ha='right', fontsize=8)
axes[1][1].set_yticks(range(len(zone_visit_matrix))); axes[1][1].set_yticklabels([f"U{uid}" for uid in zone_visit_matrix.index], fontsize=8)
axes[1][1].set_xlabel("Zone"); axes[1][1].set_ylabel("Individual")
axes[1][1].set_title("Time Spent per Zone per Individual")

plt.tight_layout()
plt.savefig("mobility_metrics.png", dpi=150)
plt.show()
```

---

## Advanced Usage

### Commuting Pattern Analysis

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ---- Hourly trip departure distribution ------------------------ #
df_stops['hour_of_day'] = df_stops['arrive_ts'].dt.hour
trip_hours = []
for uid, group in df_stops.groupby('uid'):
    g = group.sort_values('arrive_ts')
    for i in range(len(g)-1):
        o_zone = g.iloc[i]['zone']
        d_zone = g.iloc[i+1]['zone']
        if o_zone != d_zone:
            trip_hours.append(g.iloc[i]['leave_ts'].hour if 'leave_ts' in g.columns else g.iloc[i]['arrive_ts'].hour)

fig, ax = plt.subplots(figsize=(10, 4))
ax.hist(trip_hours, bins=24, range=(0,24), color='steelblue', edgecolor='black', linewidth=0.5)
ax.set_xlabel("Hour of day"); ax.set_ylabel("Number of trips")
ax.set_title("Trip Departure Distribution by Hour"); ax.grid(axis='y', alpha=0.3)
ax.set_xticks(range(0, 25, 2)); ax.axvspan(7, 9, alpha=0.1, color='red', label='AM peak')
ax.axvspan(17, 19, alpha=0.1, color='orange', label='PM peak')
ax.legend(); plt.tight_layout(); plt.savefig("trip_hours.png", dpi=150); plt.show()
```

---

## Troubleshooting

### Stop detection misses short stops

```python
# Reduce minimum duration
stops = detect_stops(trace, min_duration_min=5)  # 5 min instead of 15
```

### GPS traces with gaps (network outage)

```python
# Filter out inter-point intervals > threshold
df_gps['time_gap'] = df_gps.groupby('uid')['ts'].diff().dt.total_seconds() / 60
df_gps = df_gps[df_gps['time_gap'] < 60]  # Remove gaps > 60 min
```

### Haversine distance overflow for antipodal points

```python
# Clip arcsin argument to [-1, 1]
return 2 * R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
```

### Version Compatibility

| Package | Tested versions | Notes |
|:--------|:----------------|:------|
| pandas | 2.0, 2.1 | `dt.hour`, `groupby` stable |
| scikit-learn | 1.3, 1.4 | `KMeans` API stable |
| geopandas | 0.14 | For spatial joining with zones |

---

## External Resources

### Official Documentation

- [pandas datetime](https://pandas.pydata.org/docs/user_guide/timeseries.html)

### Key Papers

- González, M.C., Hidalgo, C.A. & Barabási, A.-L. (2008). *Understanding individual human mobility patterns*. Nature.
- Song, C. et al. (2010). *Limits of predictability in human mobility*. Science.

---

## Examples

### Example 1: Daily Mobility Profile

```python
import numpy as np
import matplotlib.pyplot as plt

# Average time at home vs. outside by hour
for uid in range(3):
    user_trace = df_gps[df_gps['uid'] == uid].copy()
    user_trace['hour'] = user_trace['ts'].dt.hour
    # Simplified: count records near home
    home_lat = user_trace.groupby('uid').apply(lambda x: x.iloc[0]['lat']).values[0] if len(user_trace) > 0 else 40.71
    home_lon = user_trace.groupby('uid').apply(lambda x: x.iloc[0]['lon']).values[0] if len(user_trace) > 0 else -74.01

print("Daily profile analysis requires full trace data — see Step 1 output.")
```

### Example 2: Mobility Predictability (Maximum Entropy)

```python
import numpy as np

# Maximum entropy = log(n_zones) if uniformly distributed
n_zones_visited = df_mobility['n_zones_visited'].values
max_entropy = np.log(n_zones_visited)
actual_entropy = df_mobility['mobility_entropy'].values
predictability = 1 - actual_entropy / (max_entropy + 1e-8)

print("Individual mobility predictability (1 = perfectly predictable):")
for uid, pred in zip(df_mobility['uid'], predictability):
    print(f"  User {uid}: {pred:.3f}")
print(f"\nMean predictability: {predictability.mean():.3f}")
```

---

*Last updated: 2026-03-17 | Maintainer: @xjtulyc*
*Issues: [GitHub Issues](https://github.com/xjtulyc/awesome-rosetta-skills/issues)*
