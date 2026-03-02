# Plotting Refactoring Plan

Working document — edit freely, check off items as done.

---

## Goals

1. All datasets are loaded via `ggpymanager.load(name, config)` — no hardcoded paths,
   no direct `xr.open_dataset()` calls in the plotting layer
2. `ggpymanager` owns all generic GRAMM/GRAL workflow logic; `paris_2025` contains
   only Paris-paper-specific code
3. All plot functions accept only `xr.Dataset` / `xr.DataArray` — no `p.*` calls inside
   plot functions
4. All shared data loading lives in one place (`_loaders.py`) in the plotting package
5. No `lru_cache` hidden state in plot modules
6. No hardcoded absolute paths anywhere
7. Station plotting primitives are shared, not buried in `tracer_comparison.py`
8. Functions are unit-testable with a small synthetic dataset

---

## All loaded files

The table below maps every data file consumed by the plotting layer to:
the GGPyManager command that creates it, whether `ggpymanager.load` already
handles it, the current `p.*` accessor (if one exists), and which plotting
modules consume it.

### GGPyManager output files — already in `ggpymanager.load` ✅

| `load()` key | File | Created by | `p.*` accessor | Plotting module(s) |
|---|---|---|---|---|
| `matching_loss` | `output_path/matching_loss.nc` | `ggpy match` | — | `matching_methods`, `meteo_from_catalog` |
| `concentration_timeseries` | `output_path/concentration_timeseries.nc` | `ggpy timeseries` | `p.model.get_co2_data()` | `tracer_comparison`, `tracer_from_catalog` |
| `gramm_meteo_timeseries` | `output_path/gramm_meteo_timeseries.nc` | `ggpy timeseries` | `p.model.get_gramm_meteo_data()` ⚠️ | `meteo_from_catalog` |
| `gral_meteo_timeseries` | `output_path/gral_meteo_timeseries.nc` | `ggpy timeseries` | `p.model.get_gral_meteo_data()` ⚠️ | `matching_methods`, `meteo_from_catalog`, `gradient_for_matching` |
| `temperature` | `data_path/meteo_path/temperature.nc` | `prepare_inputs.py` | — | (not yet in plotting) |
| `pressure` | `data_path/meteo_path/pressure.nc` | `prepare_inputs.py` | — | (not yet in plotting) |

> ⚠️ `p.model.get_gramm/gral_meteo_data()` also does post-load preprocessing (variable
> renaming, wind component derivation) on top of the raw timeseries file —
> see Phase A below.

### Model input files — now in `ggpymanager.load` ✅

| `load()` key | File | Config key | `p.*` accessor | Plotting module(s) |
|---|---|---|---|---|
| `source_groups` | `source_groups_path` (top-level) | `source_groups_path` ✅ | — | `fluxes`, `matching_methods`, `tracer_from_catalog` |
| `temporal_profiles` | `temporal_profiles_path` (top-level) | `temporal_profiles_path` ✅ | — | `fluxes` |
| `gramm_meteo_catalog` | `gramm_meteo_path/meteo.nc` | `gramm_meteo_path` ✅ | `p.model.get_gramm_meteo_data()` | `meteo_from_catalog` |
| `gral_meteo_catalog` | `gral_meteo_path/meteo.nc` | `gral_meteo_path` ✅ | `p.model.get_gral_meteo_data()` | `meteo_from_catalog`, `matching_methods` |
| `gral_co2_catalog` | `gral_co2_path/co2.nc` | `gral_co2_path` ✅ | — | `matching_methods` (currently hardcoded ⚠️) |

> `cadastre.dat` and `point.dat` are GRAL binary/text files parsed by
> `ggpymanager.io.parsers` — they do NOT belong in `ggpymanager.load()` which uses
> `xr.open_mfdataset`. Keep using the parser directly.

### Measurement files — missing from `ggpymanager.load` ❌

| Proposed `load()` key | File | Config key | `p.*` accessor | Plotting module(s) |
|---|---|---|---|---|
| `meteo_measurements` | `data_path/meteo_path/meteo.nc` | derivable from `data_path` + `meteo_path` ✅ | `p.meteo.get_meteo_measurements()` | `meteo_measurements`, `meteo_from_catalog`, `tracer_background`, `misc` |
| `co2_measurements` | `data_path/6_measurements/6_2_tracers/co2.nc` | ❌ no config key yet | `p.tracers.get_co2_measurements()` | `tracer_measurements`, `tracer_comparison`, `tracer_background`, `misc` |
| `background_co2` | `output_path/background_co2.nc` | derivable from `output_path` ✅ | `p.background.get_dynamic_background_co2()` | `tracer_comparison`, `tracer_background` |

> `background_co2` does **not yet exist as a file** — currently three on-the-fly
> functions compute it. The goal is to pre-compute all background methods and save
> them as a single NetCDF. See Phase B below.

### Files requiring new config keys

| File | Proposed config key | Currently in config.yaml? |
|---|---|---|
| CO2 measurements | `co2_measurements_path` | ❌ path is hardcoded in `tracers.py` |
| buildings.nc | `buildings_path` | ❌ hardcoded in `meteo_from_catalog.py` |
| area_id.nc | `area_id_path` | ❌ hardcoded in `tracer_from_catalog.py` |

---

## Phase A — Extend `ggpymanager.load` and clean up `p.*` accessors

This work lives in the **GGpyManager repo**, not paris_2025.

### A1 — Add missing keys to `ggpymanager.load` (file_api.py)

Add these entries to the `file_paths` dict:

```python
# Model inputs
"source_groups":    Path(c["source_groups_path"]),
"temporal_profiles": Path(c["temporal_profiles_path"]),
# Model output at stations (preprocessing auto-applied for meteo keys)
"gramm_meteo_catalog":  Path(c["gramm_meteo_path"]) / "meteo.nc",
"gral_meteo_catalog":   Path(c["gral_meteo_path"])  / "meteo.nc",
"gral_co2_catalog":     Path(c["gral_co2_path"])     / "co2.nc",
# Measurements
"meteo_measurements": Path(c["data_path"]) / c["meteo_path"] / "meteo.nc",
"co2_measurements":   Path(c["co2_measurements_path"]),
```

- [x] Add the 7 entries above to `file_api.py`
- [x] Update the docstring listing valid `data_name` values

### A2 — Move generic meteo preprocessing into ggpymanager ✅

`p.model.get_gramm_meteo_data()` and `p.model.get_gral_meteo_data()` did generic
post-load transformations that belong in ggpymanager, not a city-specific package.

**Approach chosen: hybrid of Option 1 + 2.** Public functions
`ggpy.io.preprocess_gramm_meteo(ds)` and `ggpy.io.preprocess_gral_meteo(ds)` were
added to `ggpymanager/io/file_api.py` and are also **auto-applied by `ggpy.load()`**
for the four meteo keys (`gramm_meteo_catalog`, `gral_meteo_catalog`,
`gramm_meteo_timeseries`, `gral_meteo_timeseries`).

What each function does:

- **`preprocess_gral_meteo`**: warns + renames `ux`/`vy` → `u`/`v` (old server
  naming, will be fixed in a future server update); renames `direction` →
  `synoptic_wind_direction` and `speed` → `synoptic_wind_speed`; adds derived
  `wind_speed` and `wind_direction`.
- **`preprocess_gramm_meteo`**: warns + drops spurious `speed`; adds derived
  `wind_speed` and `wind_direction`; conditionally transposes `(sim_id, station)`.

Additionally:

- **`"model_meteo"` loader** added to `ggpy.load()`: loads both catalog files
  (preprocessing auto-applied), then concatenates the station-selected slice per
  the `config["matching"]["stations"]` map.
- **`cli_functions.load_matching_model_meteo()`** refactored to a one-liner:
  `return ggp.load("model_meteo", config)`.
- **All `ux`/`vy` references in paris_2025 replaced with `u`/`v`** (in
  `meteo_from_catalog.py`, `matching_methods.py`).

- [x] Decide on option
- [x] Implement in ggpymanager (`file_api.py`: preprocessing functions + auto-dispatch in `load()`)
- [x] Slim down `p.model.get_gramm_meteo_data()` → `ggpy.load("gramm_meteo_catalog", CONFIG)`
- [x] Slim down `p.model.get_gral_meteo_data()` → `ggpy.load("gral_meteo_catalog", CONFIG)`
- [x] Slim down `p.model.get_model_meteo_data()` → `ggpy.load("model_meteo", CONFIG)`
- [x] Add `"model_meteo"` to `ggpy.load()`
- [x] Refactor `cli_functions.load_matching_model_meteo()` to use `ggpy.load()`
- [x] Replace all `ux`/`vy` with `u`/`v` in paris_2025

### A3 — Add `co2_measurements_path` to ggpymanager Config + config.yaml

The CO2 measurement path is currently hardcoded in `paris_2025/tracers.py` as
`data_path / "6_measurements/6_2_tracers/co2.nc"`. This path structure is
Paris-specific (numbered folders).

Options:
- **Option 1 (recommended):** Add `co2_measurements_path` as an explicit top-level
  config key in both the ggpymanager `Config` Pydantic model and `config.yaml`.
- **Option 2:** Keep the path derivation in `paris_2025/tracers.py` and do NOT add
  it to `ggpymanager.load` (treat CO2 measurements as city-specific).

- [x] Decide: generic enough for ggpymanager, or Paris-specific? → **Option 1: generic**
- [x] If Option 1: add `co2_measurements_path` to `ggpymanager/config.py` (Config model)
      and to `paris_2025/config.yaml`
- [x] Update `p.tracers.get_co2_measurements()` to call `ggp.load("co2_measurements", CONFIG)`
      in all callers; function marked deprecated with `warnings.warn(DeprecationWarning)`

### A4 — Add `buildings_path` and `area_id_path` to config.yaml

These are Paris-specific model input files. They need config keys to replace the
hardcoded paths in `meteo_from_catalog.py` and `tracer_from_catalog.py`.

- [x] Add `buildings_path` to `config.yaml` with the correct relative path
- [x] Add `area_id_path` to `config.yaml` with the correct relative path
- [x] These are Paris-specific → added as optional fields to ggpymanager `Config` model
      (`buildings_path: str | None`, `area_id_path: str | None`, both default `None`)

---

## Phase B — Pre-compute background CO2 and save as NetCDF

Currently `paris_2025/background.py` has three on-the-fly functions:
- `get_dynamic_background_co2()` — upwind Picarro station per timestep (dims: `time`)
- `get_minimum_background_co2()` — minimum across all Picarro stations (dims: `time`)
- `get_binned_background_co2(bins)` — minimum per height bin (dims: `time`, `height_bins`)

These are slow (load + filter + compute at import time) and not reproducible across
sessions. The fix is to pre-compute once and save to a NetCDF file.

### B1 — Design the NetCDF structure

Actual `output_path/background_co2.nc` structure (implemented — richer than original sketch):

```
Dimensions: time, height_bins, station
Variables:
  dynamic_background(time)                  — CO2 ppm
  dynamic_background_station(time)          — string, selected station name
  minimum_background(time)                  — CO2 ppm
  minimum_background_station(time)          — string, selected station name
  binned_background(time, station)          — CO2 ppm, station-matched
    coord: matched_height_bin_center(station)
  binned_background_by_label(time, height_bins) — CO2 ppm, label-indexed
    coords: height_bin_left, height_bin_right, height_bin_center
  binned_background_station(time, height_bins)  — string, selected station name
```

Station names stored as plain string variables (not coordinates) to avoid
CF-check issues with string-typed coordinate arrays.

- [x] Confirm the structure above covers all use-cases in the plotting modules
- [x] Decide: does `height_bins` dimension have a fixed default
      (e.g., `[0, 40, 80, 120, 200]`) or should it be config-driven?
      → **config-driven** via `background.height_bins` in `config.yaml`

### B2 — Add a `create_background_co2()` function

Add to `paris_2025/background.py`:
```python
def create_background_co2() -> None:
    """Compute all background methods and save to background_co2.nc."""
```

This function:
- calls the three existing computation functions
- assembles them into one `xr.Dataset`
- saves via `ggp.io.writers.save_netcdf_with_cf_check()`
- is called from `scripts/prepare_inputs.py`

- [x] Implement `create_background_co2()` in `paris_2025/background.py`
- [x] Add a call to it in `scripts/prepare_inputs.py`

### B3 — Add `background_co2` to `ggpymanager.load`

```python
"background_co2": Path(c["output_path"]) / BACKGROUND_CO2_FILE_NAME,
```

- [x] Add `BACKGROUND_CO2_FILE_NAME = "background_co2.nc"` to `ggpymanager/config.py`
- [x] Add the `"background_co2"` entry to `file_api.py`
- [x] Update the docstring

### B4 — Update callers

After B2/B3, `p.background.get_dynamic_background_co2()` can be replaced by
`ggp.load("background_co2", config)["dynamic_background"]` everywhere.

- [x] Update `tracer_comparison.cache_data()` → direct `ggp.load("background_co2", CONFIG)["binned_background_by_label"]`
- [x] Update `tracer_comparison._load_sector_enhancement_data()` → same
- [x] Update `tracer_background.py` plot functions → direct `ggp.load("background_co2", CONFIG)[...]`
- [x] Remove the three `get_*_background_co2()` wrappers from `background.py`

---

## Phase 1 — Fix hardcoded paths (low risk, start here — paris_2025 only)

**Files:** `matching_methods.py`, `tracer_from_catalog.py`, `meteo_from_catalog.py`

- [x] `matching_methods.py`: replace the 3 hardcoded `matching_loss.nc` paths
      → `Path(CONFIG["output_path"]) / ggp.config.MATCHING_LOSS_FILE_NAME`
- [x] `matching_methods._load_and_prepare_data()`: replace hardcoded `gral_concentration_path`
      → `Path(CONFIG["gral_co2_path"]) / "co2.nc"`
- [x] `matching_methods._load_matching_analysis_data()`: same
- [x] `meteo_from_catalog.plot_meteo_model_comparison()`: replace hardcoded `buildings_path`
      → `Path(CONFIG["buildings_path"])` (after Phase A4 adds the config key)
- [x] `meteo_from_catalog.plot_comparison_of_different_matching_methods()`: replace hardcoded path
      → `Path(CONFIG["output_path"]) / ggp.config.MATCHING_LOSS_FILE_NAME`
- [x] `tracer_from_catalog.plot_source_group_contribution_to_stations()`: replace both hardcoded paths
      → `Path(CONFIG["source_groups_path"])` and `Path(CONFIG["area_id_path"])`
      (after Phase A4 adds `area_id_path`)
- [x] Verify `create_figures.py` still runs (syntax check passed, modules import successfully)

---

## Phase 2 — Create `_loaders.py` in the plotting package

**Status:** ✅ COMPLETE (complex loaders moved, simple one-line `ggp.load()` calls kept inline)

Created `src/paris_2025/plotting/_loaders.py` containing complex loaders that do
preprocessing. Simple one-line `ggp.load()` calls remain inline in plot functions.

### Loaders written

| Loader function | Returns | Complexity |
|---|---|---|
| `load_and_prepare_matching_data()` | tuple of (concentration, meteo, loss, source_groups, sim_ids) | Unit conversion + top-N selection |
| `load_matching_analysis_data()` | tuple of (concentration, loss, speed, direction, stab_class) | Unit conversion + derived calculations |
| `load_flux_maps_data()` | tuple of (cadastre, source_groups, point_da, GRAL) | Cadastre/point parsing + type merging |

Simple one-line loaders (e.g., `load_source_groups()`, `load_temporal_profiles()`)
remain inline in plot functions to avoid over-abstraction.

### Existing helpers migrated

- [x] `fluxes._load_flux_maps_data()` → `_loaders.load_flux_maps_data()`
- [x] `matching_methods._load_and_prepare_data()` → `_loaders.load_and_prepare_matching_data()`
- [x] `matching_methods._load_matching_analysis_data()` → `_loaders.load_matching_analysis_data()`

All loaders accept path parameters with defaults derived from `CONFIG`.

---

## Phase 3 — Move cached loaders to `_loaders.py`

**Status:** ✅ COMPLETE

The `cache_data()` and `_load_sector_enhancement_data()` functions load immutable NetCDF
files, making `@lru_cache` a legitimate performance optimization. These have been moved
to `_loaders.py` **keeping the cache decorator** because:

- ✓ Deterministic: same loss_type always returns same data
- ✓ Safe: NetCDF files on disk never change during a run
- ✓ Performant: multiple plot functions hit the cache
- ✓ Transparent: caching is an implementation detail
- ✓ Simple: `create_figures.py` stays unchanged

**Completed tasks:**
1. [x] Moved `cache_data(loss_type)` → `_loaders.cache_data(loss_type)` with `@lru_cache`
2. [x] Moved `_load_sector_enhancement_data(loss_type)` → `_loaders.load_sector_enhancement_data(loss_type)` with `@lru_cache`
3. [x] Updated `tracer_comparison.py` to import and call the loaders
4. [x] Left `create_figures.py` completely unchanged
5. [x] Removed old function definitions from `tracer_comparison.py`
6. [x] Removed unused imports (`lru_cache`, `ProgressBar`)
7. [x] Updated docstring references to use new function names
8. [x] Verified syntax and imports work correctly

**Changes made in `tracer_comparison.py`:**
- [x] Import: `from paris_2025.plotting._loaders import cache_data, load_sector_enhancement_data`
- [x] Removed old `cache_data()` function definition
- [x] Removed old `_load_sector_enhancement_data()` function definition
- [x] Updated 2 function calls: `plot_sector_cycles_per_station()` and `plot_diurnal_cycle_by_weekday()`
- [x] Updated docstring references

---

## Phase 4 — Move station primitives to `common.py`

These three functions in `tracer_comparison.py` are general-purpose:

- [ ] `station_scatter_plot(fig_path, data_x, data_y, co2, …)` → move to `common.py`
- [ ] `station_line_plot(model, model_data, measurement_data, background_data, …)` → move to `common.py`
- [ ] `station_sector_plot(model_enhancement, co2, background, …)` → move to `common.py`
- [ ] Update all callers in `tracer_comparison.py` to import from `common`

---

## Phase 5 — Enforce xarray-only signatures

After Phases A + 2, every `p.*` call in a plot function should have been lifted
out. Final audit:

| File | Function | Current non-xarray call | Fix |
|---|---|---|---|
| `gradient_for_matching.py` | `create_figure()` | `p.model.*` internally | receive datasets as parameters |
| `tracer_from_catalog.py` | all 3 | inline `p.tracers.*` / `p.model.*` | lift to `_loaders.py` |
| `meteo_from_catalog.py` | most | inline `p.meteo.*`, `p.model.*` | lift to `_loaders.py` |
| `tracer_background.py` | all | `p.meteo.get_meteo_measurements()` | lift to `_loaders.py` |
| `tracer_measurements.py` | all | `p.tracers.get_co2_measurements()` | lift to `_loaders.py` |
| `meteo_measurements.py` | all | `p.meteo.get_meteo_measurements()` | lift to `_loaders.py` |
| `misc.py` | `plot_temperature_anomaly_with_co2()` | `p.tracers.*`, `p.meteo.*` | lift to `_loaders.py` |
| `tracer_comparison.py` | `cache_data()` | `p.background.*`, `p.tracers.*` | see Phase 3 |

---

## Target architecture

### ggpymanager (generic, city-agnostic)

```
ggpymanager/io/
├── file_api.py        # ggpy.load() — extended with new keys (Phase A1)
├── preprocessing.py   # NEW: preprocess_gramm_meteo(), preprocess_gral_meteo() (Phase A2)
├── readers.py
├── parsers.py         # cadastre.dat, point.dat — stays as-is
└── writers.py
```

### paris_2025 (Paris-paper-specific)

```
src/paris_2025/
├── background.py      # get_dynamic_background_co2() — thin wrapper or moved to ggpy
├── meteo.py           # get_meteo_measurements() → ggp.load("meteo_measurements", CONFIG)
├── model.py           # get_*_meteo_data() → ggp.load(...) + ggpy.io.preprocess_*()
├── tracers.py         # get_co2_measurements() → ggp.load("co2_measurements", CONFIG)
└── plotting/
    ├── __init__.py
    ├── common.py       # get_metadata, save_table_as_png, station plot primitives
    ├── _loaders.py     # NEW — all load_*() call ggp.load() or slim p.* wrappers
    ├── fluxes.py
    ├── matching_methods.py
    ├── tracer_comparison.py
    ├── tracer_background.py
    ├── tracer_measurements.py
    ├── tracer_from_catalog.py
    ├── meteo_measurements.py
    ├── meteo_from_catalog.py
    ├── gradient_for_matching.py
    └── misc.py
```

---

## Open questions

- [ ] **`co2_measurements_path`** — add to ggpymanager Config + config.yaml, or treat
      as Paris-specific and keep in paris_2025? (See Phase A3)
- [x] **`gramm_meteo_timeseries` vs `gramm_meteo_catalog`** — resolved: they are
      different files. `gramm_meteo_catalog` is the full simulation catalog at
      `gramm_meteo_path/meteo.nc`; `gramm_meteo_timeseries` is a sim_id-selected
      subset saved to `output_path/gramm_meteo_timeseries.nc` by `ggpy timeseries`.
      Both are preprocessed by `preprocess_gramm_meteo()` when loaded via `ggpy.load()`.
- [ ] **`year` parameter pattern** — `tracer_background.py` and others use `year="2023"`.
      Should this become `slice` or `int`?
- [ ] **`station_line_plot` signature** — receives `model: xr.DataArray` only for
      `.code`/`.height` coords. Should the signature be unified?
- [ ] **`get_plot_data()` in `tracer_comparison.py`** — filters by `afternoon_only` /
      `main_wind_direction_only`. `_loaders.py` or plot-level helper?

---

## create_figures.py — keep as-is, incremental improvements only

| Idea | Effort | Benefit |
|---|---|---|
| Add `ProcessPoolExecutor` for parallel figure generation | Low | Faster runs, zero new deps |
| Group figures into named batches (run only "meteo" figures etc.) | Low | Faster dev iteration |
| Add a `--force` flag to regenerate all regardless of mtime | Low | Useful for style changes |

---

## Other improvements to consider

- **Testing**: after Phase 2/3, each plot function is testable with a small synthetic
  `xr.Dataset` — no disk access, no `p.*` imports
- **Type annotations**: `xr.Dataset` vs `xr.DataArray` matters; annotate all loader
  return values and plot function parameters
- **ggpymanager Config Pydantic model**: currently has `extra="forbid"` — every new
  config key must be added to the model or it will raise on load
