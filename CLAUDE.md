# Claude Code — project conventions

This file is read automatically by Claude Code at the start of every session.
**Agents should update this file whenever they discover new conventions or patterns
that would save future sessions from having to re-read source files.**

---

## Plotting conventions (`src/paris_2025/plotting/`)

### Function signature

- First parameter: `fig_path: str | Path`
- One optional path parameter per dataset loaded, with a default derived from
  the module-level path constants or CONFIG — never hardcoded absolute strings:
  ```python
  # preferred
  source_groups_path: str | Path = p.model_input.fluxes.SOURCE_GROUP_NETCDF_PATH
  cadastre_path: str | Path = Path(CONFIG["domain"]["gral"]["conf_path"]) / "cadastre.dat"
  ```

### Data loading helpers

Extract a `_load_*_data()` helper whenever two or more plot functions share the
same datasets. The helper:
- accepts the same path parameters (with the same defaults) as the callers
- returns a plain tuple of loaded/pre-processed objects
- contains no plotting code

Reference implementations:
- `matching_methods._load_and_prepare_data` — GRAL concentration + matching loss
- `matching_methods._load_matching_analysis_data` — sensitivity analysis data
- `fluxes._load_flux_maps_data` — cadastre emissions + source groups + point sources

### Saving and closing

Every plot function must end with:
```python
plt.savefig(fig_path, metadata=get_metadata("One-sentence caption."), bbox_inches="tight")
plt.close(fig)
```

Use `plt.close()` (no argument) only when no explicit `fig` handle exists.

### Registering a new figure

Add a `create_figures_if_missing` call in `scripts/create_figures.py` under the
appropriate `DIR` section:
```python
create_figures_if_missing(
    FIGURE_PATH / DIR / "filename.png",
    module.plot_function,
    # keyword arguments if needed
)
```

### Path sources (in priority order)

1. `p.model_input.<module>.<CONSTANT>` — use when the constant already exists
2. `Path(CONFIG["..."]["path"]) / "filename"` — for paths derivable from config
3. Hardcoded strings as last resort, only as default parameter values (never
   inside the function body)

---

## General coding conventions

### Do not silently replace existing functions

When a user asks to add new plotting logic, **keep existing functions intact**
and introduce a new function with a distinct name. Only replace an existing
function when the user explicitly asks to overwrite it.

- If unsure whether to replace or extend, **ask first**.
- Example: `plot_temporal_scaling_factors` (raw timeseries) and
  `plot_temporal_scaling_factor_cycles` (styled diurnal/weekly/annual) coexist
  rather than the second replacing the first.
