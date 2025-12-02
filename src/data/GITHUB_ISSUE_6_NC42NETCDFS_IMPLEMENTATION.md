# GitHub Issue #6 Implementation Summary - CORRECTED VERSION

## Problem
LRAUV NetCDF files sometimes contain non-monotonic time data, which breaks downstream processing tools that expect monotonic time coordinates. **The critical issue was that each NetCDF group contains multiple independent time variables (e.g., `time_NAL9602`, `time_CTD_NeilBrown`) that each need their own monotonic filtering.**

## Solution Implemented
Complete rewrite of time filtering to handle **multiple independent time variables per group** with the following architecture:

### 1. Per-Variable Time Detection and Filtering
- **`_get_time_filters_for_variables()`**: Identifies ALL time variables in the extraction list and computes monotonic filtering for each independently
- **`_is_time_variable()`**: Determines if a variable is a time coordinate using name patterns and units
- **`_get_monotonic_indices()`**: Computes monotonic indices for any time data array

### 2. Multi-Variable Time Processing
- **`_copy_variable_with_appropriate_time_filter()`**: Applies the correct time filtering based on the specific variable:
  - If the variable IS a time coordinate: applies its own monotonic filtering
  - If the variable DEPENDS on time coordinates: uses the appropriate time dimension's filtering
  - If no time dependencies: copies all data unchanged
- **`_create_dimensions_with_time_filters()`**: Adjusts dimension sizes for each filtered time coordinate
- **`_apply_multidimensional_time_filter()`**: Handles complex multi-dimensional filtering

### 3. Independent Time Coordinate Processing
Each time variable (like `time_NAL9602`, `time_CTD_NeilBrown`) gets:
- Its own monotonic analysis
- Its own filtered indices 
- Its own dimension size adjustment
- Independent logging of filtering results

### 4. Command Line Control (Unchanged)
- **`--filter_monotonic_time`**: Enable time filtering (default behavior)
- **`--no_filter_monotonic_time`**: Disable filtering to preserve all time values

## Key Methods - CORRECTED ARCHITECTURE

```python
def _get_time_filters_for_variables(self, src_group, vars_to_extract) -> dict[str, dict]:
    """Get time filtering info for EACH time variable in the extraction list.
    Returns: {time_var_name: {"indices": list[int], "filtered": bool}}"""
    
def _is_time_variable(self, var_name: str, var) -> bool:
    """Check if a variable is a time coordinate variable."""
    
def _get_monotonic_indices(self, time_data) -> list[int]:
    """Get monotonic indices for any time data array."""
    
def _copy_variable_with_appropriate_time_filter(self, src_group, dst_dataset, var_name, time_filters):
    """Copy variable with the APPROPRIATE time filtering for that specific variable."""
    
def _create_dimensions_with_time_filters(self, src_group, dst_dataset, dims_needed, time_filters):
    """Create dimensions with MULTIPLE time coordinate filtering."""
    
def _apply_multidimensional_time_filter(self, src_var, dst_var, var_name, filtered_dims):
    """Apply time filtering to multi-dimensional variables."""
```

## Testing - CORRECTED VALIDATION
- ✅ Created test with multiple time variables in single group (`time_NAL9602`, `time_CTD_NeilBrown`)
- ✅ Verified independent filtering: `time_NAL9602` (10→8 points), `time_CTD_NeilBrown` (8→6 points)  
- ✅ Confirmed each time variable gets its own monotonic indices
- ✅ Validated that data variables use appropriate time coordinate filtering

## Root Cause Fix
**Previous implementation incorrectly assumed ONE time coordinate per group.** The corrected implementation recognizes that:

1. **Each group can have multiple time variables** (`time_NAL9602`, `time_CTD_NeilBrown`, etc.)
2. **Each time variable needs independent monotonic filtering**
3. **Data variables must use the filtering from their specific time coordinate**
4. **Different time coordinates can have different amounts of filtering**

## Backward Compatibility
- Default behavior enables time filtering for safer processing
- Users can disable filtering with `--no_filter_monotonic_time` if needed
- No breaking changes to existing API
- Works with single time coordinate groups (backward compatible) AND multiple time coordinate groups (new functionality)

This corrected implementation properly addresses GitHub issue #6 by handling the real-world complexity of LRAUV NetCDF files with multiple independent time coordinates per group.