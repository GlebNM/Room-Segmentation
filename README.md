# HOV-SG-based room segmentation
## Environment
To setup conda environemnt:
```bash
conda env create -f environment/conda.yml
```

To activate the environment:
```bash
conda activate room_segmentation
```

## Script parameters
```bash
python main.py --floor_path=<path_to_point_cloud> --output_path=<output_path> 
```

## Config variables
- `floor_bounds_quantiles`: Quantiles to determine floor bounds (only for wall extraction).
- `below_ceiling`: Slice bound below the ceiling (only for wall extraction).
- `above_floor`: Slice bound above the floor.
- `below_ceiling_full`: Slice bound below the ceiling for full point cloud.
- `resolution`: Resolution of the map. `num_bins` = `grid_size` / `resolution`
- `hist_threshold`: Threshold in percentage of the maximum value of the distance map.
- `min_area_m`: Minimum area to be considered a room is calculated as (`min_area_m` / `resolution`) ^ 2