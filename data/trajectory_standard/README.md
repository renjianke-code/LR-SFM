# Standard Trajectory Data

ETH-UCY and SDD are not bundled in this repository. Put the processed trajectory
files here before running those configs, or pass `--data-root` to a directory
with the same structure.

Expected layout:

```text
data/trajectory_standard/
  datasets/
    eth.plist
    hotel.plist
    univ.plist
    zara1.plist
    zara2.plist
    sdd.plist
    subsets/
      *.plist
  data/
    */
      true_pos_.csv
```

The loader uses only observed trajectories and the train/test split metadata.
Future trajectories from the held-out test split are used only as evaluation
targets.
