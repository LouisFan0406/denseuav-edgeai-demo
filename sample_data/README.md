# Sample Data

This directory should contain a minimal `(UAV image, satellite tile)` pair for
verifying the demo. The actual image files are not included in the public
repository to avoid leaking location-identifying information.

## Required Files

| File | Description |
|------|-------------|
| `uav_sample.jpg` | A 224×224 (or larger, will be resized) UAV image, downward-facing, ImageNet-compatible |
| `sat_sample.jpg` | A 224×224 (or larger, will be resized) satellite tile centered around the UAV's anchor GPS |
| `expected_output.json` | Reference output values for sanity-checking the demo |

## How to Obtain

For academic verification, please contact the author (see top-level README) to
request a sample pair. The pair is provided with EXIF stripped and no GPS
metadata, so it is safe to use without revealing the actual flight location.

## Alternative: Use Your Own Image Pair

You can run the demo on any (UAV-style overhead photo, satellite tile) pair:

```bash
python ../inference_demo.py \
    --uav path/to/your/uav.jpg \
    --sat path/to/your/satellite_tile.jpg
```

The model expects:

- Both inputs are roughly the same ground area
- Camera roughly downward (gimbal pitch close to -90°)
- Altitude roughly within the trained range (40 / 50 / 60 / 80 m)
- Daylight, clear weather

For best results, the satellite tile should be from Google Earth or similar
imagery, cropped to a 512×512 (or 224×224) square around the same anchor.

## Expected Output Format

`expected_output.json` should look like:

```json
{
  "dx_m": 0.182,
  "dy_m": -0.094,
  "z_m": 50,
  "z_probabilities": [0.071, 0.812, 0.105, 0.012],
  "comment": "Reference output on the provided sample pair. Slight variations from this are expected on different hardware."
}
```
