# iLQR Debug Tests Organization

This directory contains satellite trajectory optimization debug tests organized in a systematic structure.

## Directory Structure

```
optimizer/
├── configs/                    # Satellite configurations
│   ├── sat_3_0_mtq.py         # 3 MTQ + 0 RW
│   ├── sat_0_3_rw.py          # 0 MTQ + 3 RW
│   ├── sat_3_1_hybrid.py      # 3 MTQ + 1 RW
│   └── sat_3_3_hybrid.py      # 3 MTQ + 3 RW
│
└── ilqr/                       # Debug scripts and common utilities
    ├── debug_*.py             # Test cases (see matrix below)
    ├── ilqr.py                # iLQR algorithm
    ├── ilqr_viewer.py         # Visualization
    └── trajOpt.py             # Trajectory optimization wrapper
```

## Test Matrix

| Satellite Type | Mission Profile | Timestep | Filename |
|----------------|----------------|----------|----------|
| **3+0** (3 MTQ) | Detumble | dt=60s | `debug_3_0_detumble_dt60.py` |
| **3+0** (3 MTQ) | 90° Slew | dt=60s | `debug_3_0_slew90_dt60.py` |
| **0+3** (3 RW) | 90° Slew | dt=10s | `debug_0_3_slew90_dt10.py` |
| **0+3** (3 RW) | 90° Slew (long) | dt=10s | `debug_0_3_slew90_dt10_long.py` |
| **0+3** (3 RW) | 180° Slew | dt=10s | `debug_0_3_slew180_dt10.py` |
| **0+3** (3 RW) | Multi-waypoint | dt=10s | `debug_0_3_multi_dt10.py` |
| **3+1** (hybrid) | 90° Slew | dt=10s | `debug_3_1_slew90_dt10.py` |
| **3+3** (hybrid) | Detumble | dt=60s | `debug_3_3_detumble_dt60.py` |
| **3+3** (hybrid) | 90° Slew | dt=10s | `debug_3_3_slew90_dt10.py` |

### Satellite Types
- **3+0**: 3 Magnetorquers + 0 Reaction Wheels (MTQ-only)
- **0+3**: 0 Magnetorquers + 3 Reaction Wheels (RW-only)
- **3+1**: 3 Magnetorquers + 1 Reaction Wheel (Hybrid)
- **3+3**: 3 Magnetorquers + 3 Reaction Wheels (Full Hybrid)

### Mission Profiles
- **Detumble**: Remove angular velocity (zero velocity target)
- **Slew90**: 90-degree attitude slew
- **Slew180**: 180-degree attitude slew
- **Multi**: Multiple waypoint maneuver

### Timesteps
- **dt=10s**: Fine time discretization (1000s mission = 100 steps)
- **dt=60s**: Coarse time discretization (5400s mission = 90 steps)

## Adding New Tests

To add a new test case:

1. Choose the appropriate satellite config from `configs/`
2. Create a new debug file: `debug_<sat_type>_<mission>_dt<step>.py`
3. Import the satellite: `from sat_X_X_* import create_satellite`
4. Define the mission profile in `main()`

## Usage Example

```python
# Run a specific test
python tests/debug/optimizer/ilqr/debug_3_1_slew90_dt10.py

# All test files can be run directly
```
