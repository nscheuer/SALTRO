"""Check boresight dimensions in resampling for MTQ case."""
import sys
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tests" / "debug" / "optimizer" / "alilqr_python"))

from trajOpt import _resample_zero_order_hold

# Test case from debug_3_0_slew90_dt60
jtime = np.array([0.22, 0.22 + 5400/(36525 * 86400)])
qgoal = np.array([
    [np.sqrt(2)/2, np.sqrt(2)/2],
    [0.0, 0.0],           
    [0.0, 0.0],            
    [np.sqrt(2)/2, np.sqrt(2)/2]
])
boresight = np.array([
    [1.0, 1.0],
    [0.0, 0.0],
    [0.0, 0.0]
])

dt_sec = 60.0

print("BEFORE resampling:")
print(f"  qgoal shape: {qgoal.shape}")
print(f"  boresight shape: {boresight.shape}")
print(f"  jtime shape: {jtime.shape}")

jtime_fine, q_goal_fine, boresight_fine = _resample_zero_order_hold(
    jtime, qgoal, boresight, dt_sec
)

print("\nAFTER resampling:")
print(f"  jtime_fine shape: {jtime_fine.shape}")
print(f"  q_goal_fine shape: {q_goal_fine.shape}")
print(f"  boresight_fine shape: {boresight_fine.shape}")

print(f"\nExpected shapes:")
print(f"  q_goal_fine: (4, N) where N = number of timesteps")
print(f"  boresight_fine: (3, N)")

if q_goal_fine.shape[0] != 4:
    print(f"\n✗ ERROR: q_goal_fine has {q_goal_fine.shape[0]} rows, expected 4")
    
if boresight_fine.shape[0] != 3:
    print(f"\n✗ ERROR: boresight_fine has {boresight_fine.shape[0]} rows, expected 3")
else:
    print(f"\n✓ SUCCESS: Shapes are correct")
