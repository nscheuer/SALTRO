"""
POINTING ERROR FIX SUMMARY
==========================

PROBLEM:
--------
When viewing C++ trajOpt results using final_viewer.py, the pointing error
displayed at certain times (especially at goal waypoint boundaries like t=100s)
differed from the Python trajOpt viewer:

- At t=100s with Python: ~33° pointing error
- At t=100s with C++ viewer: ~57° pointing error  
- Difference: ~24°

ROOT CAUSE:
-----------
The optimizer internally resamples compact goal quaternions (e.g., 3 waypoints)
into a full trajectory (21 timesteps) using zero-order-hold with searchsorted
semantics. The viewer was re-expanding the compact goal independently without
knowing the actual jtime values, leading to different knot boundary behavior:

Python optimizer (correct):
  - Uses: idx = np.searchsorted(jtime[1:], t, side='right')  
  - At t=100s (exactly on boundary): idx=0, uses qgoal[:,0] (90° target)
  
C++ viewer (incorrect):
  - Assumed evenly-spaced waypoints in INDEX space
  - At t=100s: computed seg=1, used qgoal[:,1] (180° target)

SOLUTION:
---------
Modified final_viewer.py to:
1. Accept jtime parameter in plot_final_trajectory()
2. Added _resample_zero_order_hold() function matching optimizer behavior
3. Updated _expand_q_goal() to use proper time-based resampling when jtime provided
4. Modified debug_0_3_slew180_dt10.py to pass jtime to the viewer

Now the viewer uses exactly the same resampling logic as the optimizer.

VERIFICATION:
-------------
Run: tests/debug/auto_debug/verify_pointing_error_fix.py

Results:
  - Goal quaternions match at all timesteps (max diff: 0.0)
  - Pointing errors match at all times:
      t=0s:   90.00° (both)
      t=50s:   1.81° (both) 
      t=100s: 32.97° (both) ← FIXED!
      t=150s:  2.02° (both)
      t=200s:  0.47° (both)

TRAJECTORY VALIDATION:
---------------------
Run: tests/debug/auto_debug/compare_slew180_outputs.py

Confirms that Python and C++ trajOpt produce IDENTICAL trajectories:
  - Max state difference: 0.0
  - Max control difference: 0.0

The pointing error discrepancy was purely a visualization issue, not an
optimization difference.

FILES MODIFIED:
--------------
1. tests/debug/optimizer/alilqr_cpp/final_viewer.py
   - Added _resample_zero_order_hold() function
   - Modified _expand_q_goal() to accept jtime and dt parameters
   - Updated plot_final_trajectory() signature to accept jtime

2. tests/debug/optimizer/alilqr_cpp/debug_0_3_slew180_dt10.py  
   - Pass jtime to plot_final_trajectory()

FILES CREATED:
-------------
1. tests/debug/auto_debug/compare_slew180_outputs.py
   - Compares X and U outputs between Python and C++ trajOpt
   
2. tests/debug/auto_debug/compare_pointing_errors.py
   - Detailed diagnosis of goal expansion and pointing error differences
   
3. tests/debug/auto_debug/verify_pointing_error_fix.py
   - Validates that the fix resolves the discrepancy
"""
