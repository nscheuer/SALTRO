Satellite Augmented Lagrangian Trajectory Optimization (SALTRO)
===============================================================

SALTRO is a high-performance trajectory optimizer for satellite attitude control.

.. image:: _static/animate_3_0_slew90_dt10.gif
   :alt: Demo animation
   :align: center

Using trajectory planning and advance knowledge of the magnetic field, it is for instance able to slew a MTQ-only satellite within one orbit, a task that usually takes multiple.

SALTRO's predecessor flew on the MIT STAR Lab "BeaverCube 1" mission. SALTRO has a performance improvement of approximately 50x compared to the previous version.

.. list-table:: Compute times for a 3 MTQ satellite over 5400 seconds, 10 second timestep
   :header-rows: 1
   :widths: 40 30 30

   * - CPU
     - Algorithm
     - Compute Time
   * - i9-13900K
     - BC2 SALTRO
     - 1.3 s
   * - i9-13900K
     - BC1 trajectory optimizer
     - 50 s

.. toctree::
   :maxdepth: 1

   api/limits
   api/constants/index
   api/math/index
   api/optimizer/index
   api/orbit_generation/index
   api/pybind/index
   api/validation/index

