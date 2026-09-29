depth-keys
==========

``depth-keys`` turns synchronized depth-camera recordings into 2D keypoints,
registered 3D keypoints, and videos for checking the results. It uses SLEAP-NN
models for tracking and camera calibration data for 3D registration.

Start with :doc:`installation`, then :doc:`configuration` and
:doc:`processing`. If you process many sessions on a Slurm cluster, see
:doc:`batch`.

.. toctree::
   :maxdepth: 2
   :caption: Setup

   installation
   configuration

.. toctree::
   :maxdepth: 2
   :caption: Use

   processing
   batch
