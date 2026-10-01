depth-keys
==========

``depth-keys`` turns synchronized depth-camera recordings 3D keypoints. It leverages SLEAP-NN
models for 2D tracking, then uses camera depth values to compute respective 3D predictions. Includes visualization suite for plotting 3D keypoints over time and overlaying keypoint predictions on input videos.

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
