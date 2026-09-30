Installation
============

You need Conda, Git, access to the two Git-hosted package dependencies, and a
Linux machine with an NVIDIA GPU for 2D inference. The repository's
``environment.yml`` provides Python 3.12 and FFmpeg. The package install adds
SLEAP-NN, Markovids, and the other Python dependencies.

Clone the repository and create the environment::

   git clone https://github.com/TiernonRR/depth_keys.git
   cd depth_keys
   conda env create -f environment.yml
   conda activate depth-keys
   python -m pip install -e .

For the CUDA 12.6 setup described by this project, install the matching
PyTorch wheels in the same environment::

   python -m pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu126

Check the commands before processing data::

   depth-keys --help
   depth-keys compute-keypoints --help
   markolab-cli --help
   markovids --help

The 2D stage requests CUDA in its SLEAP-NN command. Run that stage on a
GPU-enabled compute node; the 3D stage can run separately after inference.
You will also need trained centroid and centered-instance model directories,
camera intrinsics, and (for multiview registration) camera transforms. See
:doc:`configuration` for the paths and file formats.
