Process a session
=================

The command takes a **session directory** containing a ``_proc`` subdirectory.
It looks for camera AVI files directly inside ``_proc``::

   session_001/
     _proc/
       camera_A.avi
       camera_B.avi

If your recordings start as ``.dat`` files, convert them with
``markolab-cli convert-dat-to-avi PATH_TO_DAT_FILE`` and place the resulting
camera AVIs in this layout. Keep the source recordings until you have checked
the converted videos.

Copy and edit the example config files for your cameras and models, then set
their paths in your shell::

   export DEPTHKEYS_CONFIG="/path/to/my_config.toml"
   export DEPTHKEYS_NODES="/path/to/my_nodes.toml"
   export DEPTHKEYS_SKELETON="/path/to/my_skeleton.json"
   export DEPTHKEYS_TRANSFORM="/path/to/my_transforms.toml"
   export DEPTHKEYS_INTRINSICS="/path/to/intrinsics.toml"
   export DEPTHKEYS_CENTROID_MODEL="/path/to/centroid_model"
   export DEPTHKEYS_CI_MODEL="/path/to/centered_instance_model"
   export DEPTHKEYS_REFERENCE_CAMERA="camera_A"

Run the stages in order. Use a GPU node for the first command::

   depth-keys compute-keypoints /path/to/session_001 --compute-2d
   depth-keys compute-keypoints /path/to/session_001 --compute-3d
   depth-keys compute-keypoints /path/to/session_001 --render

You can select all stages in one run on a GPU node::

   depth-keys compute-keypoints /path/to/session_001 --compute-2d --compute-3d --render

For cable recordings, set ``DEPTHKEYS_CONFIG`` to your cable config and add
``--cable`` to the command. Explicit CLI path options override the corresponding
environment variables. Stage flags are required; without them the command
does not process any stage.

Outputs are written beside the AVIs::

   session_001/_proc/_kpoints_v1_2d/     # per-camera .slp predictions and arrays
   session_001/_proc/_kpoints_v1_3d/     # per-camera 3D files and merged_keypoints.h5
   session_001/_proc/renders/            # matplotlib_render.mp4 and keypoints_overlay_v1.mp4

Existing 2D and per-camera 3D files are reused. Add ``--force`` when you need
to regenerate the selected stage's files, for example after changing a model
or config. Rendering reads ``merged_keypoints.h5`` and its companion
``merged_keypoints.toml`` from the 3D output directory.
