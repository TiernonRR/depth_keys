Configuration
=============

Start with :download:`example_configs/default/config.toml
<../example_configs/default/config.toml>` and keep your edited copies together.
There is a second starting point for recordings with a cable:
:download:`example_configs/default_cable/config_cable.toml
<../example_configs/default_cable/config_cable.toml>`.

Files to provide
----------------

.. list-table::
   :header-rows: 1
   :widths: 22 36 42

   * - CLI option / environment variable
     - Example file
     - What it supplies
   * - ``--config-path`` / ``DEPTHKEYS_CONFIG``
     - :download:`config.toml <../example_configs/default/config.toml>`
     - Experiment, depth conversion, and registration settings in TOML.
   * - ``--node-path`` / ``DEPTHKEYS_NODES``
     - :download:`nodes.toml <../example_configs/default/nodes.toml>`
     - Ordered SLEAP keypoint names. Their order sets the output array order.
   * - ``--skeleton-path`` / ``DEPTHKEYS_SKELETON``
     - :download:`skeleton.json <../example_configs/default/skeleton.json>`
     - Pairs of node names joined in the 3D render.
   * - ``--transform-path`` / ``DEPTHKEYS_TRANSFORM``
     - :download:`avg_transforms.toml <../example_configs/default/avg_transforms.toml>`
     - Camera-to-reference-camera rigid transforms for multiview registration.
   * - ``--intrinsics-path`` / ``DEPTHKEYS_INTRINSICS``
     - Your camera calibration TOML
     - Intrinsic matrices and distortion coefficients for the recorded cameras.
   * - ``--centroid-model-path`` / ``DEPTHKEYS_CENTROID_MODEL``
     - Your trained centroid model directory
     - SLEAP-NN centroid model for 2D inference.
   * - ``--ci-model-path`` / ``DEPTHKEYS_CI_MODEL``
     - Your trained centered-instance model directory
     - SLEAP-NN instance model for 2D inference.

The model directories and intrinsics file are specific to your recording
setup; they are not supplied by the example configs. The example transforms
contain camera IDs and calibration values for one rig. Replace them with
transforms measured for your cameras. Set ``--reference-camera`` (or
``DEPTHKEYS_REFERENCE_CAMERA``) to the exact camera ID used in the AVI filename,
intrinsics, config, and transforms.

Format of ``config.toml``
-------------------------

Use TOML tables and arrays, following the supplied example. The main fields
are:

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Field or table
     - How to set it
   * - ``fps``, ``reference_camera``
     - Recording frame rate and the exact reference camera ID.
   * - ``incl_kpoints_fit_transform``
     - Stable body points to use when fitting camera registration.
   * - ``proc_order`` (default config)
     - Ordered post-processing stages such as ``"temporal"``, ``"bone"``, and
       ``"interpolate"``. The example repeats ``"bone"`` after interpolation.
   * - ``noisy_keypoints``, ``plt_kpoints``
     - Names used for smoothing settings and plotting. Use names from
       ``nodes.toml``.
   * - ``skeleton``
     - TOML array of ``[start_node, end_node, unique_bone_name]`` entries for
       bone constraints. This is separate from ``skeleton.json``.
   * - ``[smoothing_params]``, ``[hampel_params]``,
       ``[renderer_kwargs]``, ``[index_conf_map]``
     - Processing and registration settings shown in the example. Quote
       numeric keys in ``[index_conf_map]`` because TOML keys are strings.
   * - ``[resolve_z.depth_processing]`` and
       ``[resolve_z.depth_processing_cable]``
     - Depth filter options. ``--cable`` selects the cable table.
   * - ``[resolve_z.depth_patch_parameters]``
     - One entry per node in ``nodes.toml``. Each needs ``patch_radius`` in
       pixels and ``agg_func`` as a depth percentile from 0 to 100.
   * - ``[post_processing]`` and its subtables
     - ``incl_kpoints_post_processing`` selects retained keypoints. The nested
       tables set temporal, bone, alignment, smoothing, and interpolation
       behavior. Start with the example values and adjust for your data.

For example, a depth patch entry is an inline TOML table::

   [resolve_z.depth_patch_parameters]
   tail_tip = { patch_radius = 7, agg_func = 95 }

The 3D conversion reads this table by node name and raises an error if a
listed node has no patch settings. Keep node names consistent across the
SLEAP model, ``nodes.toml``, the config, and ``skeleton.json``.

``nodes.toml`` has a single ordered array::

   nodes = ["tail_tip", "tail_middle", "tail_base"]

``skeleton.json`` contains pairs, without the bone names used in the TOML
config::

   [["tail_tip", "tail_middle"], ["tail_middle", "tail_base"]]

Each transform entry in ``avg_transforms.toml`` maps a camera/reference pair
to a 3-by-3 rotation matrix and a three-value translation vector. The
reference camera has an identity transform. Use the supplied file as a shape
example, with your own camera IDs and calibrated values.

Cable recordings
----------------

For cable recordings, use ``config_cable.toml`` with ``--cable``. That example
keeps its depth settings under ``[post_processing.depth_processing]``,
``[post_processing.depth_processing_cable]``, and
``[post_processing.depth_patch_parameters]``. The converter accepts this
layout as well as the ``[resolve_z.*]`` layout in the default config.
The cable example also has the top-level switches ``constrain_bones``,
``impute_pca``, and ``regularize_temporal`` in place of ``proc_order``.

The cable directory supplies its own :download:`skeleton.json
<../example_configs/default_cable/skeleton.json>` and
:download:`avg_transforms.toml
<../example_configs/default_cable/avg_transforms.toml>`, but no
``nodes.toml``. Use the default nodes file only if its names and order match
your SLEAP model; otherwise write your own. Check that every edge in the
skeleton JSON names a node in that list, especially if you remove ``snout``
from a cable model.
