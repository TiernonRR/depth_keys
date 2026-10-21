Configuration
=============

Start with :download:`example_configs/default/config.toml
<../example_configs/default/config.toml>` and keep your edited copies together.
If your data includes a cable, for example in experiments with fiber
photometry, use
:download:`example_configs/default_cable/config_cable.toml
<../example_configs/default_cable/config_cable.toml>`. We suggest skipping
``snout`` in your predictions for these recordings.

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
   * - ``proc_order``
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
       These are zero-based indices into ``nodes.toml``; keep each threshold
       associated with its keypoint when changing the node list.
   * - ``[resolve_z.depth_processing]``
     - Active depth filter options for the recording. ``bilateral_kwargs``
       controls the bilateral filter. ``replace_height_spikes_kwargs`` is
       either ``false`` or a table of spike-filter parameters.
   * - ``[resolve_z.depth_patch_parameters]``
     - One entry per node in ``nodes.toml``. Each needs ``patch_radius`` in
       pixels and ``agg_func`` as a depth percentile from 0 to 100.
   * - ``[post_processing]`` and its subtables
     - ``incl_kpoints_post_processing`` selects keypoints to post-process. The nested
       tables set temporal, bone, alignment, smoothing, and interpolation
       behavior. Start with the example values and adjust for your data.

Both example configs use the same tables. The selected config supplies the
depth filters and ordered processing stages for the recording. A stage's
parameter table takes effect only when that stage is included in
``proc_order``. For example, ``"interpolate"`` uses
``[post_processing.interpolation]``, ``"pca_impute"`` uses
``[post_processing.pca]``, and ``"final_smooth"`` uses
``[post_processing.post_align_sgolay]``.

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

If your data includes a cable, for example in fiber photometry experiments,
use :download:`config_cable.toml
<../example_configs/default_cable/config_cable.toml>` and the accompanying
:download:`nodes.toml <../example_configs/default_cable/nodes.toml>`,
:download:`skeleton.json
<../example_configs/default_cable/skeleton.json>` and
:download:`avg_transforms.toml
<../example_configs/default_cable/avg_transforms.toml>`.

We suggest skipping ``snout`` in your predictions. The cable nodes file
omits it, and the cable skeleton has no snout edges. Match your SLEAP model's
keypoints to this node list. The cable config supplies its depth filtering,
temporal regularization, and bone settings. Its ``proc_order`` applies
temporal smoothing, two bone-constraint passes, and final smoothing.
