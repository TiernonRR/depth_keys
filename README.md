# Install (GT PACE)

SSH into a PACE login node:

1. First, ensure you have PACE's anaconda module loaded. The latest version (as of 26-07-23) can be loaded via

    ```bash
    module load anaconda3/2023.03
    ```

2. Next, clone this repository to a convenient directory on PACE (it's advisable to have code in your project storage not home directory)

    ```bash
    git clone https://github.com/TiernonRR/depth_keys/tree/cleanup-jm
    ```

3. Install the depth-keys conda environment, by navigating the cloned repository and running,

    ```bash
    conda env create -f environment.yml
    ```

4. Fire up a bash terminal on a GPU-enabled node in interactive mode. Now, activate the new environment,

    ```bash
    module load anaconda3/2023.03
    conda activate depth-keys
    ```

5. Install the repository by running this in the cloned repo directory,

    ```bash
    pip install -e .
    ```

6. To ensure sleap-nn is compatible with this CUDA version install the appropriate intermediate libraries

    ```bash
    pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu126
    ```

7. Run the following commands to ensure everything is working properly,

    ```bash
    markolab-cli --help
    markovids --help
    depth-keys --help
    ```

8. Now you will need to set some environment variables to point the code to the right things, add them to bashrc so they're loaded on compute nodes. Replace vim with your editor of choice (e.g. nano)

    ```bash
    vim ~/.bashrc
    ```

9. Point the following variables to appropriate places:

    ```
    # DEFAULT CONFIG, IN THE GITHUB REPO
    REPO_PATH=<path_to_your_repository>
    export DEPTHKEYS_CONFIG="${REPO_PATH}/example_configs/default/config.toml"
    export DEPTHKEYS_SKELETON="${REPO_PATH}/example_configs/default/skeleton.json"
    export DEPTHKEYS_TRANSFORM="${REPO_PATH}/example_configs/default/avg_transforms.toml"
    export DEPTHKEYS_NODES="${REPO_PATH}$/example_configs/default/nodes.toml"
    export DEPTHKEYS_INTRINSICS="<path_to_instrinsics_toml_from_cammy>"
    export DEPTHKEYS_CENTROID_MODEL="<path_to_centroid_model>"
    export DEPTHKEYS_CI_MODEL="<path_to_kp_model>"
    # CAMERA ID, APPENDED BY CAMMY TO RECORDED DATA BY DEFAULT, e.g. Lucid Vision Labs-HTP003S-001-224500508
    export DEPTHKEYS_REFERENCE_CAMERA="<camera_id>"
    ```

10. Reload your bash terminal and try running some basic commands.

# Process a directory

1. To process a directory, first convert your raw .dat file to .avi . 

    ```
    markolab-cli convert-dat-to-avi PATH_TO_DAT_FILE
    ```

    It is recommended to run without deleting the raw data until you are comfortable with all commands. If you wish to delete the original file after processing.

    ```
    markolab-cli convert-dat-to-avi --delete PATH_TO_DAT_FILE
    ```

    Note that the function will validate all data prior to deleting the original .dat file.

2. Once all .dat files have been converted, you can estimate 2d keypoints. Run this command from a GPU-enabled compute node.

    ```
    depth-keys compute-keypoints --compute-2d DIR_WITH_AVI_FILE
    ```

3. To convert the 3D keypoints to 3D, you can run the following on any node. 

    ```
    depth-keys compute-keypoints --compute-3d DIR_WITH_AVI_FILE
    ```
  
4. Finally, to render the output, run...

    ```
    depth-keys compute-keypoints --render DIR_WITH_AVI_FILE
    ```

5. Steps can be chained, e.g.,

    ```
    depth-keys compute-keypoints --compute-2d --compute-3d --render DIR_WITH_AVI_FILE
    ```