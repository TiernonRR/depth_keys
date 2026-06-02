"""
Preprocessing script for depth images.
Loads a specific video file, applies morphological hole-filling, 
and saves the result using a custom FFmpeg writer (AviWriter).
params are dynamically matched to the input video.
"""

import argparse
import os
import sys
import subprocess
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm

# Add custom module path
sys.path.append("/storage/home/hcoda1/3/triesenmy3/r-jmarkowitz30-0/markovids/src")
try:
    from markovids.vid import util, io
except ImportError:
    print("Could not import markovids. Ensure the path is correct.")
    sys.exit(1)

# Configuration constants
FILL_KERNEL_SIZE = (10, 10)
FILL_ITERATIONS = 4
BATCH_SIZE = 1000 
FFMPEG_PREPEND = "source ~/conda_activate ; conda activate ffmpeg"

class AviWriter:
    def __init__(
        self,
        filepath,
        frame_size=(640, 480),
        dtype=np.dtype("<u2"),
        fps=100,
        pixel_format="gray16le",
        codec="ffv1",
        threads=6,
        slices=25,
        slicecrc=1,
        prepend_args=None
    ):
        ext = os.path.splitext(filepath)[1]
        if ext != ".avi":
            raise RuntimeError("Must use avi container (extension must be avi)")
        self.filepath = filepath
        self.fps = fps
        self.pixel_format = pixel_format
        self.codec = codec
        self.threads = threads
        self.slices = slices
        self.slicecrc = slicecrc
        self.frame_size = frame_size
        self.dtype = dtype
        self.pipe = None
        self.prepend_args = prepend_args

    def open(self):
        command = [
            "ffmpeg",
            "-y",
            "-loglevel",
            "fatal",
            "-framerate",
            str(self.fps),
            "-f",
            "rawvideo",
            "-s",
            "{:d}x{:d}".format(*self.frame_size),
            "-pix_fmt",
            self.pixel_format,
            "-i",
            "-",
            "-an",
            "-vcodec",
            self.codec,
            "-threads",
            str(self.threads),
            "-slices",
            str(self.slices),
            "-slicecrc",
            str(self.slicecrc),
            "-r",
            str(self.fps),
            f"'{self.filepath}'",
        ]
        
        full_cmd = " ".join(command)
        if self.prepend_args is not None:
            full_cmd = f"{self.prepend_args} ; {full_cmd}"

        self.pipe = subprocess.Popen(full_cmd, shell=True, stdin=subprocess.PIPE, stderr=subprocess.STDOUT, executable='/bin/bash')

    def write_frames(self, frames, progress_bar=True):
        if self.pipe is None:
            self.open()
        
        iterator = range(len(frames))
        if progress_bar:
            iterator = tqdm(iterator, leave=False, disable=not progress_bar)
            
        for i in iterator:
            # Ensure data is written as bytes corresponding to the configured dtype
            self.pipe.stdin.write(frames[i].astype(self.dtype).tobytes())

    def close(self):
        if self.pipe is not None:
            self.pipe.stdin.close()
            self.pipe.wait()
        return None

def preprocess_batch(frames, fill_kernel):
    """
    Apply hole-filling preprocessing to a batch of frames.
    """
    processed = frames.copy()
    
    # If frames are (N, H, W, 1), squeeze to (N, H, W) for cv2 processing
    if processed.ndim == 4 and processed.shape[-1] == 1:
        processed = processed.squeeze(-1)
        
    for idx in range(len(processed)):
        processed[idx] = util.fill_holes(
            processed[idx], 
            fill_kernel=fill_kernel, 
            iterations=FILL_ITERATIONS
        )
    
    return processed

def process_video(input_path, output_path):
    """
    Reads input video, fills holes, and writes to output path using AviWriter.
    Dynamically grabs parameters (fps, resolution, pixel_format, dtype) from input.
    """
    print(f"Processing: {input_path}")
    print(f"Output target: {output_path}")
    
    # 1. Setup Video Reader
    avi_reader = io.AviReader(
        input_path, 
        prepend_args=FFMPEG_PREPEND
    )
    # This populates frame_size, pixel_format, fps, bit_depth, dtype, nframes
    avi_reader.get_file_info() 
    
    total_frames = avi_reader.nframes
    
    if total_frames == 0:
        print(f"Error: {input_path} contains 0 frames.")
        return False

    # Extract parameters from Reader to match exactly in Writer
    # Note: AviReader stores frame_size as (width, height)
    r_width, r_height = avi_reader.frame_size
    r_fps = avi_reader.fps
    r_pix_fmt = avi_reader.pixel_format
    r_dtype = avi_reader.dtype

    print(f"  > Parameters: {r_width}x{r_height} | {r_fps} FPS | {r_pix_fmt} | {r_dtype}")

    # Ensure output directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # 2. Setup Video Writer (AviWriter) with matched parameters
    writer = AviWriter(
        filepath=output_path,
        frame_size=(r_width, r_height), 
        dtype=r_dtype,     
        fps=r_fps,
        pixel_format=r_pix_fmt,  
        codec="ffv1", # Lossless codec is standard for this pipeline
        prepend_args=FFMPEG_PREPEND
    )
    
    # Pre-calculate kernel once
    fill_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, FILL_KERNEL_SIZE)

    # 3. Process in batches
    for start_idx in tqdm(range(0, total_frames, BATCH_SIZE), desc=f"Processing {Path(input_path).name}"):
        end_idx = min(start_idx + BATCH_SIZE - 1, total_frames - 1)
        frame_indices = np.arange(start_idx, end_idx + 1)
        
        try:
            # Load frames
            frames = avi_reader.get_frames(frame_indices)
            
            # Fill holes
            processed_frames = preprocess_batch(frames, fill_kernel)
            
            # Safety check: ensure dtype matches what the writer expects
            # (Hole filling might implicitly cast, so we explicitly cast back)
            if processed_frames.dtype != r_dtype:
                processed_frames = processed_frames.astype(r_dtype)

            # Write frames
            writer.write_frames(processed_frames, progress_bar=False)
                
        except Exception as e:
            print(f"Error processing batch {start_idx}-{end_idx}: {e}")
            writer.close()
            return False

    writer.close()
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess videos with hole filling.")
    
    # Path construction arguments
    parser.add_argument("--base_dir", type=str, required=True, help="Root directory containing the project folders")
    parser.add_argument("--temp_dir", type=str, required=True, help="Temporary root directory for output")
    parser.add_argument("--project", type=str, required=True, help="Project name")
    parser.add_argument("--session", type=str, required=True, help="Session name")
    parser.add_argument("--camera", type=str, required=True, help="Camera name (acts as filename, e.g. 'cam1')")
    
    args = parser.parse_args()
    
    # Construct input file path: base_dir/project/session/_proc/camera.avi
    input_video_path = Path(args.base_dir) / args.project / args.session / "_proc" / f"{args.camera}.avi"
    
    # Construct output file path: temp_dir/project/session/camera.avi
    output_video_path = Path(args.temp_dir) / args.project / args.session / f"{args.camera}.avi"
    
    if not input_video_path.exists():
        print(f"Error: Input file does not exist: {input_video_path}")
        sys.exit(1)

    process_video(str(input_video_path), str(output_video_path))