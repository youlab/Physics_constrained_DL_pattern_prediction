import os, glob, cv2, random
import pipeline
from cldm.preprocess import preprocess_simulation_graybackground
import argparse
from cldm.config import OUTPUT_DIR_SIMTOEXP_INFOENCODING,SIM_FOLDER_TEST_INFOENCODING_MORESEEDS

p = argparse.ArgumentParser()
p.add_argument('--specific_folder', type=str, default='simtoexp')
args = p.parse_args()


# ------------------------------
# Set up output folder
# ------------------------------

OUTPUT_DIR = OUTPUT_DIR_SIMTOEXP_INFOENCODING
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ------------------------------
# Parameters and File Paths 
# ------------------------------
INPUT_DIR      = SIM_FOLDER_TEST_INFOENCODING_MORESEEDS  # test images folder


os.makedirs(OUTPUT_DIR, exist_ok=True)

# fixed hyper-params:
TOTAL_SAMPLES  = 3   # total outputs per input image
ARGS = {
  "prompt":   "",
  "a_prompt": "",
  "n_prompt": "longbody, lowres, bad anatomy, cropped, worst quality, low quality",
  "num_samples":   TOTAL_SAMPLES,
  "image_resolution":256,
  "ddim_steps":     50,
  "guess_mode":     False,
  "strength":       1.0,
  "scale":          15.1,
  "eta":            0.0
}



# SLURM array: each task processes every num_tasks-th file (interleaved for balanced load)
# Falls back to processing all files when run outside a SLURM array
task_id   = int(os.environ.get('SLURM_ARRAY_TASK_ID',   1))
num_tasks = int(os.environ.get('SLURM_ARRAY_TASK_COUNT', 1))

all_files  = sorted(glob.glob(os.path.join(INPUT_DIR, "*.png")))[:30000]  # note normal sort 
task_files = all_files[task_id - 1::num_tasks]  # interleaved slice for this task
total      = len(task_files)
print(f"Task {task_id}/{num_tasks}: processing {total}/{len(all_files)} files", flush=True)

# Pre-generate all base seeds NOW, before pipeline.process() calls seed_everything()
# and corrupts the random state. Use os.urandom to ensure true randomness even
# when multiple tasks start simultaneously (avoids the time-based collision problem).
rng = random.Random(int.from_bytes(os.urandom(8), 'big') ^ (task_id * 0xDEADBEEF))
base_seeds = [rng.randint(0, 2**31 - 1) for _ in task_files]

for idx, (fp, base_seed) in enumerate(zip(task_files, base_seeds), start=1):
    prefix = os.path.splitext(os.path.basename(fp))[0]  # e.g. "1_1.TIF" -> "1_1", "Fixed_2_1.TIF" -> "Fixed_2_1"

    # load & convert to H×W×C RGB
    # img = cv2.cvtColor(cv2.imread(fp), cv2.COLOR_BGR2RGB)

    # process images with the simulation preprocessing step
    img= preprocess_simulation_graybackground(fp)

    # single call — 3 samples fit in one forward pass, no batching needed
    outs = pipeline.process(img, **ARGS, seed=base_seed % (2**31))

    # save as prefix_1.png … prefix_100.png
    for i, out in enumerate(outs, start=1):
        fn  = f"{prefix}_{i}.png"
        out_bgr = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(OUTPUT_DIR, fn), out_bgr)

    print(f"[task {task_id}] [{idx}/{total}] Done: {prefix}", flush=True)