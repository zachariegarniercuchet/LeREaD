#!/bin/bash
# Coref inference on test with the Qwen3.8-27B coref adapters (silver,
# train+silver), gold + pipeline mode, one run per GPU:
#   GPU 0: silver        gold
#   GPU 1: silver        pipeline
#   GPU 2: train+silver  gold
#   GPU 3: train+silver  pipeline
# Reruns resume per document (<doc>_coref_mapping.json), so if the time limit
# hits, just sbatch this script again.
#
#   sbatch scripts/run_coref_lora_inference.sh
#SBATCH --job-name=coref_infer
#SBATCH --account=aip-azouaq
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --output=jobs/%x-%j.out
#SBATCH --error=jobs/%x-%j.err

source ~/myenv/bin/activate
module load cuda/12.9
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Starting job at $(date) on $(hostname), job $SLURM_JOB_ID"
nvidia-smi
python -c "import fla, causal_conv1d; print('fast linear-attention kernels: OK')" \
  || echo "WARNING: fla / causal_conv1d missing -> slow PyTorch fallback"

ADAPTERS=$HOME/links/scratch/output_training
declare -A EXT=(
  [silver]="output/test_qwen38-27b-TrainedOnSilverV2-all_DEC_fs6_greedy_paragraph_long"
  [train+silver]="output/test_qwen38-27b-TrainedOnTrainSilverV2-all_DEC_fs6_greedy_paragraph_long"
)

pids=()
gpu=0
for c in silver train+silver; do
  A=$ADAPTERS/Qwen3.8-27B_${c}_coref_r16_a32_lr0.0001_ep2_q4bit_s42
  [ -d "$A" ] || { echo "missing adapter $A"; exit 1; }
  [ -d "${EXT[$c]}" ] || { echo "missing extraction dir ${EXT[$c]}"; exit 1; }

  CUDA_VISIBLE_DEVICES=$gpu python -u -m src.main_coref --split test \
    --input-source gold \
    --model qwen38-27b --adapter-path "$A" --temperature 0 \
    > "jobs/coref-infer-gold-$c-$SLURM_JOB_ID.log" 2>&1 &
  pids+=($!); gpu=$((gpu+1))
  sleep 60   # stagger model loading (shared filesystem)

  CUDA_VISIBLE_DEVICES=$gpu python -u -m src.main_coref --split test \
    --input-source extraction --extraction-dir "${EXT[$c]}" --extraction-stage final \
    --model qwen38-27b --adapter-path "$A" --temperature 0 \
    > "jobs/coref-infer-pipe-$c-$SLURM_JOB_ID.log" 2>&1 &
  pids+=($!); gpu=$((gpu+1))
  sleep 60
done

# Without this the job ends now and Slurm kills every run above.
fail=0
for p in "${pids[@]}"; do wait "$p" || { echo "FAILED pid=$p"; fail=1; }; done
echo "Job finished at $(date)"
exit $fail
