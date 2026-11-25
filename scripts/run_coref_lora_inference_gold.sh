#!/bin/bash
# Coref inference on test, GOLD mentions, Qwen3.8-27B coref adapters.
#
#   sbatch scripts/run_coref_lora_inference_gold.sh
#
#   GPU 0: silver        docs 1-19     GPU 1: silver        docs 20-38
#   GPU 2: train+silver  docs 1-19     GPU 3: train+silver  docs 20-38
#
# Output: output_coref/test_coref_lora_<adapter>_gold/
# Finished documents (<doc>_coref_mapping.json) are skipped: resubmit to continue.
#SBATCH --job-name=coref_infer_gold
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

echo "Starting GOLD job at $(date) on $(hostname), job $SLURM_JOB_ID"
nvidia-smi
python -c "import fla, causal_conv1d; print('fast linear-attention kernels: OK')" \
  || echo "WARNING: fla / causal_conv1d missing -> slow PyTorch fallback"

ADAPTERS=$HOME/links/scratch/output_training
SILVER=$ADAPTERS/Qwen3.8-27B_silver_coref_r16_a32_lr0.0001_ep2_q4bit_s42
TRAINSILVER=$ADAPTERS/Qwen3.8-27B_train+silver_coref_r16_a32_lr0.0001_ep2_q4bit_s42
for d in "$SILVER" "$TRAINSILVER"; do
  [ -d "$d" ] || { echo "missing adapter $d"; exit 1; }
done

COMMON="--split test --input-source gold --model qwen38-27b --temperature 0"

CUDA_VISIBLE_DEVICES=0 python -u -m src.main_coref $COMMON --adapter-path "$SILVER" --start 1 --end 19 \
  > jobs/coref-infer-gold-silver-docs1-19-$SLURM_JOB_ID.log 2>&1 & p1=$!
sleep 60
CUDA_VISIBLE_DEVICES=1 python -u -m src.main_coref $COMMON --adapter-path "$SILVER" --start 20 --end 38 \
  > jobs/coref-infer-gold-silver-docs20-38-$SLURM_JOB_ID.log 2>&1 & p2=$!
sleep 60
CUDA_VISIBLE_DEVICES=2 python -u -m src.main_coref $COMMON --adapter-path "$TRAINSILVER" --start 1 --end 19 \
  > jobs/coref-infer-gold-trainsilver-docs1-19-$SLURM_JOB_ID.log 2>&1 & p3=$!
sleep 60
CUDA_VISIBLE_DEVICES=3 python -u -m src.main_coref $COMMON --adapter-path "$TRAINSILVER" --start 20 --end 38 \
  > jobs/coref-infer-gold-trainsilver-docs20-38-$SLURM_JOB_ID.log 2>&1 & p4=$!

# Without this the job ends now and Slurm kills every run above.
fail=0
for p in $p1 $p2 $p3 $p4; do wait "$p" || { echo "FAILED pid=$p"; fail=1; }; done
echo "Job finished at $(date)"
exit $fail
