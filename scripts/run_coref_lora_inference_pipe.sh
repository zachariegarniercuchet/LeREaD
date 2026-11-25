#!/bin/bash
# Coref inference on test, PIPELINE mode (mentions from the fine-tuned
# extraction runs), Qwen3.8-27B coref adapters.
#
#   sbatch scripts/run_coref_lora_inference_pipe.sh
#
#   GPU 0: silver        docs 1-19     GPU 1: silver        docs 20-38
#   GPU 2: train+silver  docs 1-19     GPU 3: train+silver  docs 20-38
#
# Each coref adapter reads the extraction run trained on the same data.
# Output: output_coref/test_coref_lora_<adapter>_from-<extraction run>-final/
# Finished documents (<doc>_coref_mapping.json) are skipped: resubmit to continue.
#SBATCH --job-name=coref_infer_pipe
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

echo "Starting PIPELINE job at $(date) on $(hostname), job $SLURM_JOB_ID"
nvidia-smi
python -c "import fla, causal_conv1d; print('fast linear-attention kernels: OK')" \
  || echo "WARNING: fla / causal_conv1d missing -> slow PyTorch fallback"

ADAPTERS=$HOME/links/scratch/output_training
SILVER=$ADAPTERS/Qwen3.8-27B_silver_coref_r16_a32_lr0.0001_ep2_q4bit_s42
TRAINSILVER=$ADAPTERS/Qwen3.8-27B_train+silver_coref_r16_a32_lr0.0001_ep2_q4bit_s42
EXT_SILVER=output/test_qwen38-27b-TrainedOnSilverV2-all_DEC_fs6_greedy_paragraph_long
EXT_TRAINSILVER=output/test_qwen38-27b-TrainedOnTrainSilverV2-all_DEC_fs6_greedy_paragraph_long
for d in "$SILVER" "$TRAINSILVER" "$EXT_SILVER" "$EXT_TRAINSILVER"; do
  [ -d "$d" ] || { echo "missing folder $d"; exit 1; }
done

COMMON="--split test --input-source extraction --extraction-stage final --model qwen38-27b --temperature 0"

CUDA_VISIBLE_DEVICES=0 python -u -m src.main_coref $COMMON --extraction-dir "$EXT_SILVER" \
  --adapter-path "$SILVER" --start 1 --end 19 \
  > jobs/coref-infer-pipe-silver-docs1-19-$SLURM_JOB_ID.log 2>&1 & p1=$!
sleep 60
CUDA_VISIBLE_DEVICES=1 python -u -m src.main_coref $COMMON --extraction-dir "$EXT_SILVER" \
  --adapter-path "$SILVER" --start 20 --end 38 \
  > jobs/coref-infer-pipe-silver-docs20-38-$SLURM_JOB_ID.log 2>&1 & p2=$!
sleep 60
CUDA_VISIBLE_DEVICES=2 python -u -m src.main_coref $COMMON --extraction-dir "$EXT_TRAINSILVER" \
  --adapter-path "$TRAINSILVER" --start 1 --end 19 \
  > jobs/coref-infer-pipe-trainsilver-docs1-19-$SLURM_JOB_ID.log 2>&1 & p3=$!
sleep 60
CUDA_VISIBLE_DEVICES=3 python -u -m src.main_coref $COMMON --extraction-dir "$EXT_TRAINSILVER" \
  --adapter-path "$TRAINSILVER" --start 20 --end 38 \
  > jobs/coref-infer-pipe-trainsilver-docs20-38-$SLURM_JOB_ID.log 2>&1 & p4=$!

# Without this the job ends now and Slurm kills every run above.
fail=0
for p in $p1 $p2 $p3 $p4; do wait "$p" || { echo "FAILED pid=$p"; fail=1; }; done
echo "Job finished at $(date)"
exit $fail
