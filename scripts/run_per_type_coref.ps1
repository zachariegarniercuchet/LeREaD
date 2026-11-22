$ErrorActionPreference = "Stop"
$root = "C:\Users\zakga\OneDrive\Documents\code\LeREaD"
Set-Location $root

$folders = @(
    "output_coref/1-final/gpt5.2_gold",
    "output_coref/1-final/gpt5.2_pip",
    "output_coref/1-final/gemma4_gold",
    "output_coref/1-final/gemma4_pip",
    "output_coref/test_deberta-silver_ep5_best4_coref_from-AIO_fs6_greedy_paragraph_long-processed",
    "output_coref/test_deberta-v3-large_coref_encoder_deberta-v3-large_coref_silver_lr2e-05_hlr0.001_ep5_hn5_trained-silver_ep5_best4_gold",
    "output_coref/test_baseline_coref_strong_from-test_baseline_rules-final",
    "output_coref/test_baseline_coref_strong_gold"
)

foreach ($f in $folders) {
    python -m src.evaluation.evaluate_coref_per_type batch --batch-folder $f --split test --match-mode span --unmatched singleton
}
