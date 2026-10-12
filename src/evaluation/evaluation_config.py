"""
Evaluation Configuration

Defines parameters for evaluation runs, can be extended/customized for specific experiments.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Literal


@dataclass
class EvaluationConfig:
    """Configuration for evaluation runs."""
    
    # Required: Where to find evaluation outputs
    output_dir: Path = Path("./output")
    
    # Optional: Ground truth split (required for single file evaluation)
    # For batch evaluation, can be extracted from folder name
    split: Optional[str] = None
    
    # Ground truth data directory
    data_dir: Path = Path("./data")
    
    # Evaluation mode: "single" (one file) or "batch" (folder)
    mode: Literal["single", "batch"] = "batch"
    
    # Single mode only: path to the system output HTML file
    system_file: Optional[str] = None
    
    # Batch mode only: path to the folder containing outputs
    batch_folder: Optional[str] = None
    
    # Verbose output per file
    verbose_per_file: bool = False
    
    # Context characters for span matching
    context_chars: int = 200

    # Overlap matching as an extra comparison (exact match stays the main metric)
    overlap: bool = True
    
    def get_ground_truth_dir(self) -> Path:
        """Get the ground truth directory for the configured split."""
        if not self.split:
            raise ValueError("split must be set to get ground truth directory")
        return self.data_dir / "annotated" / self.split
    
    def validate(self) -> None:
        """Validate configuration for the configured mode."""
        if self.mode == "single":
            if not self.system_file:
                raise ValueError("mode='single' requires system_file parameter")
            if not self.split:
                raise ValueError("mode='single' requires split parameter")
        elif self.mode == "batch":
            if not self.batch_folder:
                raise ValueError("mode='batch' requires batch_folder parameter")
        else:
            raise ValueError(f"Invalid mode: {self.mode}")


# Example configurations for different experiments
# You can save these to YAML files or import them directly

EXAMPLE_CONFIG_SINGLE = EvaluationConfig(
    mode="single",
    split="test",
    system_file="./output/1989CanLII1415ONCA_processed.html",
    verbose_per_file=True,
)

EXAMPLE_CONFIG_BATCH = EvaluationConfig(
    mode="batch",
    batch_folder="./output/test_gpt-5.2_DEC_fs6_greedy_sentence/",
    verbose_per_file=False,
)
