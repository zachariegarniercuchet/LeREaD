# =============================================================================
# Run configuration
# =============================================================================

from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path

@dataclass
class CorefRunConfig:
    """Sibling of RunConfig, scoped to the coref-resolution pipeline."""

    mode: str  # "single" | "split"
    filename: Optional[str]
    split: Optional[str]
    single_split: str  # split to use for gold lookups when mode == "single"

    input_source: str  # "gold" | "extraction"
    data_dir: Path
    variant: str
    extraction_dir: Optional[Path]
    extraction_stage: str  # "processed" | "final" | "0" | "1" | "2" | "3"

    fewshot_method: str
    fewshot_examples: int
    seed: int

    model: str
    temperature: float
    prompt_type: str

    max_fragments: int
    profile_attributes: List[str]

    output_dir: Path
    with_timestamp: bool

    encoder_path: Optional[Path] = None
    encoder_train_split: Optional[str] = None
    encoder_seed: Optional[int] = None
    encoder_epochs: Optional[int] = None
    encoder_best_epoch: Optional[int] = None

    def resolve_input_path(self, filename: str) -> Path:
        if self.input_source == "gold":
            split = self.split if self.mode == "split" else self.single_split
            return Path(self.data_dir) / self.variant / split / f"{filename}.html"

        # input_source == "extraction": read a checkpoint produced by
        # main_extraction.py. That script writes either
        # <exp_dir>/<filename>/<filename>_<stage>.html (split mode) or
        # <exp_dir>/<filename>_<stage>.html (single mode).
        suffix = f"_{self.extraction_stage}"
        nested = self.extraction_dir / filename / f"{filename}{suffix}.html"
        flat = self.extraction_dir / f"{filename}{suffix}.html"
        return nested if nested.exists() else flat

    def get_filenames(self) -> List[str]:
        if self.mode == "single":
            return [self.filename]

        if self.input_source == "gold":
            folder = Path(self.data_dir) / self.variant / self.split
            return sorted(p.stem for p in folder.glob("*.html"))

        # split mode, input_source == "extraction": one subfolder per file,
        # as produced by main_extraction.py's split mode.
        return sorted(p.name for p in self.extraction_dir.iterdir() if p.is_dir())

    def run_tag(self) -> str:
        eval_split = self.split or self.single_split
        if self.encoder_path is not None:
            tag = f"{eval_split}_{self.model}_coref_encoder"
            if self.encoder_train_split:
                tag += f"_trained-{self.encoder_train_split}"
            if self.encoder_seed is not None:
                tag += f"_seed{self.encoder_seed}"
            if self.encoder_epochs is not None:
                tag += f"_ep{self.encoder_epochs}"
            if self.encoder_best_epoch is not None:
                tag += f"_best{self.encoder_best_epoch}"
        else:
            tag = f"{eval_split}_{self.model}_coref_fs{self.fewshot_examples}_{self.fewshot_method}"
        if self.input_source == "extraction":
            tag += f"_from-{self.extraction_dir.name}-{self.extraction_stage}"
        else:
            tag += "_gold"
        return tag

    def output_root(self) -> Path:
        root = Path(self.output_dir)
        if self.with_timestamp:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            root = root / f"{self.run_tag()}_{stamp}"
        else:
            root = root / self.run_tag()
        root.mkdir(parents=True, exist_ok=True)
        return root