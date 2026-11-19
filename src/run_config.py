# run_config.py
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Literal
from configs.config import DATA_DIR, SPLITS
from datetime import datetime

@dataclass
class RunConfig:
    mode: Literal["single", "split"]
    filename: Optional[str] = None
    split: Optional[str] = None

    method: str = "AIO"
    chunker: str = "paragraph"
    fewshot_method: str = "greedy"
    fewshot_examples: int = 6
    model: str = "gpt-5.2"
    temperature: float = 1.0
    disable_fallback: bool = False
    with_timestamp: bool = False

    data_dir: Path = Path(DATA_DIR)       # dataset root
    variant: str = "original"             # "original" | "annotated"
    output_dir: Path = Path("./output")
    prompt_type: str = "long"
    # Extra folder-name component, e.g. "trained-DECall". Keeps adapter runs
    # from reusing (and silently checkpoint-skipping) a baseline run folder.
    run_tag: str = ""

    def resolve_split_dir(self, split: str) -> Path:
        """data/original/train  — or whatever data_dir/variant/split resolves to."""
        return self.data_dir / self.variant / split

    def resolve_input_path(self, filename: str) -> Path:
        """Locate a file by scanning all known split folders."""
        if self.mode == "split":
            return self.resolve_split_dir(self.split) / f"{filename}.html"

        # Single-file mode: check explicit data_dir first, then scan splits
        if self.data_dir != Path(DATA_DIR):
            # User passed --data-dir explicitly, trust it
            candidate = self.data_dir / self.variant / f"{filename}.html"
            if candidate.exists():
                return candidate

        # Auto-scan all splits
        for split in SPLITS:
            candidate = self.data_dir / self.variant / split / f"{filename}.html"
            if candidate.exists():
                print(f"  [Info] Found '{filename}' in split '{split}'")
                self.split = split  # Update split for downstream logic
                return candidate

        searched = [str(self.data_dir / self.variant / s) for s in SPLITS]
        raise FileNotFoundError(
            f"'{filename}.html' not found in any split folder:\n" +
            "\n".join(f"  - {p}" for p in searched)
        )
        

    def get_filenames(self) -> list[str]:
        if self.mode == "single":
            return [self.filename]
        split_dir = self.resolve_split_dir(self.split)
        return sorted(p.stem for p in split_dir.glob("*.html"))

    def output_root(self) -> Path:
        tag = f"-{self.run_tag}" if self.run_tag else ""
        if self.mode == "split":
            folder = f"{self.split}_{self.model}{tag}_{self.method}_fs{self.fewshot_examples}_{self.fewshot_method}_{self.chunker}_{self.prompt_type}"
        else:
            folder = f"single_{self.filename}_{self.model}{tag}_{self.method}_fs{self.fewshot_examples}_{self.fewshot_method}_{self.chunker}_{self.prompt_type}"
        if self.with_timestamp:
            folder += f"_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        root = self.output_dir / folder
        root.mkdir(parents=True, exist_ok=True)
        return root