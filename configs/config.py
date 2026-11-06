from pathlib import Path


PROJECT_ROOT = Path(__file__).parent.parent

DATA_DIR  = PROJECT_ROOT / "data"
CACHE_DIR = PROJECT_ROOT / "cache"
IMG_DIR   = PROJECT_ROOT / "img"

CHUNK_CACHE_DIR   = CACHE_DIR / "chunks"
PATTERN_CACHE_DIR = CACHE_DIR / "patterns"
FEWSHOT_CACHE_DIR = CACHE_DIR / "fewshot"
PROFILE_CACHE_DIR = CACHE_DIR / "rpr"

PROMPT_DIR = PROJECT_ROOT / "src" /"prompts"

MIN_TOKENS         = 500
CITATION_THRESHOLD = 25
SPLITS             = ["train", "test", "dev", "incoming", "extended_test", "dev_train"]

FEWSHOT_N          = 100
FEWSHOT_METHOD     = "random"   # "greedy" | "random" 
FEWSHOT_MAX_INPUT_LEN = 2000


FS_MIN_TOKENS = 100
LABEL_SCHEME = {
    "legislation": {
        "attributes": ["docid", "uri"],  
    },
    "decision": {
        "attributes": ["docid", "uri"],  
    },
    "secondary sources": {
        "attributes": ["docid", "uri"],
    },
    "title": {
        "attributes": ["titletype"], 
    },
    "citation": {
        "attributes": [],
    },
    "source": {
        "attributes": [],
    },
    "authors": {
        "attributes": [],   
    },
    "fragment": {
        "attributes": ["fragmentid", 'non_standard'],
    }
}

STRUCTURAL_LABELS = {"title", "citation", "source", "authors", "fragment"}

LABEL_SCHEME_PATH = PROJECT_ROOT / "configs" / "label_scheme.json"


# How input will be given to the LLM for the chunk and for the fewshots :
USE_SIMPLIFIED_LABELS = True # normal form : <auto_label labelname="decision" docid="123" uri="http://example.com/doc"> ... </auto_label> simplifed version : <decision docid="123" uri="http://example.com/doc"> ... </decision>
KEEP_ATRIBUTES = ["labelname"]  # Only keep labelname attribute for fewshot selection and input formatting

# For few shot greedy selection there is two types of patterns : surface and structural patterns.
GREEDY_CONFIG = {
    "surface_pattern": 1.0,
    "structural_pattern": 1.0,
}
# In LeREaD v1 we used  surface_pattern 1.0 and structural pattern 0.0.

# COREF
CONTEXT_MAX_TOKENS = 500
FEWSHOT_CONTEXT_MAX_TOKENS = 100






