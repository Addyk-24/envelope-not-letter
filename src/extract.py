import os, json
import numpy as np
import torch
from nnsight import LanguageModel
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from scipy.stats import wilcoxon

LAYER     = 16
MODEL_ID  = "allenai/Olmo-3-7B-Instruct"
RUN_EVALS = ["L1", "L2", "L3"]

os.makedirs("activations", exist_ok=True)
os.makedirs("data/contrast", exist_ok=True)

print("=" * 70); print("STAGE 0: SETUP"); print("=" * 70)

model = LanguageModel(MODEL_ID, device_map="auto", torch_dtype=torch.float16)
if model.tokenizer.pad_token is None:
    model.tokenizer.pad_token = model.tokenizer.eos_token
model.tokenizer.padding_side = "left"


def get_acts(texts, batch=1):
    chunks = []
    for i in range(0, len(texts), batch):
        with model.trace(texts[i:i + batch]):
            h = model.model.layers[LAYER].output
            a = h[:, -1, :].save()
        t = a.value if hasattr(a, "value") else a
        chunks.append(t.detach().float().cpu().numpy())
    return np.concatenate(chunks, axis=0)


def norm_acts(X):
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)


def extract_or_load(texts, path):
    if os.path.exists(path):
        X = np.load(path)
        if X.shape[0] == len(texts):
            print(f"  cached  {path}  {X.shape}")
            return X
        print(f"  stale cache ({X.shape[0]} vs {len(texts)}) -> re-extracting")
    print(f"  extract {path}  n={len(texts)}")
    X = get_acts(texts)
    np.save(path, X)
    return X


_t = get_acts(["hello world", "bonjour le monde"])
assert _t.shape[0] == 2, f"bad shape {_t.shape}"
assert not np.allclose(_t[0], _t[1]), "identical rows — extraction is broken"
print(f"  smoke test OK, d_model={_t.shape[1]}")
