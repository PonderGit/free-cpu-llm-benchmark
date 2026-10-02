from pathlib import Path
from huggingface_hub import hf_hub_download, list_repo_files

REPO_ID = "Qwen/Qwen2.5-1.5B-Instruct-GGUF"

files = list_repo_files(REPO_ID)
candidates = [
    f for f in files
    if f.lower().endswith(".gguf") and "q4_k_m" in f.lower()
]

if not candidates:
    raise RuntimeError("No Q4_K_M GGUF file found in repository")

filename = sorted(candidates)[0]
path = hf_hub_download(repo_id=REPO_ID, filename=filename)

Path("model_path.txt").write_text(path, encoding="utf-8")
Path("model_manifest.txt").write_text(
    f"repo_id={REPO_ID}\nfilename={filename}\nlocal_path={path}\n",
    encoding="utf-8",
)

print(f"MODEL_REPO={REPO_ID}")
print(f"MODEL_FILE={filename}")
print(f"MODEL_PATH={path}")
