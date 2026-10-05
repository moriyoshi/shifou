"""Download pinned public evaluation inputs; no Python packages required."""
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[2] / ".agents-workspace" / "tmp"
MODEL = "Qwen/Qwen3-0.6B"
REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
FILES = {
    "config.json": "660db3b73d788119c04535e48cf9be5f55bc3100841a718637ae695b442f27dd",
    "tokenizer.json": "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4",
    "tokenizer_config.json": "d5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101",
    "model.safetensors": "f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b",
}
CORPUS_URL = (
    "https://raw.githubusercontent.com/pytorch/examples/"
    "d5678bc8ac0cdd79dbd5e44d4130271018bcec4e/word_language_model/data/wikitext-2/test.txt"
)
CORPUS_SHA = "d790b833ef8cf03a90db7bf1271b7520b83c45ce07ba3c1a9699df81e239eca0"


def checksum(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def fetch(url, destination, expected):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if checksum(destination) != expected:
            raise RuntimeError(f"Existing file has a different checksum: {destination}")
    else:
        # Exclusive create protects existing local data, including partial downloads.
        with urllib.request.urlopen(url) as source, destination.open("xb") as target:
            while chunk := source.read(1024 * 1024):
                target.write(chunk)
        if checksum(destination) != expected:
            raise RuntimeError(f"Download checksum mismatch: {destination}")
    print(destination)


def main():
    model_dir = ROOT / "models" / "Qwen3-0.6B"
    for name, digest in FILES.items():
        fetch(f"https://huggingface.co/{MODEL}/resolve/{REVISION}/{name}", model_dir / name, digest)
    revision = {"model": MODEL, "revision": REVISION}
    revision_path = model_dir / "revision.json"
    if revision_path.exists():
        assert json.loads(revision_path.read_text()) == revision
    else:
        revision_path.write_text(json.dumps(revision))
    fetch(CORPUS_URL, ROOT / "corpora" / "wikitext-2-test.txt", CORPUS_SHA)


if __name__ == "__main__":
    main()
