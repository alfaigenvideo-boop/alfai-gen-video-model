import os, subprocess, pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]

# Sadece gerekli dosyalar:
FILES_TO_DOWNLOAD = {
    "unet": {
        "repo_id": "stabilityai/stable-diffusion-xl-base-1.0",
        "files": ["sd_xl_base_1.0.safetensors"],
        "dest": ROOT / "models/sdxl/base/minimal"
    },
    "vae": {
        "repo_id": "madebyollin/sdxl-vae-fp16-fix",  # veya sd-xl base içindeki VAE
        "files": ["sdxl_vae.safetensors"],           # wildcard için: "*.safetensors"
        "dest": ROOT / "models/sdxl_vae/minimal"
    }
}


def hf_download_file(repo_id: str, filename: str, dest: pathlib.Path):
    dest.mkdir(parents=True, exist_ok=True)

    cmd = [
        "huggingface-cli", "download",
        repo_id,
        filename,
        "--local-dir", str(dest),
        "--local-dir-use-symlinks", "False"
    ]

    print(" ".join(cmd))
    subprocess.check_call(cmd)


def main():
    for group, item in FILES_TO_DOWNLOAD.items():
        repo = item["repo_id"]
        dest = item["dest"]

        print(f"\n==> {group.upper()} | Repo: {repo}")
        for file in item["files"]:
            print(f"   Downloading file: {file}")
            hf_download_file(repo, file, dest)

    print("\nOnly UNet + VAE files downloaded. Done.")


if __name__ == "__main__":
    main()
