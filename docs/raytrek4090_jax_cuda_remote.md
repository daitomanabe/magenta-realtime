# Raytrek4090 JAX/CUDA Remote Render

Use this path when the Raytrek 4090 machine is reached through Windows SSH, but
Magenta RT generation should run inside WSL2 Linux with NVIDIA CUDA and JAX.

## Remote Requirements

- Windows OpenSSH server accepts the `raytrek4090` SSH alias.
- WSL2 distro is installed, preferably Ubuntu.
- `wsl.exe -e uname -s` over SSH returns `Linux`.
- `nvidia-smi` is visible inside WSL2.
- WSL2 has `rsync`, `python3`, and `curl`.
- `uv` is optional; the batch script can bootstrap it into `$HOME/.local/bin`.

## Smoke Test

Dry-run without loading the model:

```bash
scripts/raytrek4090_magenta_dark_ambient_batch.sh \
  --remote-exec windows-wsl \
  --wsl-distro Ubuntu \
  --dry-run \
  --limit 1 \
  --skip-install \
  --skip-model-download
```

One real CUDA/JAX audio render:

```bash
scripts/raytrek4090_magenta_dark_ambient_batch.sh \
  --remote-exec windows-wsl \
  --wsl-distro Ubuntu \
  --limit 1
```

Full 16-take render:

```bash
scripts/raytrek4090_magenta_dark_ambient_batch.sh \
  --remote-exec windows-wsl \
  --wsl-distro Ubuntu
```

## Notes

- `--remote-exec auto` first tries direct Linux SSH, then falls back to
  `windows-wsl`.
- Relative `--remote-dir` values are resolved under WSL `$HOME`, so the default
  becomes `$HOME/workspaces/magenta-realtime`.
- Repo transfer and output sync use `rsync --rsync-path="wsl.exe ... rsync"` in
  `windows-wsl` mode.
- The default JAX CUDA extra is `cuda12`, which is usually safest for current
  RTX 4090 drivers. Use `--jax-cuda-extra cuda13` only when the installed NVIDIA
  driver supports it.
