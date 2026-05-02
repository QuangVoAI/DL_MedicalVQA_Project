Reconstruct the archive:

```bash
cat b2_best_ckpt.tar.gz.part.* > b2_best_ckpt.tar.gz
shasum -a 256 b2_best_ckpt.tar.gz
tar -xzf b2_best_ckpt.tar.gz
mkdir -p checkpoints/B2
mv checkpoint-2013 checkpoints/B2/
```

Expected SHA256 is listed in `SHA256SUMS.txt`.
