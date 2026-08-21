# W16 Public Commands

## Combined Run

The combined runner trains the candidate generators and runs the deterministic
summaries:

```bash
python w16/run_w16_gpu.py \
  --australian-rda <path/to/ausprivauto0405.rda> \
  --belgian-rda <path/to/beMTPL97.rda> \
  --kuairand-dir <local KuaiRand-Pure/data path materialized from the public source> \
  --output-dir <run output dir> \
  --device cuda:0 \
  --seeds <N> \
  --epochs <N> \
  --lambda-grid-points 41
```

Record the exact command, Git commit, pinned requirements, raw input source URLs
and hashes, output CSV paths, and output CSV checksums.

## KuaiRand Commands

These commands run KuaiRand candidate generation and summarization directly.

```bash
python w16/w16_kuairand_torch_gpu.py \
  --input-dir <KuaiRand-Pure/data> \
  --max-rows 300000 \
  --seeds 50 \
  --epochs 8 \
  --batch-size 8192 \
  --device cuda:0 \
  --output w16_kuairand_300k_seed50_candidates.csv

python w16/summarize_w16_boundary_fast.py \
  --input w16_kuairand_300k_seed50_candidates.csv \
  --output w16_kuairand_300k_seed50_summary.csv \
  --lambda-grid-points 41

python w16/w16_kuairand_torch_gpu.py \
  --input-dir <KuaiRand-Pure/data> \
  --max-rows 0 \
  --seeds 10 \
  --epochs 8 \
  --batch-size 8192 \
  --device cuda:0 \
  --output w16_kuairand_full_seed10_candidates.csv

python w16/summarize_w16_boundary_fast.py \
  --input w16_kuairand_full_seed10_candidates.csv \
  --output w16_kuairand_full_seed10_summary.csv \
  --lambda-grid-points 41
```

Keep `--lambda-grid-points 41` when reproducing the reported grid summaries.

## CAS Commands

These commands run Australian and Belgian CASdatasets candidate generation and
summarization directly.

```bash
python w16/w16_cas_torch_gpu.py \
  --australian-rda <ausprivauto0405.rda> \
  --belgian-rda <beMTPL97.rda> \
  --datasets australian,belgian \
  --belgian-n 100000 \
  --seeds <N> \
  --epochs 8 \
  --batch-size 4096 \
  --device cuda:0 \
  --output w16_cas_torch_gpu_candidates.csv

python w16/summarize_w16_boundary_fast.py \
  --input w16_cas_torch_gpu_candidates.csv \
  --output w16_cas_torch_gpu_summary.csv \
  --lambda-grid-points 41
```
