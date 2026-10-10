# Paper Sync, 2026-10-10

The code-to-paper discrepancy audit found one paper wording error: the
group-aware complaint price is normalized by validation group-denominator value
masses, not validation complaint rates.

Resolved references:

- Paper Overleaf commit:
  `cd69ad8afaa25131ff70981af7a960b895e77ae4`
- Data/evidence Overleaf commit:
  `05212d1558ae8eff930511b9b78a2f8939075876`
- Local compile proof for the synced paper:
  `main.pdf` SHA-256
  `db03bb4564dd18cad1d788e06cbe91129c7a8b17115df326a8b4442baa992d65`

The local Tectonic compile had 13 pages, references beginning on page 8, the
appendix beginning on page 9, and zero fatal, undefined-reference, or rerun
warnings. That compile check is a source/PDF provenance gate; it is not an
anonymous-submission or page-limit gate.
