# Score-Learning Runners

These scripts generate per-seed score-learning CSVs for the public datasets
used in the paper.  They train the scoring models and write local CSV outputs.
The paper's frozen CSVs, aggregation scripts, and plotting scripts live in the
submission evidence bundle rather than this public runner repository.

Main entry points:

- `wsdm_kuairand.py`: KuaiRand-Pure randomized recommendation data.
- `wsdm_insurance.py`: OpenML freMTPL2.
- `australian_mtpl_nested.py`: Australian MTPL nested protocol.
- `belgian_mtpl.py`: Belgian MTPL.
- `lending_club.py`: Lending Club.
- `prosper.py`: Prosper.
- `sba.py`: SBA 7(a).
- `bondora.py`: Bondora.
- `mortgage_nested.py`: Fannie Mae and Freddie Mac mortgage samples.
- `mortgage_lambdarank.py`: mortgage LambdaMART diagnostic.

Run each script with `--help` for dataset paths and output options.
