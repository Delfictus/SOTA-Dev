# CryptoBench Batch Commands

Total: 194 structures in 39 batches of 5

## Quick Start

```bash
# Run everything:
bash /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/run_all_batches.sh

# Or run individual batches:
bash /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/batches/batch_001.sh
bash /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/batches/batch_002.sh
bash /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/batches/batch_003.sh
# ... (36 more batches)
```

## Individual Structure Commands

Each structure runs:
```
nhs_rt_full -t <topo>.json -o <results_dir> --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v
```

## All Batches

### Batch 1 (1arl, 1bk2, 1bzj, 1cwq, 1dq2)
```bash
bash /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/batches/batch_001.sh
```

Or manually:
```bash
mkdir -p /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/1arl && /home/diddy/Desktop/Prism4D-bio/target/release/nhs_rt_full -t /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/topologies/1arl.topology.json -o /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/1arl --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v
mkdir -p /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/1bk2 && /home/diddy/Desktop/Prism4D-bio/target/release/nhs_rt_full -t /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/topologies/1bk2.topology.json -o /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/1bk2 --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v
mkdir -p /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/1bzj && /home/diddy/Desktop/Prism4D-bio/target/release/nhs_rt_full -t /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/topologies/1bzj.topology.json -o /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/1bzj --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v
mkdir -p /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/1cwq && /home/diddy/Desktop/Prism4D-bio/target/release/nhs_rt_full -t /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/topologies/1cwq.topology.json -o /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/1cwq --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v
mkdir -p /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/1dq2 && /home/diddy/Desktop/Prism4D-bio/target/release/nhs_rt_full -t /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/topologies/1dq2.topology.json -o /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/1dq2 --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v
```

### Batch 2 (1e6k, 1evy, 1fd4, 1fe6, 1g1m)
```bash
bash /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/batches/batch_002.sh
```

Or manually:
```bash
mkdir -p /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/1e6k && /home/diddy/Desktop/Prism4D-bio/target/release/nhs_rt_full -t /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/topologies/1e6k.topology.json -o /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/1e6k --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v
mkdir -p /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/1evy && /home/diddy/Desktop/Prism4D-bio/target/release/nhs_rt_full -t /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/topologies/1evy.topology.json -o /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/1evy --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v
mkdir -p /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/1fd4 && /home/diddy/Desktop/Prism4D-bio/target/release/nhs_rt_full -t /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/topologies/1fd4.topology.json -o /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/1fd4 --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v
mkdir -p /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/1fe6 && /home/diddy/Desktop/Prism4D-bio/target/release/nhs_rt_full -t /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/topologies/1fe6.topology.json -o /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/1fe6 --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v
mkdir -p /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/1g1m && /home/diddy/Desktop/Prism4D-bio/target/release/nhs_rt_full -t /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/topologies/1g1m.topology.json -o /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/1g1m --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v
```

### Batch 3 (1h13, 1i7n, 1ksg, 1kx9, 1kxr)
```bash
bash /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/batches/batch_003.sh
```

Or manually:
```bash
mkdir -p /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/1h13 && /home/diddy/Desktop/Prism4D-bio/target/release/nhs_rt_full -t /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/topologies/1h13.topology.json -o /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/1h13 --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v
mkdir -p /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/1i7n && /home/diddy/Desktop/Prism4D-bio/target/release/nhs_rt_full -t /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/topologies/1i7n.topology.json -o /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/1i7n --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v
mkdir -p /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/1ksg && /home/diddy/Desktop/Prism4D-bio/target/release/nhs_rt_full -t /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/topologies/1ksg.topology.json -o /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/1ksg --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v
mkdir -p /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/1kx9 && /home/diddy/Desktop/Prism4D-bio/target/release/nhs_rt_full -t /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/topologies/1kx9.topology.json -o /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/1kx9 --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v
mkdir -p /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/1kxr && /home/diddy/Desktop/Prism4D-bio/target/release/nhs_rt_full -t /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/topologies/1kxr.topology.json -o /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/1kxr --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v
```

### Batch 4 (1lbe, 1nd7, 1p4o, 1p9o, 1pu5)
```bash
bash /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/batches/batch_004.sh
```

Or manually:
```bash
mkdir -p /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/1lbe && /home/diddy/Desktop/Prism4D-bio/target/release/nhs_rt_full -t /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/topologies/1lbe.topology.json -o /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/1lbe --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v
mkdir -p /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/1nd7 && /home/diddy/Desktop/Prism4D-bio/target/release/nhs_rt_full -t /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/topologies/1nd7.topology.json -o /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/1nd7 --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v
mkdir -p /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/1p4o && /home/diddy/Desktop/Prism4D-bio/target/release/nhs_rt_full -t /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/topologies/1p4o.topology.json -o /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/1p4o --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v
mkdir -p /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/1p9o && /home/diddy/Desktop/Prism4D-bio/target/release/nhs_rt_full -t /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/topologies/1p9o.topology.json -o /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/1p9o --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v
mkdir -p /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/1pu5 && /home/diddy/Desktop/Prism4D-bio/target/release/nhs_rt_full -t /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/topologies/1pu5.topology.json -o /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/1pu5 --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v
```

### Batch 5 (1q4k, 1r3m, 1rjb, 1rtc, 1se8)
```bash
bash /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/batches/batch_005.sh
```

Or manually:
```bash
mkdir -p /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/1q4k && /home/diddy/Desktop/Prism4D-bio/target/release/nhs_rt_full -t /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/topologies/1q4k.topology.json -o /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/1q4k --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v
mkdir -p /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/1r3m && /home/diddy/Desktop/Prism4D-bio/target/release/nhs_rt_full -t /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/topologies/1r3m.topology.json -o /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/1r3m --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v
mkdir -p /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/1rjb && /home/diddy/Desktop/Prism4D-bio/target/release/nhs_rt_full -t /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/topologies/1rjb.topology.json -o /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/1rjb --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v
mkdir -p /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/1rtc && /home/diddy/Desktop/Prism4D-bio/target/release/nhs_rt_full -t /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/topologies/1rtc.topology.json -o /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/1rtc --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v
mkdir -p /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/1se8 && /home/diddy/Desktop/Prism4D-bio/target/release/nhs_rt_full -t /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/topologies/1se8.topology.json -o /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/1se8 --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v
```

### Batch 6 (1tmi, 1uka, 1ute, 1vsn, 1x2g)
```bash
bash /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/batches/batch_006.sh
```

Or manually:
```bash
mkdir -p /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/1tmi && /home/diddy/Desktop/Prism4D-bio/target/release/nhs_rt_full -t /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/topologies/1tmi.topology.json -o /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/1tmi --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v
mkdir -p /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/1uka && /home/diddy/Desktop/Prism4D-bio/target/release/nhs_rt_full -t /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/topologies/1uka.topology.json -o /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/1uka --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v
mkdir -p /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/1ute && /home/diddy/Desktop/Prism4D-bio/target/release/nhs_rt_full -t /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/topologies/1ute.topology.json -o /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/1ute --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v
mkdir -p /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/1vsn && /home/diddy/Desktop/Prism4D-bio/target/release/nhs_rt_full -t /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/topologies/1vsn.topology.json -o /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/1vsn --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v
mkdir -p /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/1x2g && /home/diddy/Desktop/Prism4D-bio/target/release/nhs_rt_full -t /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/topologies/1x2g.topology.json -o /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/1x2g --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v
```

### Batch 7 (1xgd, 1xjf, 1xqv, 1xtc, 1xxo)
```bash
bash /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/batches/batch_007.sh
```

Or manually:
```bash
mkdir -p /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/1xgd && /home/diddy/Desktop/Prism4D-bio/target/release/nhs_rt_full -t /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/topologies/1xgd.topology.json -o /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/1xgd --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v
mkdir -p /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/1xjf && /home/diddy/Desktop/Prism4D-bio/target/release/nhs_rt_full -t /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/topologies/1xjf.topology.json -o /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/1xjf --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v
mkdir -p /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/1xqv && /home/diddy/Desktop/Prism4D-bio/target/release/nhs_rt_full -t /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/topologies/1xqv.topology.json -o /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/1xqv --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v
mkdir -p /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/1xtc && /home/diddy/Desktop/Prism4D-bio/target/release/nhs_rt_full -t /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/topologies/1xtc.topology.json -o /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/1xtc --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v
mkdir -p /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/1xxo && /home/diddy/Desktop/Prism4D-bio/target/release/nhs_rt_full -t /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/topologies/1xxo.topology.json -o /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/1xxo --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v
```

### Batch 8 (1zm0, 2aka, 2czd, 2d05, 2dfp)
```bash
bash /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/batches/batch_008.sh
```

Or manually:
```bash
mkdir -p /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/1zm0 && /home/diddy/Desktop/Prism4D-bio/target/release/nhs_rt_full -t /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/topologies/1zm0.topology.json -o /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/1zm0 --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v
mkdir -p /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/2aka && /home/diddy/Desktop/Prism4D-bio/target/release/nhs_rt_full -t /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/topologies/2aka.topology.json -o /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/2aka --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v
mkdir -p /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/2czd && /home/diddy/Desktop/Prism4D-bio/target/release/nhs_rt_full -t /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/topologies/2czd.topology.json -o /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/2czd --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v
mkdir -p /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/2d05 && /home/diddy/Desktop/Prism4D-bio/target/release/nhs_rt_full -t /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/topologies/2d05.topology.json -o /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/2d05 --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v
mkdir -p /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/2dfp && /home/diddy/Desktop/Prism4D-bio/target/release/nhs_rt_full -t /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/topologies/2dfp.topology.json -o /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/2dfp --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v
```

### Batch 9 (2fem, 2fhz, 2h7s, 2huw, 2i3a)
```bash
bash /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/batches/batch_009.sh
```

Or manually:
```bash
mkdir -p /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/2fem && /home/diddy/Desktop/Prism4D-bio/target/release/nhs_rt_full -t /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/topologies/2fem.topology.json -o /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/2fem --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v
mkdir -p /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/2fhz && /home/diddy/Desktop/Prism4D-bio/target/release/nhs_rt_full -t /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/topologies/2fhz.topology.json -o /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/2fhz --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v
mkdir -p /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/2h7s && /home/diddy/Desktop/Prism4D-bio/target/release/nhs_rt_full -t /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/topologies/2h7s.topology.json -o /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/2h7s --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v
mkdir -p /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/2huw && /home/diddy/Desktop/Prism4D-bio/target/release/nhs_rt_full -t /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/topologies/2huw.topology.json -o /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/2huw --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v
mkdir -p /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/2i3a && /home/diddy/Desktop/Prism4D-bio/target/release/nhs_rt_full -t /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/topologies/2i3a.topology.json -o /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/2i3a --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v
```

### Batch 10 (2i3r, 2idj, 2iyt, 2phz, 2pkf)
```bash
bash /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/batches/batch_010.sh
```

Or manually:
```bash
mkdir -p /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/2i3r && /home/diddy/Desktop/Prism4D-bio/target/release/nhs_rt_full -t /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/topologies/2i3r.topology.json -o /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/2i3r --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v
mkdir -p /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/2idj && /home/diddy/Desktop/Prism4D-bio/target/release/nhs_rt_full -t /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/topologies/2idj.topology.json -o /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/2idj --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v
mkdir -p /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/2iyt && /home/diddy/Desktop/Prism4D-bio/target/release/nhs_rt_full -t /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/topologies/2iyt.topology.json -o /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/2iyt --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v
mkdir -p /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/2phz && /home/diddy/Desktop/Prism4D-bio/target/release/nhs_rt_full -t /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/topologies/2phz.topology.json -o /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/2phz --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v
mkdir -p /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/2pkf && /home/diddy/Desktop/Prism4D-bio/target/release/nhs_rt_full -t /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/topologies/2pkf.topology.json -o /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/2pkf --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v
```

### Batch 11 (2pwz, 2qbv, 2rfj, 2v6m, 2vl2)
```bash
bash /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/batches/batch_011.sh
```

Or manually:
```bash
mkdir -p /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/2pwz && /home/diddy/Desktop/Prism4D-bio/target/release/nhs_rt_full -t /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/topologies/2pwz.topology.json -o /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/2pwz --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v
mkdir -p /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/2qbv && /home/diddy/Desktop/Prism4D-bio/target/release/nhs_rt_full -t /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/topologies/2qbv.topology.json -o /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/2qbv --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v
mkdir -p /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/2rfj && /home/diddy/Desktop/Prism4D-bio/target/release/nhs_rt_full -t /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/topologies/2rfj.topology.json -o /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/2rfj --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v
mkdir -p /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/2v6m && /home/diddy/Desktop/Prism4D-bio/target/release/nhs_rt_full -t /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/topologies/2v6m.topology.json -o /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/2v6m --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v
mkdir -p /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/2vl2 && /home/diddy/Desktop/Prism4D-bio/target/release/nhs_rt_full -t /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/topologies/2vl2.topology.json -o /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/2vl2 --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v
```

### Batch 12 (2vqz, 2vyr, 2w8n, 2x47, 2xdo)
```bash
bash /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/batches/batch_012.sh
```

Or manually:
```bash
mkdir -p /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/2vqz && /home/diddy/Desktop/Prism4D-bio/target/release/nhs_rt_full -t /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/topologies/2vqz.topology.json -o /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/2vqz --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v
mkdir -p /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/2vyr && /home/diddy/Desktop/Prism4D-bio/target/release/nhs_rt_full -t /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/topologies/2vyr.topology.json -o /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/2vyr --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v
mkdir -p /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/2w8n && /home/diddy/Desktop/Prism4D-bio/target/release/nhs_rt_full -t /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/topologies/2w8n.topology.json -o /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/2w8n --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v
mkdir -p /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/2x47 && /home/diddy/Desktop/Prism4D-bio/target/release/nhs_rt_full -t /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/topologies/2x47.topology.json -o /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/2x47 --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v
mkdir -p /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/2xdo && /home/diddy/Desktop/Prism4D-bio/target/release/nhs_rt_full -t /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/topologies/2xdo.topology.json -o /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/2xdo --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v
```

### Batch 13 (2xsa, 2zcg, 2zj7, 3a0x, 3b1o)
```bash
bash /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/batches/batch_013.sh
```

Or manually:
```bash
mkdir -p /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/2xsa && /home/diddy/Desktop/Prism4D-bio/target/release/nhs_rt_full -t /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/topologies/2xsa.topology.json -o /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/2xsa --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v
mkdir -p /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/2zcg && /home/diddy/Desktop/Prism4D-bio/target/release/nhs_rt_full -t /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/topologies/2zcg.topology.json -o /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/2zcg --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v
mkdir -p /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/2zj7 && /home/diddy/Desktop/Prism4D-bio/target/release/nhs_rt_full -t /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/topologies/2zj7.topology.json -o /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/2zj7 --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v
mkdir -p /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/3a0x && /home/diddy/Desktop/Prism4D-bio/target/release/nhs_rt_full -t /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/topologies/3a0x.topology.json -o /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/3a0x --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v
mkdir -p /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/3b1o && /home/diddy/Desktop/Prism4D-bio/target/release/nhs_rt_full -t /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/topologies/3b1o.topology.json -o /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/3b1o --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v
```

### Batch 14 (3bjp, 3f4k, 3flg, 3fzo, 3gdg)
```bash
bash /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/batches/batch_014.sh
```

Or manually:
```bash
mkdir -p /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/3bjp && /home/diddy/Desktop/Prism4D-bio/target/release/nhs_rt_full -t /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/topologies/3bjp.topology.json -o /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/3bjp --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v
mkdir -p /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/3f4k && /home/diddy/Desktop/Prism4D-bio/target/release/nhs_rt_full -t /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/topologies/3f4k.topology.json -o /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/3f4k --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v
mkdir -p /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/3flg && /home/diddy/Desktop/Prism4D-bio/target/release/nhs_rt_full -t /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/topologies/3flg.topology.json -o /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/3flg --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v
mkdir -p /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/3fzo && /home/diddy/Desktop/Prism4D-bio/target/release/nhs_rt_full -t /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/topologies/3fzo.topology.json -o /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/3fzo --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v
mkdir -p /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/3gdg && /home/diddy/Desktop/Prism4D-bio/target/release/nhs_rt_full -t /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/topologies/3gdg.topology.json -o /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/3gdg --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v
```

### Batch 15 (3h8a, 3hrm, 3i8s, 3idh, 3jzg)
```bash
bash /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/batches/batch_015.sh
```

Or manually:
```bash
mkdir -p /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/3h8a && /home/diddy/Desktop/Prism4D-bio/target/release/nhs_rt_full -t /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/topologies/3h8a.topology.json -o /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/3h8a --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v
mkdir -p /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/3hrm && /home/diddy/Desktop/Prism4D-bio/target/release/nhs_rt_full -t /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/topologies/3hrm.topology.json -o /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/3hrm --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v
mkdir -p /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/3i8s && /home/diddy/Desktop/Prism4D-bio/target/release/nhs_rt_full -t /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/topologies/3i8s.topology.json -o /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/3i8s --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v
mkdir -p /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/3idh && /home/diddy/Desktop/Prism4D-bio/target/release/nhs_rt_full -t /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/topologies/3idh.topology.json -o /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/3idh --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v
mkdir -p /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/3jzg && /home/diddy/Desktop/Prism4D-bio/target/release/nhs_rt_full -t /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/topologies/3jzg.topology.json -o /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/3jzg --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v
```

### Batch 16 (3k01, 3kjr, 3la7, 3lnz, 3ly8)
```bash
bash /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/batches/batch_016.sh
```

Or manually:
```bash
mkdir -p /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/3k01 && /home/diddy/Desktop/Prism4D-bio/target/release/nhs_rt_full -t /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/topologies/3k01.topology.json -o /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/3k01 --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v
mkdir -p /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/3kjr && /home/diddy/Desktop/Prism4D-bio/target/release/nhs_rt_full -t /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/topologies/3kjr.topology.json -o /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/3kjr --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v
mkdir -p /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/3la7 && /home/diddy/Desktop/Prism4D-bio/target/release/nhs_rt_full -t /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/topologies/3la7.topology.json -o /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/3la7 --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v
mkdir -p /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/3lnz && /home/diddy/Desktop/Prism4D-bio/target/release/nhs_rt_full -t /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/topologies/3lnz.topology.json -o /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/3lnz --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v
mkdir -p /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/3ly8 && /home/diddy/Desktop/Prism4D-bio/target/release/nhs_rt_full -t /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/topologies/3ly8.topology.json -o /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/3ly8 --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v
```

### Batch 17 (3mwg, 3n4u, 3nx1, 3pbf, 3pfp)
```bash
bash /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/batches/batch_017.sh
```

Or manually:
```bash
mkdir -p /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/3mwg && /home/diddy/Desktop/Prism4D-bio/target/release/nhs_rt_full -t /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/topologies/3mwg.topology.json -o /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/3mwg --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v
mkdir -p /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/3n4u && /home/diddy/Desktop/Prism4D-bio/target/release/nhs_rt_full -t /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/topologies/3n4u.topology.json -o /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/3n4u --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v
mkdir -p /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/3nx1 && /home/diddy/Desktop/Prism4D-bio/target/release/nhs_rt_full -t /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/topologies/3nx1.topology.json -o /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/3nx1 --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v
mkdir -p /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/3pbf && /home/diddy/Desktop/Prism4D-bio/target/release/nhs_rt_full -t /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/topologies/3pbf.topology.json -o /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/3pbf --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v
mkdir -p /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/3pfp && /home/diddy/Desktop/Prism4D-bio/target/release/nhs_rt_full -t /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/topologies/3pfp.topology.json -o /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/3pfp --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v
```

### Batch 18 (3rwv, 3st6, 3t8b, 3tpo, 3ugk)
```bash
bash /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/batches/batch_018.sh
```

Or manually:
```bash
mkdir -p /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/3rwv && /home/diddy/Desktop/Prism4D-bio/target/release/nhs_rt_full -t /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/topologies/3rwv.topology.json -o /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/3rwv --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v
mkdir -p /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/3st6 && /home/diddy/Desktop/Prism4D-bio/target/release/nhs_rt_full -t /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/topologies/3st6.topology.json -o /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/3st6 --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v
mkdir -p /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/3t8b && /home/diddy/Desktop/Prism4D-bio/target/release/nhs_rt_full -t /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/topologies/3t8b.topology.json -o /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/3t8b --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v
mkdir -p /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/3tpo && /home/diddy/Desktop/Prism4D-bio/target/release/nhs_rt_full -t /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/topologies/3tpo.topology.json -o /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/3tpo --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v
mkdir -p /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/3ugk && /home/diddy/Desktop/Prism4D-bio/target/release/nhs_rt_full -t /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/topologies/3ugk.topology.json -o /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/3ugk --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v
```

### Batch 19 (3uyi, 3v55, 3ve9, 3vgm, 3w90)
```bash
bash /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/batches/batch_019.sh
```

Or manually:
```bash
mkdir -p /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/3uyi && /home/diddy/Desktop/Prism4D-bio/target/release/nhs_rt_full -t /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/topologies/3uyi.topology.json -o /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/3uyi --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v
mkdir -p /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/3v55 && /home/diddy/Desktop/Prism4D-bio/target/release/nhs_rt_full -t /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/topologies/3v55.topology.json -o /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/3v55 --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v
mkdir -p /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/3ve9 && /home/diddy/Desktop/Prism4D-bio/target/release/nhs_rt_full -t /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/topologies/3ve9.topology.json -o /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/3ve9 --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v
mkdir -p /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/3vgm && /home/diddy/Desktop/Prism4D-bio/target/release/nhs_rt_full -t /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/topologies/3vgm.topology.json -o /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/3vgm --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v
mkdir -p /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/3w90 && /home/diddy/Desktop/Prism4D-bio/target/release/nhs_rt_full -t /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/topologies/3w90.topology.json -o /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/3w90 --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v
```

### Batch 20 (3wb9, 4aem, 4amv, 4bg8, 4cmw)
```bash
bash /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/batches/batch_020.sh
```

Or manually:
```bash
mkdir -p /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/3wb9 && /home/diddy/Desktop/Prism4D-bio/target/release/nhs_rt_full -t /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/topologies/3wb9.topology.json -o /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/3wb9 --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v
mkdir -p /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/4aem && /home/diddy/Desktop/Prism4D-bio/target/release/nhs_rt_full -t /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/topologies/4aem.topology.json -o /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/4aem --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v
mkdir -p /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/4amv && /home/diddy/Desktop/Prism4D-bio/target/release/nhs_rt_full -t /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/topologies/4amv.topology.json -o /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/4amv --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v
mkdir -p /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/4bg8 && /home/diddy/Desktop/Prism4D-bio/target/release/nhs_rt_full -t /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/topologies/4bg8.topology.json -o /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/4bg8 --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v
mkdir -p /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/4cmw && /home/diddy/Desktop/Prism4D-bio/target/release/nhs_rt_full -t /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/topologies/4cmw.topology.json -o /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/4cmw --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v
```

### Batch 21 (4dnc, 4e1y, 4fkm, 4gpi, 4hye)
```bash
bash /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/batches/batch_021.sh
```

Or manually:
```bash
mkdir -p /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/4dnc && /home/diddy/Desktop/Prism4D-bio/target/release/nhs_rt_full -t /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/topologies/4dnc.topology.json -o /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/4dnc --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v
mkdir -p /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/4e1y && /home/diddy/Desktop/Prism4D-bio/target/release/nhs_rt_full -t /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/topologies/4e1y.topology.json -o /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/4e1y --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v
mkdir -p /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/4fkm && /home/diddy/Desktop/Prism4D-bio/target/release/nhs_rt_full -t /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/topologies/4fkm.topology.json -o /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/4fkm --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v
mkdir -p /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/4gpi && /home/diddy/Desktop/Prism4D-bio/target/release/nhs_rt_full -t /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/topologies/4gpi.topology.json -o /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/4gpi --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v
mkdir -p /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/4hye && /home/diddy/Desktop/Prism4D-bio/target/release/nhs_rt_full -t /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/topologies/4hye.topology.json -o /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/4hye --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v
```

### Batch 22 (4ikv, 4ilg, 4j4e, 4jfr, 4kmy)
```bash
bash /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/batches/batch_022.sh
```

Or manually:
```bash
mkdir -p /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/4ikv && /home/diddy/Desktop/Prism4D-bio/target/release/nhs_rt_full -t /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/topologies/4ikv.topology.json -o /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/4ikv --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v
mkdir -p /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/4ilg && /home/diddy/Desktop/Prism4D-bio/target/release/nhs_rt_full -t /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/topologies/4ilg.topology.json -o /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/4ilg --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v
mkdir -p /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/4j4e && /home/diddy/Desktop/Prism4D-bio/target/release/nhs_rt_full -t /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/topologies/4j4e.topology.json -o /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/4j4e --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v
mkdir -p /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/4jfr && /home/diddy/Desktop/Prism4D-bio/target/release/nhs_rt_full -t /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/topologies/4jfr.topology.json -o /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/4jfr --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v
mkdir -p /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/4kmy && /home/diddy/Desktop/Prism4D-bio/target/release/nhs_rt_full -t /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/topologies/4kmy.topology.json -o /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/4kmy --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v
```

### Batch 23 (4mwi, 4nzv, 4oqo, 4p2f, 4p32)
```bash
bash /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/batches/batch_023.sh
```

Or manually:
```bash
mkdir -p /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/4mwi && /home/diddy/Desktop/Prism4D-bio/target/release/nhs_rt_full -t /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/topologies/4mwi.topology.json -o /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/4mwi --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v
mkdir -p /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/4nzv && /home/diddy/Desktop/Prism4D-bio/target/release/nhs_rt_full -t /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/topologies/4nzv.topology.json -o /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/4nzv --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v
mkdir -p /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/4oqo && /home/diddy/Desktop/Prism4D-bio/target/release/nhs_rt_full -t /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/topologies/4oqo.topology.json -o /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/4oqo --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v
mkdir -p /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/4p2f && /home/diddy/Desktop/Prism4D-bio/target/release/nhs_rt_full -t /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/topologies/4p2f.topology.json -o /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/4p2f --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v
mkdir -p /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/4p32 && /home/diddy/Desktop/Prism4D-bio/target/release/nhs_rt_full -t /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/topologies/4p32.topology.json -o /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/4p32 --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v
```

### Batch 24 (4qvk, 4r0x, 4rvt, 4ttp, 4uum)
```bash
bash /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/batches/batch_024.sh
```

Or manually:
```bash
mkdir -p /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/4qvk && /home/diddy/Desktop/Prism4D-bio/target/release/nhs_rt_full -t /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/topologies/4qvk.topology.json -o /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/4qvk --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v
mkdir -p /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/4r0x && /home/diddy/Desktop/Prism4D-bio/target/release/nhs_rt_full -t /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/topologies/4r0x.topology.json -o /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/4r0x --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v
mkdir -p /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/4rvt && /home/diddy/Desktop/Prism4D-bio/target/release/nhs_rt_full -t /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/topologies/4rvt.topology.json -o /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/4rvt --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v
mkdir -p /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/4ttp && /home/diddy/Desktop/Prism4D-bio/target/release/nhs_rt_full -t /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/topologies/4ttp.topology.json -o /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/4ttp --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v
mkdir -p /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/4uum && /home/diddy/Desktop/Prism4D-bio/target/release/nhs_rt_full -t /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/topologies/4uum.topology.json -o /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/4uum --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v
```

### Batch 25 (4x19, 4zm7, 4zoe, 5acv, 5aon)
```bash
bash /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/batches/batch_025.sh
```

Or manually:
```bash
mkdir -p /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/4x19 && /home/diddy/Desktop/Prism4D-bio/target/release/nhs_rt_full -t /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/topologies/4x19.topology.json -o /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/4x19 --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v
mkdir -p /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/4zm7 && /home/diddy/Desktop/Prism4D-bio/target/release/nhs_rt_full -t /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/topologies/4zm7.topology.json -o /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/4zm7 --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v
mkdir -p /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/4zoe && /home/diddy/Desktop/Prism4D-bio/target/release/nhs_rt_full -t /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/topologies/4zoe.topology.json -o /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/4zoe --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v
mkdir -p /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/5acv && /home/diddy/Desktop/Prism4D-bio/target/release/nhs_rt_full -t /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/topologies/5acv.topology.json -o /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/5acv --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v
mkdir -p /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/5aon && /home/diddy/Desktop/Prism4D-bio/target/release/nhs_rt_full -t /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/topologies/5aon.topology.json -o /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/5aon --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v
```

### Batch 26 (5b0e, 5caz, 5dy9, 5e0v, 5ey7)
```bash
bash /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/batches/batch_026.sh
```

Or manually:
```bash
mkdir -p /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/5b0e && /home/diddy/Desktop/Prism4D-bio/target/release/nhs_rt_full -t /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/topologies/5b0e.topology.json -o /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/5b0e --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v
mkdir -p /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/5caz && /home/diddy/Desktop/Prism4D-bio/target/release/nhs_rt_full -t /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/topologies/5caz.topology.json -o /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/5caz --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v
mkdir -p /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/5dy9 && /home/diddy/Desktop/Prism4D-bio/target/release/nhs_rt_full -t /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/topologies/5dy9.topology.json -o /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/5dy9 --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v
mkdir -p /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/5e0v && /home/diddy/Desktop/Prism4D-bio/target/release/nhs_rt_full -t /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/topologies/5e0v.topology.json -o /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/5e0v --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v
mkdir -p /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/5ey7 && /home/diddy/Desktop/Prism4D-bio/target/release/nhs_rt_full -t /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/topologies/5ey7.topology.json -o /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/5ey7 --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v
```

### Batch 27 (5gmc, 5hij, 5igh, 5kcg, 5loc)
```bash
bash /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/batches/batch_027.sh
```

Or manually:
```bash
mkdir -p /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/5gmc && /home/diddy/Desktop/Prism4D-bio/target/release/nhs_rt_full -t /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/topologies/5gmc.topology.json -o /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/5gmc --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v
mkdir -p /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/5hij && /home/diddy/Desktop/Prism4D-bio/target/release/nhs_rt_full -t /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/topologies/5hij.topology.json -o /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/5hij --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v
mkdir -p /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/5igh && /home/diddy/Desktop/Prism4D-bio/target/release/nhs_rt_full -t /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/topologies/5igh.topology.json -o /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/5igh --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v
mkdir -p /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/5kcg && /home/diddy/Desktop/Prism4D-bio/target/release/nhs_rt_full -t /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/topologies/5kcg.topology.json -o /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/5kcg --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v
mkdir -p /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/5loc && /home/diddy/Desktop/Prism4D-bio/target/release/nhs_rt_full -t /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/topologies/5loc.topology.json -o /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/5loc --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v
```

### Batch 28 (5m7r, 5n49, 5o8b, 5sc2, 5tc0)
```bash
bash /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/batches/batch_028.sh
```

Or manually:
```bash
mkdir -p /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/5m7r && /home/diddy/Desktop/Prism4D-bio/target/release/nhs_rt_full -t /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/topologies/5m7r.topology.json -o /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/5m7r --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v
mkdir -p /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/5n49 && /home/diddy/Desktop/Prism4D-bio/target/release/nhs_rt_full -t /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/topologies/5n49.topology.json -o /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/5n49 --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v
mkdir -p /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/5o8b && /home/diddy/Desktop/Prism4D-bio/target/release/nhs_rt_full -t /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/topologies/5o8b.topology.json -o /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/5o8b --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v
mkdir -p /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/5sc2 && /home/diddy/Desktop/Prism4D-bio/target/release/nhs_rt_full -t /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/topologies/5sc2.topology.json -o /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/5sc2 --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v
mkdir -p /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/5tc0 && /home/diddy/Desktop/Prism4D-bio/target/release/nhs_rt_full -t /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/topologies/5tc0.topology.json -o /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/5tc0 --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v
```

### Batch 29 (5tvi, 5ujp, 5uxa, 5wbm, 5wm9)
```bash
bash /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/batches/batch_029.sh
```

Or manually:
```bash
mkdir -p /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/5tvi && /home/diddy/Desktop/Prism4D-bio/target/release/nhs_rt_full -t /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/topologies/5tvi.topology.json -o /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/5tvi --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v
mkdir -p /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/5ujp && /home/diddy/Desktop/Prism4D-bio/target/release/nhs_rt_full -t /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/topologies/5ujp.topology.json -o /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/5ujp --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v
mkdir -p /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/5uxa && /home/diddy/Desktop/Prism4D-bio/target/release/nhs_rt_full -t /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/topologies/5uxa.topology.json -o /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/5uxa --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v
mkdir -p /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/5wbm && /home/diddy/Desktop/Prism4D-bio/target/release/nhs_rt_full -t /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/topologies/5wbm.topology.json -o /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/5wbm --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v
mkdir -p /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/5wm9 && /home/diddy/Desktop/Prism4D-bio/target/release/nhs_rt_full -t /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/topologies/5wm9.topology.json -o /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/5wm9 --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v
```

### Batch 30 (5yhb, 5yqp, 5ysb, 5zj4, 6a98)
```bash
bash /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/batches/batch_030.sh
```

Or manually:
```bash
mkdir -p /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/5yhb && /home/diddy/Desktop/Prism4D-bio/target/release/nhs_rt_full -t /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/topologies/5yhb.topology.json -o /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/5yhb --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v
mkdir -p /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/5yqp && /home/diddy/Desktop/Prism4D-bio/target/release/nhs_rt_full -t /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/topologies/5yqp.topology.json -o /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/5yqp --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v
mkdir -p /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/5ysb && /home/diddy/Desktop/Prism4D-bio/target/release/nhs_rt_full -t /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/topologies/5ysb.topology.json -o /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/5ysb --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v
mkdir -p /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/5zj4 && /home/diddy/Desktop/Prism4D-bio/target/release/nhs_rt_full -t /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/topologies/5zj4.topology.json -o /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/5zj4 --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v
mkdir -p /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/6a98 && /home/diddy/Desktop/Prism4D-bio/target/release/nhs_rt_full -t /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/topologies/6a98.topology.json -o /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/6a98 --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v
```

### Batch 31 (6bty, 6cqe, 6du4, 6eqj, 6f52)
```bash
bash /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/batches/batch_031.sh
```

Or manually:
```bash
mkdir -p /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/6bty && /home/diddy/Desktop/Prism4D-bio/target/release/nhs_rt_full -t /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/topologies/6bty.topology.json -o /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/6bty --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v
mkdir -p /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/6cqe && /home/diddy/Desktop/Prism4D-bio/target/release/nhs_rt_full -t /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/topologies/6cqe.topology.json -o /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/6cqe --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v
mkdir -p /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/6du4 && /home/diddy/Desktop/Prism4D-bio/target/release/nhs_rt_full -t /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/topologies/6du4.topology.json -o /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/6du4 --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v
mkdir -p /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/6eqj && /home/diddy/Desktop/Prism4D-bio/target/release/nhs_rt_full -t /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/topologies/6eqj.topology.json -o /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/6eqj --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v
mkdir -p /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/6f52 && /home/diddy/Desktop/Prism4D-bio/target/release/nhs_rt_full -t /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/topologies/6f52.topology.json -o /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/6f52 --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v
```

### Batch 32 (6fc2, 6fgj, 6g6y, 6hei, 6isu)
```bash
bash /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/batches/batch_032.sh
```

Or manually:
```bash
mkdir -p /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/6fc2 && /home/diddy/Desktop/Prism4D-bio/target/release/nhs_rt_full -t /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/topologies/6fc2.topology.json -o /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/6fc2 --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v
mkdir -p /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/6fgj && /home/diddy/Desktop/Prism4D-bio/target/release/nhs_rt_full -t /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/topologies/6fgj.topology.json -o /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/6fgj --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v
mkdir -p /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/6g6y && /home/diddy/Desktop/Prism4D-bio/target/release/nhs_rt_full -t /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/topologies/6g6y.topology.json -o /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/6g6y --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v
mkdir -p /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/6hei && /home/diddy/Desktop/Prism4D-bio/target/release/nhs_rt_full -t /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/topologies/6hei.topology.json -o /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/6hei --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v
mkdir -p /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/6isu && /home/diddy/Desktop/Prism4D-bio/target/release/nhs_rt_full -t /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/topologies/6isu.topology.json -o /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/6isu --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v
```

### Batch 33 (6jq9, 6ksc, 6n5j, 6nei, 6syh)
```bash
bash /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/batches/batch_033.sh
```

Or manually:
```bash
mkdir -p /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/6jq9 && /home/diddy/Desktop/Prism4D-bio/target/release/nhs_rt_full -t /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/topologies/6jq9.topology.json -o /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/6jq9 --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v
mkdir -p /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/6ksc && /home/diddy/Desktop/Prism4D-bio/target/release/nhs_rt_full -t /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/topologies/6ksc.topology.json -o /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/6ksc --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v
mkdir -p /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/6n5j && /home/diddy/Desktop/Prism4D-bio/target/release/nhs_rt_full -t /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/topologies/6n5j.topology.json -o /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/6n5j --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v
mkdir -p /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/6nei && /home/diddy/Desktop/Prism4D-bio/target/release/nhs_rt_full -t /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/topologies/6nei.topology.json -o /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/6nei --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v
mkdir -p /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/6syh && /home/diddy/Desktop/Prism4D-bio/target/release/nhs_rt_full -t /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/topologies/6syh.topology.json -o /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/6syh --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v
```

### Batch 34 (6tx0, 6vle, 6w10, 7c48, 7c63)
```bash
bash /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/batches/batch_034.sh
```

Or manually:
```bash
mkdir -p /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/6tx0 && /home/diddy/Desktop/Prism4D-bio/target/release/nhs_rt_full -t /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/topologies/6tx0.topology.json -o /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/6tx0 --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v
mkdir -p /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/6vle && /home/diddy/Desktop/Prism4D-bio/target/release/nhs_rt_full -t /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/topologies/6vle.topology.json -o /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/6vle --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v
mkdir -p /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/6w10 && /home/diddy/Desktop/Prism4D-bio/target/release/nhs_rt_full -t /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/topologies/6w10.topology.json -o /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/6w10 --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v
mkdir -p /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/7c48 && /home/diddy/Desktop/Prism4D-bio/target/release/nhs_rt_full -t /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/topologies/7c48.topology.json -o /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/7c48 --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v
mkdir -p /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/7c63 && /home/diddy/Desktop/Prism4D-bio/target/release/nhs_rt_full -t /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/topologies/7c63.topology.json -o /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/7c63 --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v
```

### Batch 35 (7de1, 7e5q, 7f2m, 7nc8, 7nlx)
```bash
bash /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/batches/batch_035.sh
```

Or manually:
```bash
mkdir -p /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/7de1 && /home/diddy/Desktop/Prism4D-bio/target/release/nhs_rt_full -t /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/topologies/7de1.topology.json -o /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/7de1 --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v
mkdir -p /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/7e5q && /home/diddy/Desktop/Prism4D-bio/target/release/nhs_rt_full -t /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/topologies/7e5q.topology.json -o /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/7e5q --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v
mkdir -p /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/7f2m && /home/diddy/Desktop/Prism4D-bio/target/release/nhs_rt_full -t /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/topologies/7f2m.topology.json -o /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/7f2m --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v
mkdir -p /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/7nc8 && /home/diddy/Desktop/Prism4D-bio/target/release/nhs_rt_full -t /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/topologies/7nc8.topology.json -o /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/7nc8 --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v
mkdir -p /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/7nlx && /home/diddy/Desktop/Prism4D-bio/target/release/nhs_rt_full -t /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/topologies/7nlx.topology.json -o /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/7nlx --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v
```

### Batch 36 (7o1i, 7qoq, 7w19, 7x0f, 7x0g)
```bash
bash /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/batches/batch_036.sh
```

Or manually:
```bash
mkdir -p /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/7o1i && /home/diddy/Desktop/Prism4D-bio/target/release/nhs_rt_full -t /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/topologies/7o1i.topology.json -o /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/7o1i --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v
mkdir -p /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/7qoq && /home/diddy/Desktop/Prism4D-bio/target/release/nhs_rt_full -t /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/topologies/7qoq.topology.json -o /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/7qoq --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v
mkdir -p /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/7w19 && /home/diddy/Desktop/Prism4D-bio/target/release/nhs_rt_full -t /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/topologies/7w19.topology.json -o /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/7w19 --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v
mkdir -p /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/7x0f && /home/diddy/Desktop/Prism4D-bio/target/release/nhs_rt_full -t /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/topologies/7x0f.topology.json -o /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/7x0f --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v
mkdir -p /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/7x0g && /home/diddy/Desktop/Prism4D-bio/target/release/nhs_rt_full -t /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/topologies/7x0g.topology.json -o /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/7x0g --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v
```

### Batch 37 (7x0i, 7xgf, 7yjc, 8aeq, 8aqi)
```bash
bash /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/batches/batch_037.sh
```

Or manually:
```bash
mkdir -p /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/7x0i && /home/diddy/Desktop/Prism4D-bio/target/release/nhs_rt_full -t /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/topologies/7x0i.topology.json -o /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/7x0i --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v
mkdir -p /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/7xgf && /home/diddy/Desktop/Prism4D-bio/target/release/nhs_rt_full -t /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/topologies/7xgf.topology.json -o /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/7xgf --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v
mkdir -p /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/7yjc && /home/diddy/Desktop/Prism4D-bio/target/release/nhs_rt_full -t /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/topologies/7yjc.topology.json -o /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/7yjc --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v
mkdir -p /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/8aeq && /home/diddy/Desktop/Prism4D-bio/target/release/nhs_rt_full -t /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/topologies/8aeq.topology.json -o /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/8aeq --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v
mkdir -p /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/8aqi && /home/diddy/Desktop/Prism4D-bio/target/release/nhs_rt_full -t /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/topologies/8aqi.topology.json -o /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/8aqi --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v
```

### Batch 38 (8b9p, 8bre, 8gxj, 8h27, 8h49)
```bash
bash /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/batches/batch_038.sh
```

Or manually:
```bash
mkdir -p /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/8b9p && /home/diddy/Desktop/Prism4D-bio/target/release/nhs_rt_full -t /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/topologies/8b9p.topology.json -o /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/8b9p --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v
mkdir -p /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/8bre && /home/diddy/Desktop/Prism4D-bio/target/release/nhs_rt_full -t /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/topologies/8bre.topology.json -o /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/8bre --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v
mkdir -p /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/8gxj && /home/diddy/Desktop/Prism4D-bio/target/release/nhs_rt_full -t /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/topologies/8gxj.topology.json -o /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/8gxj --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v
mkdir -p /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/8h27 && /home/diddy/Desktop/Prism4D-bio/target/release/nhs_rt_full -t /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/topologies/8h27.topology.json -o /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/8h27 --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v
mkdir -p /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/8h49 && /home/diddy/Desktop/Prism4D-bio/target/release/nhs_rt_full -t /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/topologies/8h49.topology.json -o /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/8h49 --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v
```

### Batch 39 (8j11, 8onn, 8vxu, 9atc)
```bash
bash /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/batches/batch_039.sh
```

Or manually:
```bash
mkdir -p /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/8j11 && /home/diddy/Desktop/Prism4D-bio/target/release/nhs_rt_full -t /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/topologies/8j11.topology.json -o /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/8j11 --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v
mkdir -p /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/8onn && /home/diddy/Desktop/Prism4D-bio/target/release/nhs_rt_full -t /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/topologies/8onn.topology.json -o /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/8onn --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v
mkdir -p /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/8vxu && /home/diddy/Desktop/Prism4D-bio/target/release/nhs_rt_full -t /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/topologies/8vxu.topology.json -o /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/8vxu --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v
mkdir -p /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/9atc && /home/diddy/Desktop/Prism4D-bio/target/release/nhs_rt_full -t /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/topologies/9atc.topology.json -o /home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results/9atc --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v
```

