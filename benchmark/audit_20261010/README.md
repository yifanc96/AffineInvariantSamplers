# Conditional audit of saved benchmark cells

- Raw samples and historical source revisions/effective settings are unavailable; saved ESS/moments are retained, never recomputed.
- Conditional corrected production cost assumes current PEACHES trajectory accounting: N*W*(mean_L+1), thin_by=1. Historical code execution is not proven by these JSON cells.
- PEANUTS conditional corrected production cost is 2*saved n_grad_evals_prod: current tree leaves compute two gradients, while historical counters counted one. Historical implementation matching is not established.
- Preparation is excluded from ESS/gradient denominators; historical --warm-start W implementation ran 2W stretch iterations.
- For fresh-gradient fixed-trajectory samplers, missing/nonfinite mean_L gives unavailable conditional cost; no fabricated cost, no imputed successful tuning.
- Gradient-free saved ESS/grad is historically mislabeled; conditional_n_log_prob_evals_prod and conditional_ess_per_log_prob explicitly separate log-density work, assuming thin_by=1.
- Rank and ensemble ESS fields are kept distinct. Missing ess_method in older Gaussian cells is labeled an assumption, not confirmed metadata.
- Walkers within an interacting ensemble are dependent; within-ensemble R-hat is descriptive and is not used to filter seed averages.
- Only saved common settings establish matching; unrecorded max_L, target_accept, initialization search, hardware and revision may differ.
- These conditional corrections do not resolve frozen-chain false positives or mixing bias in historical ESS, because temporal samples are unavailable.

| Target | D | Sampler | Tag | Walkers | Seeds | Median L | Saved ESS/grad | Conditional ESS/grad | Median std error |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| gaussian | 2 | peaches | gsW128 | 128 | 5 | 3.9734 | 0.245186 | 0.195887 | 0.000552433 |
| gaussian | 2 | peaches | gsW4d | 8 | 5 | 6.50207 | 0.123644 | 0.107163 | 0.00286653 |
| gaussian | 2 | peanuts | exp4 | 128 | 4 | 4.10146 | 0.196146 | 0.0980729 | 0.00214989 |
| gaussian | 2 | peanuts | scaleL | 128 | 3 | 2.59574 | 0.359282 | 0.179641 | 0.0008797 |
| gaussian | 2 | peanuts | scaleLb | 8 | 6 | 4.71893 | 0.106369 | 0.0531843 | 0.00199207 |
| gaussian | 2 | peanuts | scaleLd | 128 | 6 | 2.59237 | 0.359247 | 0.179623 | 0.000499925 |
| gaussian | 8 | peaches | gsW128 | 128 | 5 | 2.2357 | 0.405335 | 0.280065 | 0.00105265 |
| gaussian | 8 | peaches | gsW4d | 32 | 5 | 6.14953 | 0.175692 | 0.151118 | 0.00137353 |
| gaussian | 8 | peanuts | exp4 | 128 | 4 | 9.42057 | 0.116135 | 0.0580673 | 0.00260246 |
| gaussian | 8 | peanuts | exp4b | 32 | 4 | 13.3622 | 0.070414 | 0.035207 | 0.00409973 |
| gaussian | 8 | peanuts | exp4c | 32 | 3 | 7.25046 | 0.137237 | 0.0686184 | 0.0057198 |
| gaussian | 8 | peanuts | scaleL | 128 | 3 | 5.15717 | 0.233422 | 0.116711 | 0.00109812 |
| gaussian | 8 | peanuts | scaleLb | 32 | 6 | 7.21634 | 0.139646 | 0.0698231 | 0.00140734 |
| gaussian | 8 | peanuts | scaleLd | 128 | 6 | 5.15072 | 0.234545 | 0.117272 | 0.000740443 |
| rosenbrock | 2 | peaches | b2_w8000_i2 | 64 | 24 | 11.3296 | 0.00167566 | 0.00147129 | 0.0225077 |
| rosenbrock | 2 | peaches | rz8 | 64 | 8 | 13.6944 | 0.00179504 | 0.0015302 | 0.0120958 |
| rosenbrock | 2 | peanuts | exp1 | 128 | 3 | 11.4347 | 8.39865e-05 | 4.19932e-05 | 0.0600782 |
| rosenbrock | 2 | stretch | rz8 | 64 | 3 | unavailable | 0.00181516 | unavailable | 0.0112072 |
| rosenbrock | 4 | peaches | rz8 | 64 | 8 | 86.5618 | 0.00251547 | 0.00247071 | 0.0129287 |
| rosenbrock | 4 | stretch | rz8 | 64 | 3 | unavailable | 0.00034683 | unavailable | 0.0583204 |
| rosenbrock | 8 | peaches | rwscan_w1024 | 1024 | 3 | 19.2886 | 0.000940024 | 0.000893691 | 0.00727884 |
| rosenbrock | 8 | peaches | rwscan_w128 | 128 | 3 | 14.5229 | 0.000821723 | 0.000768787 | 0.0161387 |
| rosenbrock | 8 | peaches | rwscan_w256 | 256 | 3 | 19.3926 | 0.00141424 | 0.00134489 | 0.00600391 |
| rosenbrock | 8 | peaches | rwscan_w512 | 512 | 3 | 62.9925 | 0.00172722 | 0.00338321 | 0.014637 |
| rosenbrock | 8 | peaches | rwscan_w64 | 64 | 3 | 203.401 | 0.0035103 | 0.00349312 | 0.00301151 |
| rosenbrock | 8 | peaches | rz8 | 64 | 8 | 189.662 | 0.00320277 | 0.00318709 | 0.00735644 |
| rosenbrock | 8 | peanuts | exp1 | 128 | 3 | 49.9235 | 0.000263606 | 0.000131803 | 0.030817 |
| rosenbrock | 8 | peanuts | exp3_w100 | 128 | 3 | 52.1859 | 0.000736429 | 0.000368215 | 0.0393665 |
| rosenbrock | 8 | peanuts | exp3_w1000 | 128 | 3 | 52.7781 | 0.000215185 | 0.000107592 | 0.0267588 |
| rosenbrock | 8 | peanuts | exp3_w200 | 128 | 3 | 50.7115 | 0.000644039 | 0.00032202 | 0.0273056 |
| rosenbrock | 8 | peanuts | exp3_w50 | 128 | 3 | 50.2539 | 0.000202982 | 0.000101491 | 0.0248724 |
| rosenbrock | 8 | peanuts | exp3_w500 | 128 | 3 | 49.9237 | 9.12415e-05 | 4.56207e-05 | 0.0796292 |
| rosenbrock | 8 | peanuts | exp3b_w100 | 256 | 8 | 51.3483 | 0.00011436 | 5.71799e-05 | 0.0295156 |
| rosenbrock | 8 | peanuts | exp3b_w1000 | 256 | 8 | 49.9948 | 9.47477e-05 | 4.73739e-05 | 0.0270746 |
| rosenbrock | 8 | peanuts | exp3b_w200 | 256 | 8 | 51.4793 | 8.3971e-05 | 4.19855e-05 | 0.0353789 |
| rosenbrock | 8 | peanuts | exp3b_w50 | 256 | 8 | 49.8632 | 9.35608e-05 | 4.67804e-05 | 0.0379391 |
| rosenbrock | 8 | peanuts | exp3b_w500 | 256 | 8 | 49.6182 | 8.09516e-05 | 4.04758e-05 | 0.0280952 |
| rosenbrock | 8 | stretch | rz8 | 64 | 3 | unavailable | 0.00013062 | unavailable | 0.120058 |
| rosenbrock | 16 | peaches | rwscan_w128 | 128 | 2 | 100.975 | 0.00271218 | 0.00267879 | 0.0108186 |
| rosenbrock | 16 | peaches | rz8 | 128 | 8 | 197.145 | 0.00340371 | 0.00338861 | 0.00382008 |
| rosenbrock | 16 | stretch | rz8 | 128 | 3 | unavailable | 7.1303e-05 | unavailable | 0.433203 |
| rosenbrock | 32 | peaches | rz8 | 256 | 5 | 217.422 | 0.00318694 | 0.00317202 | 0.0034749 |
| rosenbrock | 32 | stretch | rz8 | 256 | 3 | unavailable | 6.17593e-05 | unavailable | 0.798096 |
