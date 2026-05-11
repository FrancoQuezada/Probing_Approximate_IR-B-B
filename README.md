# Approximate Information-Relaxation Branch-and-Bound

This repository contains the C++/CPLEX implementation used in the computational experiments of the manuscript on stochastic production planning with probing decisions.

The code implements two branch-and-bound algorithms:

- `wht=24`: the baseline Ma-style enhanced branch-and-bound algorithm.
- `wht=30`: the proposed approximate information-relaxation branch-and-bound algorithm, AIR-B&B.

The executable is:

```bash
bin/main
```

## Repository contents

The repository includes the source files required to compile and run the computational experiments, the `Makefile`, and a minimal example script:

```text
Makefile
*.cpp
*.h / *.hpp
script.sh
README.md
```

Generated binaries, object files, logs, and result files are not tracked by Git.

## Requirements

The code requires:

- a C++ compiler
- `make`
- IBM ILOG CPLEX

The `Makefile` should be adjusted if the local CPLEX installation path differs from the one configured in the repository.

## Build

From the repository root, run:

```bash
make clean
make
```

If `make clean` is not available in your environment, run:

```bash
make
```

The build produces the executable:

```bash
bin/main
```

## Minimal example

The repository provides one example script:

```bash
chmod +x script.sh
./script.sh
```

The script runs a small representative instance using `wht=30`. A commented example for `wht=24` may also be included in the script.

## Command-line interface

The executable expects exactly 16 user arguments:

```text
bin/main seed N_sc N_pb N_tp N_it N_pn beta lost al alphaScale wht FixMode DynamicFix viMode vfMode evalMode
```

### Instance parameters

| Argument | Description |
|---|---|
| `seed` | Random seed used for instance generation |
| `N_sc` | Number of demand scenarios |
| `N_pb` | Number of probing scenarios |
| `N_tp` | Number of time periods |
| `N_it` | Number of items |
| `N_pn` | Number of probing components |
| `beta` | Capacity level |
| `lost` | Lost-sales penalty |
| `al` | Probing-cost scaling factor |
| `alphaScale` | Multiplier applied to probing costs |

### Algorithm and feature parameters

| Argument | Description |
|---|---|
| `wht` | Algorithm selector: `24` or `30` |
| `FixMode` | Probing-variable fixing mode |
| `DynamicFix` | Dynamic fixing switch |
| `viMode` | Valid-inequality mode |
| `vfMode` | Value-function inequality mode |
| `evalMode` | Evaluation mode |

Supported values for `wht` are:

```text
24 30
```

## Main feature flags

### `FixMode`

```text
0 = off
1 = static fixing
2 = node residual fixing
3 = static + node residual fixing
```

### `DynamicFix`

```text
0 = off
1 = on
```

### `viMode`

```text
0 = off
1 = path inequalities
2 = tree inequalities
3 = both
```

### `vfMode`

```text
0 = off
1 = master
2 = subproblem
3 = both
```

### `evalMode`

```text
0 = joint evaluation
1 = item-wise evaluation
```

For `wht=30`, the intended setting is:

```text
evalMode = 1
```

## Internal defaults

The command-line interface exposes only the parameters needed to reproduce the computational runs. The remaining algorithmic options are fixed internally.

### Shared defaults

```text
mu_D = 100.0
CorrMode = 1
internal branch mode = 0
internal cache flag = 1
viRoutineMode = 0
DebugTargetMask = -1
TauPackingLBMode = 0
```

The per-component correlation vector is generated internally during instance generation.

### Defaults for `wht=24`

```text
HeurFreeWindow = 1
HeurLBMode = 1
HeurRepairMode = 0
HeurRepairV3UpdateMode = 0
HeurRepairV3MaxIter = 20
HeurIntensifyTiLim = 10.0
Wht24TimeLimitSec = 3600.0
```

### Defaults for `wht=30`

```text
ApproxGapEps = 0.05
ApproxReuseMode = 1
ApproxLocalImproveMode = 1
ApproxLBMode = 1
ApproxGlobalLagLBMode = 1
ApproxGlobalLagMaxIter = 1
ApproxGlobalLagStep0 = 1.0
ApproxExactFallbackMode = 1
```

`ApproxGapEps` is fixed internally at `0.05`. The example script does not pass it as a command-line argument.

## Output files

The algorithms write results to text files in the repository root:

```text
Results_MaBranchBound.txt   # output for wht=24
Results_AIRBranchBound.txt  # output for wht=30
```

These files are generated at runtime and should not be committed to the repository.

## Reproducibility notes

To reproduce a run, record the full command-line call, including the random seed and all instance parameters. For example:

```bash
bin/main 1 5 5 12 10 10 1.0 30 0.5 0.5 30 1 0 0 0 1
```

This command runs a small instance with the AIR-B&B algorithm (`wht=30`) using static fixing and item-wise evaluation.

## Code availability

This repository is intended to accompany the manuscript and provide the implementation used for the reported computational experiments.
