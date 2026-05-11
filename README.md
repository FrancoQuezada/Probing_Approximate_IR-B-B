# Probing Approximate Information-Relaxation Branch-and-Bound

This repository contains a cleaned C++/CPLEX implementation for two execution modes only:

- `wht=24`: Ma-style enhanced branch-and-bound.
- `wht=30`: approximate information-relaxation branch-and-bound (AIR-B&B).

Removed legacy variants are not supported in this repository.

## Requirements

- C++ compiler with C++11 support
- Make
- IBM ILOG CPLEX / Concert

The default `Makefile` expects CPLEX Studio under:

```text
/opt/ibm/ILOG/CPLEX_Studio2211
```

If your installation is elsewhere, update the CPLEX and Concert paths in `Makefile`.

## Build

```bash
make
```

To clean object/dependency files:

```bash
make clean
```

The executable is created at:

```text
bin/main
```

## Minimal Example

```bash
chmod +x script.sh
./script.sh
```

`script.sh` runs one small `wht=30` AIR-B&B example. It does not launch a grid experiment. A commented `wht=24` example is included in the script.

## Command Line Interface

`bin/main` expects exactly 16 user arguments:

```text
bin/main seed N_sc N_pb N_tp N_it N_pn beta lost al alphaScale wht FixMode DynamicFix viMode vfMode evalMode
```

Supported values for `wht`:

```text
24 30
```

Legacy `wht` values such as `11/12/13/25/27/28/29/60` are not supported.

## Main Arguments

- `seed`: random seed.
- `N_sc`: number of demand scenarios.
- `N_pb`: number of probing scenarios.
- `N_tp`: number of time periods.
- `N_it`: number of items.
- `N_pn`: number of probing components.
- `beta`: capacity level.
- `lost`: lost-sales penalty.
- `al`: probing-cost scaling parameter.
- `alphaScale`: multiplier for probing costs.
- `wht`: algorithm selector, either `24` or `30`.
- `FixMode`: fixing mode, valid range `[0,3]`.
- `DynamicFix`: dynamic fixing flag, valid range `[0,1]`.
- `viMode`: valid inequality mode, valid range `[0,3]`.
- `vfMode`: value-function mode, valid range `[0,3]`.
- `evalMode`: evaluation mode, valid range `[0,1]`. For `wht=30`, `evalMode=1` is recommended.

## Algorithm Modes

`wht=24` runs the Ma-style enhanced branch-and-bound route.

`wht=30` runs AIR-B&B, the approximate information-relaxation branch-and-bound route.

## AIR Gap Tolerance

`ApproxGapEps` is fixed internally in `Main.cpp`; it is not passed through the CLI or `script.sh`.

Current fixed value:

```text
ApproxGapEps = 0.05
```

## Output Files

The executable writes result rows to:

- `Results_BB_New.txt` for `wht=24`.
- `Results_BB_AIR.txt` for `wht=30`.

Result files are generated at runtime and are ignored by Git.
