# AGV_quantum

## Scripts

In directory ```examples``` there are AGVs scheduling problems. In order of increasing size of the problem these are:

- example_tiny.py         - 2 AGVs reduced size
- example_smallest.py     - 2 AGVs, 4 zones
- example_small.py        - 4 AGVs, 4 zones
- example_medium_small.py - 6 AGVs, 7 zones
- example_medium.py       - 7 AGVs, 7 zones
- example_large.py        - 12 AGVs, 7 zones
- example_largest.py      - 15 AGVs, 7 zones


## Usage 

To run this project in terminal use path/to/project> python -m run_examples 

There are optional boolean parameters (```1``` yes, ```0``` no): 

- ```--solve_linear``` - if ```1``` solve ILP on CPLEX, if ```0``` solve on hybrid quantum classical 
- ```--hyb_solver``` chose particular hybrid solver  ```"bqm"``` or ```"cqm"``` are supported, for ```"bqm"``` penalty parameter can be set in  the script (works if ```--solve_linear = 0```), 
- ```--train_diagram``` - plot "train diagram" for given problem (works if ```--solve_linear = 1```).

Examples: 

```python  -m run_examples  --solve_linear 1 --train_diagram 1 --example "small"```

```python  -m run_examples  --solve_linear 0 --example "small" --hyb_solver "bqm"```

following examples are supported: ```"tiny", "smallest", "small", "medium_small", "medium", "large", "largest"```

Tho check solutions use:

```python -m check_sol --example "small" --hyb_solver "cqm"```


To run tests use path/to/project> python3 -m unittest

## Computational results 

In folder ```annealing_results``` there are results on quantum and hybrid devices.

In folder ```lp_files``` there are saved linear models for checkout of quantum solver.


### Citing this work

The code was partially supported by:
-  Foundation for Polish Science (FNP) under grant number TEAM NET POIR.04.04.00-00-17C1/18-00 
-  National Science Centre, Poland under grant number 2022/47/B/ST6/02380, and under grant number 2020/38/E/ST3/00269
