# AGV_quantum

## Scripts

In directory ```examples``` there are AGVs scheduling problems. In order of increasing size of the problem these are:

- example_tiny.py         - 2 AGVs reduced size
- example_smallest.py     - 2 AGVs
- example_small.py        - 4 AGVs
- example_medium_small.py - 6 AGVs
- example_medium.py       - 7 AGVs
- example_large.py        - 12 AGVs
- example_largest.py      - 15 AGVs


## Usage 

To run this project in terminal use path/to/project> python -m run_examples 

There are optional boolean parameters (```1``` yes, ```0``` no): ```--solve_linear``` - solve on CPLEX , ```--train_diagram``` - plot "train diagram" for given problem ```--solve_quadratic``` - solve on hybrid quantum classical (the particular solver and penalty parameter can be set in the script).

Example: 

```python  -m run_examples  --solve_linear 1 --train_diagram 1 --example "small"```

following examples are supported: ```"tiny", "smallest", "small", "medium_small", "medium", "large", "largest"```

To run tests use path/to/project> python3 -m unittest

## Computational results 

In folder ```annealing_results``` there are results on quantum and hybrid devices.

In folder ```lp_files``` there are results of classical solver on the ILP.


### Citing this work

The code was partially supported by:
-  Foundation for Polish Science (FNP) under grant number TEAM NET POIR.04.04.00-00-17C1/18-00 
-  National Science Centre, Poland under grant number 2022/47/B/ST6/02380, and under grant number 2020/38/E/ST3/00269
