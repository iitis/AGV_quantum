# AGV_quantum

## Scripts

In directory ```examples``` there are AGVs scheduling problems. In order of increasing size of the problem these are:

- example_tiny.py
- example_smallest.py
- example_small.py
- example_medium_small.py
- example_medium.py
- example_large.py
- example_largest.py

There are optional boolean parameters (```1``` yes, ```0``` no): ```--solve_linear``` - solve on CPLEX , ```--train_diagram``` - plot "train diagram" for given problem ```--solve_quadratic``` - solve on hybrid quantum classical (the particular solver and penalty parameter can be set in the script).


## Usage 

To run this project in terminal use path/to/project> python -m examples.file 

To run tests use path/to/project> python3 -m unittest

## Computational results 

In folder ```annealing_results``` there are results on quantum and hybrid devices.

In folder ```lp_files``` there are results of classical solver on the ILP.


