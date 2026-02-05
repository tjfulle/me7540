# plane

Simple finite element plane elasticity equation solver

## Installation

## One time setup

```console
git clone git@github.com:tjfulle/me7540
cd me7540
python3 -m venv venv
source venv/bin/activate
cd Exercises/PlaneElasticity
python3 -m pip install -e .
```

## Run an example

```console
python3 -m plane
```

The exercise will fail to run since the solver is not set up for the vector-valued boundary conditions

## Where to find code to modify

The code to modify is found in [plane.py](./src/plane/plane.py).  The code inside of `plane` is the completed heat transfer code from Assignment 1 and must be modified.

## Recommendations

See [Assignment 3, Exercise2](./Exercise.pdf) for specific deliverables.

I recommend:

* Looking at the `bmatrix` and `pmatrix` functions to see where/how they are used.  How might they
  be modified for vector-valued problems?
* Making a verification problem to run uniaxial stress and compare with analytic solution.
* Writing a post processing function that takes the solution, connectivity, and stiffness and uses
  those to generate strain/stress in each element.  Those stresses can be used to compute the Mises
  stresses.
