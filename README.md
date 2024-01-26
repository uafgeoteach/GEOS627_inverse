# GEOS627_inverse

A public GitHub repository within the organization
[uafgeoteach](https://github.com/uafgeoteach). Contains materials for GEOS 627 Inverse Problems and Parameter Estimation, a class at the University of Alaska Fairbanks by [Carl Tape](https://sites.google.com/alaska.edu/carltape/) ([ctape@alaska.edu](mailto:ctape@alaska.edu))

Course webpage: [GEOS 627](https://sites.google.com/alaska.edu/carltape/home/teaching/inv)  

The repository can be obtained from GitHub with this command:
```
git clone --depth=1 https://github.com/uafgeoteach/GEOS627_inverse.git
```

UAF students will run these notebooks within the OpenScienceLab setup at UAF.

### Setup
---
A `.yml` file (see setup/ folder) lists dependencies. This file, executed within conda or docker, enables a user to establish the software tools needed to execute the iPython notebooks.

### How to run using Conda
---

- install conda (miniconda or anaconda, former recommended) if not done already
- navigate to the setup folder
  ```bash
  cd GEOS627_inverse/setup
  ```
- setup the conda environment
  ```bash
  conda env create -f inverse.yml
  ```
- activate the conda environment once the setup is complete
  ```bash
  conda activate inverse
  ```
- navigate back to the root of repository and launch jupyter
  ```bash
  cd ..
  jupyter notebook
  ```
- browse and run notebooks as desired
