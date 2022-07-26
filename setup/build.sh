#!/bin/bash
set -ex

docker build -f dockerfile -t geos627:latest .

# docker run -it --rm -p 8888:8888 -v $(pwd):/home/jovyan/GEOS627_inverse geos627:latest