#!/bin/bash

set -ex

# docker run -it --rm -p 8888:8888 -v $(pwd):/home/jovyan/GEOS627_inverse geos627:latest
docker run -it --rm -p 8888:8888 -v $(pwd):/home/jovyan geos627:latest

# docker run -it --rm -p 8888:8888 -v $(pwd)/home:/home/jovyan geos627:latest