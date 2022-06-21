#!/bin/bash

set -ex
# docker build -f dockerfile -t geos627:latest .

#docker run -it --rm -p 8888:8888 -u 66644:66049 -v $(pwd)/home:/home/jovyan geos627:latest
docker run -it --rm -p 8888:8888 -v $(pwd)/home:/home/jovyan geos627:latest

# docker run -it --rm -p 8888:8888 -v $(pwd)/home:/home/jovyan --entrypoint /bin/bash geos627 
