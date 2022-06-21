#!/bin/bash
set -ex

#if [ ! -d "$(pwd)/home/jovyan/.local/share"  ]; then 
#	mkdir -p "$(pwd)/home/jovyan/.local/share"
#fi

docker build -f dockerfile -t geos627:latest .
