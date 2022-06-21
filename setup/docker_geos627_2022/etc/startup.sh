#!/usr/bin/env bash

set -ex

#ls -a /home/jovyan/.local
#mkdir /home/jovyan/.local/share
#mkdir /home/jovyan/.local/lib
#chown -R jovyan:users /home/jovyan/.local/share
#chown -R jovyan:users /home/jovyan/.local/lib

python -m pip install --user \
    nbgitpuller \
    ipywidgets \
    mpldatacursor

# copy over our version of pull.py
# REMINDER: REMOVE IF CHANGES ARE MERGED TO NBGITPULLER
python=$(python --version 2>&1)
v=$(echo $python | cut -d'.' -f 2)
cp "${INVERSE_FILES}"/pull.py /home/jovyan/.local/lib/python3."$v"/site-packages/nbgitpuller/pull.py

# enable nbgitpuller
jupyter serverextension enable --py nbgitpuller

# Add Path to local pip execs.
export PATH=$HOME/.local/bin:$PATH

# Pull in any repos you would like cloned to user volumes
gitpuller https://github.com/uafgeoteach/GEOS627_inverse.git main $HOME/GEOS627_inverse

CONDARC=$HOME/.condarc
if ! test -f "$CONDARC"; then
cat <<EOT >> $CONDARC
channels:
  - conda-forge
  - defaults
channel_priority: strict
create_default_packages:
  - jupyter
  - kernda
envs_dirs:
  - /home/jovyan/.local/envs
  - /opt/conda/envs
EOT
fi

conda init

echo "create directory"
mkdir -p "$HOME"/.local/envs/

LOCAL="$HOME"/.local
ENVS="$LOCAL"/envs
NAME=inverse
PREFIX="$ENVS"/"$NAME"
SITE_PACKAGES=$PREFIX"/lib/python3."$v"/site-packages"

# Create inverse env
if [ ! -d "$PREFIX" ]; then
  echo "mamba create"
  mamba env create -f "$ENVS"/"$NAME".yml -q
  mamba run -n "$NAME" kernda --display-name "$NAME" -o --env-dir "$PREFIX" "$PREFIX"/share/jupyter/kernels/python3/kernel.json
else
  echo "mamba update"
  mamba env update -f "$ENVS"/"$NAME".yml -q
fi

mamba clean --yes --all

JN_CONFIG=$HOME/.jupyter/jupyter_notebook_config.json
if ! grep -q "\"CondaKernelSpecManager\":" "$JN_CONFIG"; then
jq '. += {"CondaKernelSpecManager": {"name_format": "{display_name}", "env_filter": ".*opt/conda.*"}}' "$JN_CONFIG" >> temp;
mv temp "$JN_CONFIG";
fi

BASH_RC=/home/jovyan/.bashrc
grep -qxF 'conda activate inverse' $BASH_RC || echo 'conda activate inverse' >> $BASH_RC

BASH_PROFILE=$HOME/.bash_profile
if ! test -f "$BASH_PROFILE"; then
cat <<EOT > $BASH_PROFILE
if [ -s ~/.bashrc ]; then
    source ~/.bashrc;
fi
EOT
fi

python -m pip install df-jupyter-magic
cat <<EOT > "$HOME"/.ipython/profile_default/ipython_config.py
c.InteractiveShellApp.extensions = ['df_jupyter_magic']
EOT
