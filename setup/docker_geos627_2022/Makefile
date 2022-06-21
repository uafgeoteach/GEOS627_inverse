all: build run

build:
	bash build.sh

run:
	bash run.sh 2>&1 | tee log


clean:
	rm -rf home/.cache home/.conda home/.config home/.empty \
		home/.ipython home/.jupyter home/insar_analysis home/.ipynb_checkpoints \
		home/.local/bin home/.local/lib home/.local/share home/.local/etc \
		home/.local/envs/inverse home/.local/envs/.conda_envs_dir_test \
		home/.bash_profile home/.bashrc home/.condarc home/GEOS627_inverse \
		home/bash_history home/.viminfo