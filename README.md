# eecs106b-final-av-swarms

## Installing the deep_rl_for_swarms dependency

```
git clone https://github.com/ALRhub/deep_rl_for_swarms.git
cd deep_rl_for_swarms
```

find ~/ -name mpicc

```
# MacOS
brew install mpich # for mpi4py to install. This happens on MacOS

# Other (might also work for MacOS)
conda install -c anaconda mpi4py

```

Open deep_rl_for_swarms/setup.py
- Change "tensorboard == 1.5.0" to "tensorboard==1.11.0"
- Change "tensorflow == 1.5.2" to "tensorflow==1.11.0"
- Change "numpy == 1.14.5" to "numpy==1.19.5" # Too low for the fast_histogram package
- Remove "mpi4py == 3.0.0". Hoping that the conda install above works

```
python -m pip install -e .
```

MacOS fix for Matplotlib's "RuntimeError: Python is not installed as a framework"
https://stackoverflow.com/questions/34977388/matplotlib-runtimeerror-python-is-not-installed-as-a-framework
In the file `\~/.matplotlib/matplotlibrc`, add the line `backend: TkAgg`

In `deep_rl_for_swarms/deep_rl_for_swarms/common/act_wrapper.py` change
`from common import logger` to `from deep_rl_for_swarms.common import logger`

To save the trained policy approximately every half hour:
On `deep_rl_for_swarms/rl_algo/trpo_mpi/trpo_mpi.py` line 325, right above `iters_so_far += 1`, add

```
if iters_so_far % 500 == 0 and iters_so_far > 0:
    pi.save()
```

## Running on Hive

Get an EECS Instructional account

Go to Hivemind and copy a server name by clicking on it https://hivemind.eecs.berkeley.edu/

ssh your_username@machine_name_you_copied

```
mkdir code
cd code

# Get conda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b
rm Miniconda3-latest-Linux-x86_64.sh
echo PATH="~/miniconda3/bin":$PATH >> ~/.bashrc
exec bash
conda --version

# Make and activate a conda environment
# NOTE this mpich is important. It makes it possible to install mpi4py in the deep_rl_for_swarms dependency
conda create -n proj python=3.6.13 mpich -y
conda init bash
source ~/.bashrc
conda activate proj

# Get our repo
git clone https://github.com/jessicamindel/eecs106b-final-av-swarms.git
cd eecs106b-final-av-swarms/

# Get deep_rl_for_swarms
## Do the instructions from the above^^^

# Install some other dependencies
python -m pip install gym opencv-python

# Open a window that persists even after you exit the SSH connection
tmux
cd ~/code/eecs106b-final-av-swarms

# Start your training script! For example
python train.py 3 maps/task1a_moreborders.png


```
