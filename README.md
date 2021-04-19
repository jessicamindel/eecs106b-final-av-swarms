# eecs106b-final-av-swarms

## Installing the deep_rl_for_swarms dependency

```
git clone https://github.com/ALRhub/deep_rl_for_swarms.git
cd deep_rl_for_swarms
```

```
brew install mpich # for mpi4py to install. This happens on MacOS
```

Open deep_rl_for_swarms/setup.py
- Change "tensorboard == 1.5.0" to "tensorboard"
- Change "tensorflow == 1.5.2" to "tensorflow"
- Change "numpy == 1.14.5" to "numpy" # Too low for the fast_histogram package

```
pip install -e .
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
