# DeepHam

DeepHam consists of two models: one built using Torch and PyG primitives, and
another built atop the TorchRL framework. The former model is the one that has
been most extensively tested and was used as our final model; this is the one we
describe below. Our TorchRL model, which has a similar architecture, can be
found in the `ppo/` directory.

### Training

The model can be run by executing `main.py`. Training parameters can be changed
by modifying the call to `train_model` in `main.py`, as follows:

* `visualize`: Enable visualization
* `notebook`: Set `True` iff running in a notebook and (`visualize` is `True` OR
  you want to write to log files rather than stdout)
* `episodes`: The number of episodes for which to train
* `use_replay`: Enable the replay buffer
* `num_verts`: The number of vertices to generate in training graphs
* `num_edges`: The approximate number of edges to randomly add to the
  Hamiltonian path "skeleton"
* `delta_e`: Configure the variability of the random edge count in generated
  graphs---the number of added random edges will be in the range
  `[num_edges - delta_e, num_edges + delta_e]`

Alternatively, you can train in the `training.py` notebook. Note that this is
currently set up for our full training suite, which was run for over an hour on
a GPU in Google Cloud Platform and likely will take much longer to run on a CPU.

### Architecture

* Our training logic can be found in `main.py`
* The actor-critic model, including our loss metric, can be found in `model.py`
* Our random graph generation code can be found in `data.py`
* Our multilayer perceptron and custom masking logic can be found in `MLP.py`
* Our graph environment code (including vertex detachment and state/reward
  management) can be found in `GraphState.py`
* Our replay buffer implementation can be found in `ReplayBuffer.py`
* As noted above, our TorchRL-based model (which is an entirely distinct model
  from the one outlined above, but is architecturally similar), which uses
  proximal policy optimization, can be found in the `ppo/` directory; most
  of the significant differences in that model are found in `ppo/main.py` and
  the environment configuration in `ppo/GraphEnv/`
* Our supervised learning model can be found in `supervised_learning.py`. Note
  that this is currently suffering from a significant memory leak and is not
  part of our final architecture. We include it principally for posterity.

We also provide the following interactive notebooks:

* `training.ipynb` is used for training the model; its current configuration
  reflects the final testing we performed
* `datagen.ipynb` is used for experimenting with random graph generation
* `state_test.ipynb` was used for debugging the graph environment and can safely
  be ignored
