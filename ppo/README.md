# DeepHam

[TODO:] What is DeepHam?

## Running

Begin by cloning the repository and installing all the needed dependencies:

```bash
conda install torch==2.0 torch_geometric numpy gym
pip install torchrl         # Unfortunately, no conda channel for this yet since it is in beta
pip install -e GraphEnv     # Custom OpenAI Gym Environment to represent our graph
```

To run the model simply run `main.py`:

```bash
python3 main.py
```
