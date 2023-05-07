import torch
from torchrl.data.tensor_specs import TensorSpec

from torchrl.envs import Transform
from torch_geometric.utils import dense_to_sparse

class DenseToSparseTransform(Transform):
    def _apply_transform(self, obs: torch.Tensor):
        return dense_to_sparse(obs)[0]
