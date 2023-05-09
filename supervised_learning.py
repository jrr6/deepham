import torch
import numpy as np
from torch import nn
from torch_geometric.data import Data
from torch_geometric.nn import GATv2Conv
from data import generate_semirandom_hampath_graph
import matplotlib.pyplot as plt


N_SUPERVISED_EPOCHS = 500
BATCH_SIZE = 20
SUPERVISED_LEARNING_RATE = 0.001

class DeepHamSupervised(nn.Module):
    def __init__(self, node_embedding_size=512, hidden_layer_size=256, relu_alpha=0.1):
        super(DeepHamSupervised, self).__init__()

        self.conv1 = GATv2Conv(-1, node_embedding_size)
        self.conv2 = GATv2Conv(node_embedding_size, node_embedding_size)
        self.conv3 = GATv2Conv(node_embedding_size, node_embedding_size)

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(node_embedding_size, hidden_layer_size),
            torch.nn.LeakyReLU(relu_alpha),
            torch.nn.Linear(hidden_layer_size, hidden_layer_size),
            torch.nn.LeakyReLU(relu_alpha),
            torch.nn.Linear(hidden_layer_size, 1),
            torch.nn.Softmax(dim=0),
        )

    def forward(self, graph: Data):
        vertices = graph.x
        edge_index = graph.edge_index

        vertex_embeddings: torch.Tensor = self.conv1(vertices, edge_index)
        vertex_embeddings = vertex_embeddings.tanh()
        vertex_embeddings = self.conv2(vertex_embeddings, edge_index)
        vertex_embeddings = vertex_embeddings.tanh()
        vertex_embeddings = self.conv3(vertex_embeddings, edge_index)
        vertex_embeddings = vertex_embeddings.tanh()

        return self.mlp(vertex_embeddings)

def run_batch(
    model: DeepHamSupervised,
    batch: np.ndarray,
    criterion: torch.nn.NLLLoss,
    optimizer: torch.optim.Adam,
    training=False,
) -> tuple[float, float]:
    correct = 0
    labels = []
    outputs = []
    for graph, label in batch:
        probs = model(graph)
        chosen_vertex = torch.argmax(probs).item()

        correct += int(chosen_vertex == label)
        labels.append(int(label))
        outputs.append(probs)

    # FIXME: memory leak here
    loss = criterion(
        torch.log(torch.stack(outputs)), torch.Tensor(labels).reshape(-1, 1).long()
    )

    acc = correct / len(batch)
    if training:
        optimizer.zero_grad()
        loss.backward()

    return loss, acc


# Call this function to run supervised learning
def supervised_learning():
    model = DeepHamSupervised()  # TODO: update this to not have mask
    criterion = torch.nn.NLLLoss(reduction="sum")
    optimizer = torch.optim.Adam(model.parameters(), lr=SUPERVISED_LEARNING_RATE)

    corpus = np.array(
        [generate_semirandom_hampath_graph(30, 0, 15, 10) for _ in range(100)],
        dtype=object,
    )
    test_batch = np.array(
        [generate_semirandom_hampath_graph(30, 0, 15, 10) for _ in range(100)],
        dtype=object,
    )

    training_loss = []
    test_loss = []
    training_acc = []
    test_acc = []

    fig, [loss_ax, acc_ax] = plt.subplots(1, 2)  # TODO: pretty plots
    for epoch in range(N_SUPERVISED_EPOCHS):
        np.random.shuffle(corpus)

        model.train(True)
        for batch_idx in range(len(corpus) // BATCH_SIZE):
            batch = corpus[batch_idx * BATCH_SIZE : (batch_idx + 1) * BATCH_SIZE]
            loss, acc = run_batch(model, batch, criterion, optimizer, training=True)

            print(
                f"[Train Epoch: {epoch}/{N_SUPERVISED_EPOCHS} Batch: {batch_idx + 1} / {len(corpus) // BATCH_SIZE}] loss: {loss} acc: {acc}"
            )

            training_loss.append(loss)
            training_acc.append(acc)

        model.train(False)
        loss, acc = run_batch(model, test_batch, criterion, optimizer)
        test_loss.append(loss)
        test_acc.append(acc)
