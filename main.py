import torch
from data import load_corpus
from model import DeepHamModel, Dist_Loss

DATA_PATH = './data'
LEARNING_RATE = 0.001

def train(model, optimizer, criterion, data):
    optimizer.zero_grad()  # Clear gradients.
    probs = model(data.x, data.edge_index)  # Perform a single forward pass.

    # print(f"Node Probabilities: {probs}")

    print("Probabilities", probs)
    chosen_node = torch.argmax(probs)
    print(chosen_node)
    print("-------------")

    # FIXME: :(
    loss = criterion(torch.tensor([chosen_node]), data.x)  # Compute the loss solely based on the training nodes.
    loss.backward()  # Derive gradients.
    optimizer.step()  # Update parameters based on gradients.
    return loss

def main():
    corpus = load_corpus(DATA_PATH)

    model = DeepHamModel()
    criterion = Dist_Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for graph in corpus[0:1]:
        train(model, optimizer, criterion, graph)

if __name__ == "__main__":
    main()
