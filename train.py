import torch
from torch.utils.data import DataLoader
import mlflow
from dataset import TrafficDataset
from model import GNNLSTM
from tqdm import tqdm

BATCH_SIZE = 32
EPOCHS = 20
LR = 1e-3


def train():
    dataset = TrafficDataset("data/METR-LA/vel.csv")
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    num_nodes = dataset.X.shape[2]
    model = GNNLSTM(num_nodes)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = torch.nn.MSELoss()

    mlflow.start_run()

    for epoch in range(EPOCHS):
        total_loss = 0

        for X, y in tqdm(loader):
            optimizer.zero_grad()
            pred = model(X, edge_index=None)
            loss = criterion(pred, y[:, -1])
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        mlflow.log_metric("loss", avg_loss, step=epoch)
        print(f"Epoch {epoch} Loss {avg_loss}")

    torch.save(model.state_dict(), "model.pt")
    mlflow.end_run()


if __name__ == "__main__":
    train()
