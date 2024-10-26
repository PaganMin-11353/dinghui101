from .base_model import BaseModel
from utils.dataloader import load_data_df, load_item_df, load_user_features, DATA_FORMAT

import os
import pandas as pd
import numpy as np
from typeguard import typechecked
from typing import Tuple, Dict, Union

import torch
import torch.nn as nn
from torch_geometric.data import Data as pygData
from torch_geometric.nn import GCNConv
from torch_geometric.utils import negative_sampling


class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(GCN, self).__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(input_dim, hidden_dim))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        self.convs.append(GCNConv(hidden_dim, hidden_dim))

        self.relu = nn.ReLU()

    def forward(self, x, edge_index):
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = self.relu(x)
        x = self.convs[-1](x, edge_index)
        return x


@typechecked
class GCNModel(BaseModel):
    def __init__(
        self,
        size: str,  # "100k" or "1m"
        num_layers: int = 2,
        hidden_dim: int = 64,
        learning_rate: float = 0.01,
        epochs: int = 10,
        batch_size: int = 512,
        device: str = 'auto'  # 'auto', 'cuda', 'mps', 'cpu'
    ) -> None:
        super().__init__()

        if size not in DATA_FORMAT:
            raise ValueError(f"Invalid size: {size}. Choose from {list(DATA_FORMAT.keys())}.")

        self.size = size
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = self._get_device(device)

        # Load the data
        self.ratings = self._load_ratings()
        self.items = self._load_items()
        self.user_features = self._load_user_features()

        # Prepare the graph data
        self.data = self._prepare_graph_data()

        # Initialize the GCN model
        self.model = GCN(
            self.data.num_features, 
            self.hidden_dim, 
            self.num_layers
        ).to(self.device)

        # Loss function and optimizer
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def _get_device(self, device_str: str) -> torch.device:
        if device_str == 'auto':
            if torch.cuda.is_available():
                print("Using CUDA (Nvidia GPU)")
                return torch.device('cuda')
            elif torch.backends.mps.is_available():
                print("Using MPS (Apple Silicon GPU)")
                return torch.device('mps')
            else:
                print("Using CPU")
                return torch.device('cpu')
        elif device_str == 'cuda':
            if torch.cuda.is_available():
                print("Using CUDA (Nvidia GPU)")
                return torch.device('cuda')
            else:
                raise ValueError("CUDA is not available")          
        elif device_str == 'mps':
            if torch.backends.mps.is_available():
                print("Using MPS (Apple Silicon GPU)")
                return torch.device('mps')
            else:
                raise ValueError("MPS is not available")
        elif device_str == 'cpu':
            print("Using CPU")
            return torch.device('cpu')
        else:
            raise ValueError("Invalid device option: 'auto', 'cuda', 'mps', 'cpu'.")

    # TODO
    def _load_ratings(self) -> pd.DataFrame:
        ratings = load_data_df(self.size)
        return ratings

    def _load_items(self) -> pd.DataFrame:
        items = load_item_df(
            size=self.size,
            item_col="MovieId",
            title_col="Title",
            genres_col="Genres",
            year_col="Year"
        )
        return items

    def _load_user_features(self) -> Union[pd.DataFrame, None]:
        user_features = load_user_features(self.size)
        return user_features

    def _prepare_graph_data(self) -> pygData:
        # Create mappings from user and item IDs to indices
        user_ids = self.ratings['UserId'].unique()
        item_ids = self.ratings['MovieId'].unique()

        self.user_id_map = {id: idx for idx, id in enumerate(user_ids)}
        self.item_id_map = {id: idx + len(user_ids) for idx, id in enumerate(item_ids)}

        num_users = len(user_ids)
        num_items = len(item_ids)
        num_nodes = num_users + num_items

        # Edges (user-item interactions)
        edge_index = []
        for _, row in self.ratings.iterrows():
            user_idx = self.user_id_map[row['UserId']]
            item_idx = self.item_id_map[row['MovieId']]
            edge_index.append([user_idx, item_idx])
            edge_index.append([item_idx, user_idx])  # Add reverse edge for undirected graph

        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

        # Node features (simple embeddings)
        x = torch.randn((num_nodes, self.hidden_dim))

        data = pygData(x=x, edge_index=edge_index)

        return data

    def train(self) -> None:
        self.model.train()

        # Move data to device
        self.data = self.data.to(self.device)

        # Negative sampling for link prediction
        num_edges = self.data.edge_index.size(1) // 2  # Original number of edges
        for epoch in range(self.epochs):
            # Positive edges
            pos_edge_index = self.data.edge_index

            # Negative edges
            neg_edge_index = negative_sampling(
                edge_index=pos_edge_index,
                num_nodes=self.data.num_nodes,
                num_neg_samples=num_edges
            )

            # Combine positive and negative edges
            edge_label_index = torch.cat([pos_edge_index, neg_edge_index], dim=1)
            edge_labels = torch.cat([
                torch.ones(pos_edge_index.size(1)),  # Positive labels
                torch.zeros(neg_edge_index.size(1))  # Negative labels
            ]).to(self.device)

            # Shuffle edges
            perm = torch.randperm(edge_label_index.size(1))
            edge_label_index = edge_label_index[:, perm]
            edge_labels = edge_labels[perm]

            self.optimizer.zero_grad()
            out = self.model(self.data.x, self.data.edge_index)

            src = edge_label_index[0]
            dst = edge_label_index[1]
            scores = (out[src] * out[dst]).sum(dim=1)

            loss = self.loss_fn(scores, edge_labels)
            loss.backward()
            self.optimizer.step()

            print(f"Epoch {epoch+1}/{self.epochs}, Loss: {loss.item():.4f}")

    def predict(self, user: Union[int, str], item: Union[int, str]) -> float:
        self.model.eval()
        with torch.no_grad():
            if user in self.user_id_map and item in self.item_id_map:
                user_idx = self.user_id_map[user]
                item_idx = self.item_id_map[item]
                out = self.model(self.data.x.to(self.device), self.data.edge_index.to(self.device))
                user_emb = out[user_idx]
                item_emb = out[item_idx]
                score = torch.sigmoid((user_emb * item_emb).sum()).item()
                return score
            else:
                print("User or Item not in training data.")
                return 0.0

    def evaluate(self) -> Dict[str, float]:
        self.model.eval()
        with torch.no_grad():
            # Positive edges
            pos_edge_index = self.data.edge_index

            # Negative edges
            num_edges = pos_edge_index.size(1) // 2
            neg_edge_index = negative_sampling(
                edge_index=pos_edge_index,
                num_nodes=self.data.num_nodes,
                num_neg_samples=num_edges
            )

            # Combine positive and negative edges
            edge_label_index = torch.cat([pos_edge_index, neg_edge_index], dim=1)
            edge_labels = torch.cat([
                torch.ones(pos_edge_index.size(1)),  # Positive labels
                torch.zeros(neg_edge_index.size(1))  # Negative labels
            ]).to(self.device)

            out = self.model(self.data.x, self.data.edge_index)

            src = edge_label_index[0]
            dst = edge_label_index[1]
            scores = (out[src] * out[dst]).sum(dim=1)
            loss = self.loss_fn(scores, edge_labels)

            # Compute accuracy
            preds = (torch.sigmoid(scores) > 0.5).float()
            accuracy = (preds == edge_labels).sum().item() / edge_labels.size(0)

        return {'Loss': loss.item(), 'Accuracy': accuracy}