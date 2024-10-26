from .base_model import BaseModel
from utils.dataloader import load_data_df, load_item_df, load_user_features, DATA_FORMAT

import os
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Union
from typeguard import typechecked

import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.utils import negative_sampling


class LightGCN(nn.Module):
    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int,
        num_layers: int,
        edge_index: torch.Tensor,
        device: torch.device,
    ):
        super(LightGCN, self).__init__()

        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.edge_index = edge_index
        self.device = device

        # Initialize embeddings
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        """
        Initialize the embeddings with a normal distribution.
        """
        nn.init.normal_(self.user_embedding.weight, std=0.1)
        nn.init.normal_(self.item_embedding.weight, std=0.1)

    def forward(self) -> Tuple[torch.Tensor, torch.Tensor]:
        all_embeddings = torch.cat(
            [self.user_embedding.weight, self.item_embedding.weight], dim=0
        )
        embeddings = [all_embeddings]

        for _ in range(self.num_layers):
            all_embeddings = self.propagate(all_embeddings)
            embeddings.append(all_embeddings)

        embeddings = torch.stack(embeddings, dim=1)
        final_embeddings = torch.mean(embeddings, dim=1)  # Mean aggregation

        user_embeddings = final_embeddings[: self.num_users]
        item_embeddings = final_embeddings[self.num_users :]

        return user_embeddings, item_embeddings

    def propagate(self, embeddings: torch.Tensor) -> torch.Tensor:
        edge_index = self.edge_index
        row, col = edge_index

        deg = torch.bincount(row, minlength=embeddings.size(0))
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        out = torch.zeros_like(embeddings)
        out.index_add_(0, row, embeddings[col] * norm.unsqueeze(1))
        return out

    def bpr_loss(
        self, users: torch.Tensor, pos_items: torch.Tensor, neg_items: torch.Tensor
    ) -> torch.Tensor:
        user_emb, item_emb = self.forward()

        u_emb = user_emb[users]
        pos_emb = item_emb[pos_items]
        neg_emb = item_emb[neg_items]

        pos_scores = torch.sum(u_emb * pos_emb, dim=1)
        neg_scores = torch.sum(u_emb * neg_emb, dim=1)

        loss = -torch.mean(torch.log(torch.sigmoid(pos_scores - neg_scores)))
        return loss

    def get_embeddings(self) -> Tuple[torch.Tensor, torch.Tensor]:
        user_embeddings, item_embeddings = self.forward()
        return user_embeddings, item_embeddings


@typechecked
class LightGCNModel(BaseModel):
    def __init__(
        self,
        size: str,  # "100k" or "1m"
        num_layers: int = 3,
        embedding_dim: int = 64,
        learning_rate: float = 0.01,
        epochs: int = 10,
        batch_size: int = 1024,
        device: str = 'auto'  # 'auto', 'cuda', 'mps', or 'cpu'
    ) -> None:
        super().__init__()

        if size not in DATA_FORMAT:
            raise ValueError(f"Invalid size: {size}. Choose from {list(DATA_FORMAT.keys())}.")

        self.size = size
        self.num_layers = num_layers
        self.embedding_dim = embedding_dim
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

        # Initialize the LightGCN model
        self.model = LightGCN(
            num_users=self.num_users,
            num_items=self.num_items,
            embedding_dim=self.embedding_dim,
            num_layers=self.num_layers
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

    def _prepare_graph_data(self) -> None:
        # Create mappings from user and item IDs to indices
        user_ids = self.ratings['UserId'].unique()
        item_ids = self.ratings['MovieId'].unique()

        self.user_id_map = {id: idx for idx, id in enumerate(user_ids)}
        self.item_id_map = {id: idx for idx, id in enumerate(item_ids)}

        self.num_users = len(user_ids)
        self.num_items = len(item_ids)

        # Edges (user-item interactions)
        user_indices = self.ratings['UserId'].map(self.user_id_map).values
        item_indices = self.ratings['MovieId'].map(self.item_id_map).values
        
        # self.edge_index = torch.tensor(
        #     [np.concatenate([user_indices, item_indices]),
        #      np.concatenate([item_indices + self.num_users, user_indices])]
        # ).long()

        # self.edge_index = self.edge_index.to(self.device)

        # Edge index for LightGCN (bipartite graph)
        # Users are [0, num_users - 1], Items are [num_users, num_users + num_items -1]
        # To create undirected edges, add reverse edges
        # So, for each user-item pair, add both (user, item) and (item, user)
        all_users = user_indices
        all_items = item_indices + self.num_users  # Shift item indices
        edge_index = torch.tensor([np.concatenate([all_users, all_items]), 
                                   np.concatenate([all_items, all_users]),],
                                  dtype=torch.long,)
        self.edge_index = edge_index.to(self.device)

    def train(self) -> None:
        self.model.train()

        # Generate all positive user-item pairs
        user_item_pairs = torch.tensor([
            self.ratings['UserId'].map(self.user_id_map).values,
            self.ratings['MovieId'].map(self.item_id_map).values,],
            dtype=torch.long,
            ).to(self.device)

        for epoch in range(self.epochs):
            # Sample negative items for each user
            users = user_item_pairs[0]
            pos_items = user_item_pairs[1]
            neg_items = torch.randint(0, self.num_items, (len(users),), device=self.device)

            self.optimizer.zero_grad()

            # Forward pass
            loss = self.model.bpr_loss(users, pos_items, neg_items)
            loss.backward()
            self.optimizer.step()

            print(f"Epoch {epoch+1}/{self.epochs}, Loss: {loss.item():.4f}")

    def predict(self, user: Union[int, str], item: Union[int, str]) -> float:
        self.model.eval()
        with torch.no_grad():
            if user in self.user_id_map and item in self.item_id_map:
                user_idx = self.user_id_map[user]
                item_idx = self.item_id_map[item]
                user_emb, item_emb = self.model.get_embeddings()
                u_emb = user_emb[user_idx]
                i_emb = item_emb[item_idx]
                score = torch.dot(u_emb, i_emb)
                return score.item()
            else:
                print("User or Item not in training data.")
                return 0.0

    def evaluate(self) -> Dict[str, float]:
        self.model.eval()
        with torch.no_grad():
            # Generate user-item pairs for evaluation
            user_item_pairs = torch.tensor(
                [
                    self.ratings["UserId"].map(self.user_id_map).values,
                    self.ratings["MovieId"].map(self.item_id_map).values,
                ],
                dtype=torch.long,
            ).to(self.device)

            users = user_item_pairs[0]
            pos_items = user_item_pairs[1]
            neg_items = torch.randint(
                0, self.num_items, (len(users),), device=self.device
            )

            # Compute BPR loss
            loss = self.model.bpr_loss(users, pos_items, neg_items)

            # Compute accuracy
            user_emb, item_emb = self.model.get_embeddings()
            pos_scores = (user_emb[users] * item_emb[pos_items]).sum(dim=1)
            neg_scores = (user_emb[users] * item_emb[neg_items]).sum(dim=1)

            correct = (pos_scores > neg_scores).float().sum()
            accuracy = correct / len(users)

        return {"Loss": loss.item(), "Accuracy": accuracy.item()}