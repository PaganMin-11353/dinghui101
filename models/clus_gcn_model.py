import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# from .base_model import BaseModel
from utils.dataloader import DataLoader
# from sklearn.model_selection import train_test_split

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Union
from typeguard import typechecked

import torch
import torch.nn as nn
from torch_geometric.data import Data
import logging

class ClusGCN(nn.Module):
    def __init__(self, num_users: int, num_items: int, embedding_dim: int, num_layers: int, embedding_path = os.path.join(sys.path[0], "embeddings")):
        super(ClusGCN, self).__init__()
        print(sys.path)
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers

        # Initialize embeddings
        self.user_embedding = nn.Embedding(num_users, self.embedding_dim)
        self.item_embedding = nn.Embedding(num_items, self.embedding_dim)

        # Pre-generated one-hot/multi-hot encoding for movie genre, user age, user occupation
        self.genre_hot_embedding = torch.load(os.path.join(embedding_path, "genre_hot_embedding.pt"))
        self.age_hot_embedding = torch.load(os.path.join(embedding_path, "user_age_hot_embedding.pt"))
        self.occupation_hot_embedding = torch.load(os.path.join(embedding_path, "user_occupation_hot_embedding.pt"))

        # Embedding layer for movie genre, user age, user occupation
        self.genre_cat_layer = torch.nn.Linear(19, self.embedding_dim)
        self.age_cat_layer = torch.nn.Linear(7, self.embedding_dim)
        self.occupation_cat_layer = torch.nn.Linear(21, self.embedding_dim)

        # Embedding layer for user

        self._init_embeddings()

    def _init_embeddings(self) -> None:
        nn.init.normal_(self.user_embedding.weight, std=0.1)
        nn.init.normal_(self.item_embedding.weight, std=0.1)
        print(f'before:{self.user_embedding.weight}')
        print(f'add : {self.age_hot_embedding}')
        self.user_embedding.weight += self.age_hot_embedding
        print(f'after: {self.user_embedding.weight}')
        # nn.init.xavier_uniform_(self.user_embedding.weight)
        # nn.init.xavier_uniform_(self.item_embedding.weight)

    def forward(self, adj_norm) -> Tuple[torch.Tensor, torch.Tensor]:
        all_embeddings = torch.cat(
            [self.user_embedding.weight, 
             self.item_embedding.weight], dim=0
        )
        embeddings_list = [all_embeddings]

        for _ in range(self.num_layers):
            all_embeddings = torch.sparse.mm(adj_norm, all_embeddings)
            embeddings_list.append(all_embeddings)

        all_embeddings = torch.stack(embeddings_list, dim=1)
        final_embeddings = torch.mean(all_embeddings, dim=1)  # Mean aggregation

        user_embeddings = final_embeddings[: self.num_users]
        item_embeddings = final_embeddings[self.num_users :]

        return user_embeddings, item_embeddings

    # def bpr_loss(self, users: torch.Tensor, pos_items: torch.Tensor, neg_items: torch.Tensor) -> torch.Tensor:
    #     user_emb, item_emb = self.forward()

    #     u_emb = user_emb[users]
    #     pos_emb = item_emb[pos_items]
    #     neg_emb = item_emb[neg_items]

    #     pos_scores = torch.sum(u_emb * pos_emb, dim=1)
    #     neg_scores = torch.sum(u_emb * neg_emb, dim=1)

    #     loss = -torch.mean(torch.log(torch.sigmoid(pos_scores - neg_scores)))
    #     return loss

    # def get_embeddings(self) -> Tuple[torch.Tensor, torch.Tensor]:
    #     user_embeddings, item_embeddings = self.forward()
    #     return user_embeddings, item_embeddings


@typechecked
class ClusGCNModel():
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
        
        logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(name)s - [%(levelname)s]: %(message)s")
        self.logger = logging.getLogger("LGCN model")

        self.data_loader = DataLoader(size=size)

        if size not in self.data_loader.DATA_FORMATS:
            raise ValueError(f"Invalid size: {size}. Choose from {list(self.data_loader.DATA_FORMATS.keys())}.")

        self.size = size
        self.num_layers = num_layers
        self.embedding_dim = embedding_dim
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = self._get_device(device)

        self.user2idx = {}
        self.item2idx = {}
        self.idx2user = {}
        self.idx2item = {}
        self.num_users = 0
        self.num_items = 0

        self.train_df = None
        self.test_df = None
        self.graph = None
        self.adj_norm = None

        self.model = None
        self.optimizer = None
        self.criterion = nn.BCEWithLogitsLoss()


    def _get_device(self, device_str: str) -> torch.device:
        if device_str == 'auto':
            if torch.cuda.is_available():
                self.logger.info("Using CUDA (Nvidia GPU)")
                return torch.device('cuda')
            elif torch.backends.mps.is_available():
                self.logger.info("Using MPS (Apple Silicon GPU)")
                return torch.device('mps')
            else:
                self.logger.info("Using CPU")
                return torch.device('cpu')
        elif device_str == 'cuda':
            if torch.cuda.is_available():
                self.logger.info("Using CUDA (Nvidia GPU)")
                return torch.device('cuda')
            else:
                raise ValueError("CUDA is not available")          
        elif device_str == 'mps':
            if torch.backends.mps.is_available():
                self.logger.info("Using MPS (Apple Silicon GPU)")
                return torch.device('mps')
            else:
                raise ValueError("MPS is not available")
        elif device_str == 'cpu':
            self.logger.info("Using CPU")
            return torch.device('cpu')
        else:
            raise ValueError("Invalid device option: 'auto', 'cuda', 'mps', 'cpu'.")

    def _load_ratings(self) -> pd.DataFrame:
        ratings = self.data_loader.load_ratings()
        self.logger.info("Data Loaded.")
        print(ratings.head())
        return ratings

    def _load_items(self) -> pd.DataFrame:
        items = self.data_loader.load_items(process_title=True, process_year=True, process_genres=True, genres_as_binary=True)
        return items

    def _load_user_features(self) -> pd.DataFrame:
        user_features = self.data_loader.load_user_features(convert_age_to_range=True,convert_occupation_to_code=False)
        return user_features

    def prepare_training_data(self) -> None:
        self.logger.info("Preparing data...")
        self.ratings = self._load_ratings()
        self.items = self._load_items()
        self.user_features = self._load_user_features()
        
        user_list = self.ratings['user'].unique().tolist()
        item_list = self.ratings['item'].unique().tolist()

        self.user2idx = {user: idx for idx, user in enumerate(user_list)}
        self.item2idx = {item: idx for idx, item in enumerate(item_list)}
        self.idx2user = {idx: user for user, idx in self.user2idx.items()}
        self.idx2item = {idx: item for item, idx in self.item2idx.items()}

        self.num_users = len(user_list)
        self.num_items = len(item_list)

        self.ratings['user_idx'] = self.ratings['user'].map(self.user2idx)
        self.ratings['item_idx'] = self.ratings['item'].map(self.item2idx)

        # print("\nuser item map：")
        # print(self.user2idx)
        # print(self.item2idx)

        # train_df, test_df = train_test_split(
        #     self.ratings,
        #     test_size=0.3,
        #     random_state=42,
        #     stratify=self.ratings['user_idx']
        # )

        # print("\Train set:")
        # print(train_df.head())
        # print("\Test set:")
        # print(test_df.head())

        # train test split, keep 1 useritem for every user
        train_list = []
        test_list = []

        grouped = self.ratings.groupby('user_idx')
        for user, group in grouped:
            if len(group) < 2:
                train_list.append(group)
            else:
                test_sample = group.sample(n=1, random_state=42)
                train_sample = group.drop(test_sample.index)
                train_list.append(train_sample)
                test_list.append(test_sample)

        train_df = pd.concat(train_list).reset_index(drop=True)
        test_df = pd.concat(test_list).reset_index(drop=True)

        global_neg_set_train = set(zip(train_df['user_idx'], train_df['item_idx']))
        train_neg_df = self._generate_negative_samples(train_df, num_negatives=4, global_neg_set=global_neg_set_train)

        global_neg_set_test = global_neg_set_train.union(set(zip(test_df['user_idx'], test_df['item_idx'])))
        test_neg_df = self._generate_negative_samples(test_df, num_negatives=4, global_neg_set=global_neg_set_test)

        
        # merge pos neg, label: pos 1 neg 0
        train_full_df = pd.concat([
            train_df[['user_idx', 'item_idx']].assign(label=1),
            train_neg_df[['user_idx', 'item_idx', 'label']]
        ], ignore_index=True)

        test_full_df = pd.concat([
            test_df[['user_idx', 'item_idx']].assign(label=1),
            test_neg_df[['user_idx', 'item_idx', 'label']]
        ], ignore_index=True)

        self.train_df = train_full_df
        self.test_df = test_full_df

        self._build_graph()
        self._init_model()
    
    def _generate_negative_samples(self, df, num_negatives=1, global_neg_set=None):
        self.logger.info("Generating negative samples...")
    
        if global_neg_set is None:
            global_neg_set = set(zip(df['user_idx'], df['item_idx']))
        
        users = df['user_idx'].unique()
        all_items = set(range(self.num_items))
        
        user_to_interacted_items = df.groupby('user_idx')['item_idx'].apply(set).to_dict()
        
        user_positive_counts = df['user_idx'].value_counts().to_dict()
        user_neg_counts = {user: count * num_negatives for user, count in user_positive_counts.items()}
        
        neg_samples = []
        
        for user, neg_count in user_neg_counts.items():
            interacted = user_to_interacted_items.get(user, set())
            available_items = list(all_items - interacted)
            
            if len(available_items) < neg_count:
                sampled_items = np.random.choice(available_items, size=neg_count, replace=True)
            else:
                sampled_items = np.random.choice(available_items, size=neg_count, replace=False)
            
            neg_samples.extend([{'user_idx': user, 'item_idx': int(item), 'label': 0} for item in sampled_items])
        
        neg_df = pd.DataFrame(neg_samples)

        # 确认每个用户生成了正确数量的负样本
        user_neg_counts_generated = neg_df['user_idx'].value_counts().to_dict()
        for user, expected_count in user_neg_counts.items():
            actual_count = user_neg_counts_generated.get(user, 0)
            if actual_count != expected_count:
                self.logger.warning(f"Warning: User {user} expected {expected_count} negatives, but got {actual_count}")
        
        return neg_df

    def _build_graph(self):
        self.logger.info("building graph...")
        train_positive = self.train_df[self.train_df['label'] == 1]

        user_indices = train_positive['user_idx'].values
        item_indices = train_positive['item_idx'].values + self.num_users 

        # print(f'user_indices: {user_indices}, len(user_indices) = {len(user_indices)}')
        # print(f'item_indices: {item_indices}, len(item_indices) = {len(item_indices)}')

        src = np.concatenate([user_indices, item_indices])
        dst = np.concatenate([item_indices, user_indices])
        # print(f'src:{src}')
        # print(f'dst:{dst}')
        edge_index_np = np.stack([src, dst], axis=0).astype(np.int64)
        edge_index = torch.tensor(edge_index_np, dtype=torch.long)
        #print(f'edge_index: {edge_index}')
        data = Data(edge_index=edge_index, num_nodes=self.num_users + self.num_items)
        data = data.to(self.device)
        #print(f'data: {data}')
        # adj_norm = self.normalize_adj(data.edge_index, data.num_nodes)
        row, col = data.edge_index
        deg = torch.bincount(row, minlength=data.num_nodes).float()
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0.0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        edge_weight = norm

        adj_norm = torch.sparse_coo_tensor(data.edge_index, edge_weight, (data.num_nodes, data.num_nodes))
        self.adj_norm = adj_norm.coalesce().to(self.device)
        print(f'adj_norm: {self.adj_norm}')

    def _init_model(self):
        self.logger.info("initilizing model...")
        self.model = ClusGCN(
            num_users=self.num_users,
            num_items=self.num_items,
            embedding_dim=self.embedding_dim,
            num_layers=self.num_layers
        ).to(self.device)

        # adam + l2
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-5)

    def train(self):
        self.model.train()

        print("starts training")

        pos_df = self.train_df[self.train_df['label'] == 1]
        neg_df = self.train_df[self.train_df['label'] == 0]

        pos_user_ids = pos_df['user_idx'].values
        pos_item_ids = pos_df['item_idx'].values
        neg_user_ids = neg_df['user_idx'].values
        neg_item_ids = neg_df['item_idx'].values

        num_negatives = 4
        pos_len = len(pos_user_ids)
        required_neg_samples = pos_len * num_negatives

        # 检查负样本数量是否足够
        actual_neg_samples = len(neg_user_ids)
        if actual_neg_samples < required_neg_samples:
            raise ValueError(f"Not enough negative samples: required {required_neg_samples}, but got {actual_neg_samples}")

        # 截取所需数量的负样本
        neg_user_ids = neg_user_ids[:required_neg_samples]
        neg_item_ids = neg_item_ids[:required_neg_samples]

        # 重复正样本以匹配负样本数量
        pos_user_ids = np.repeat(pos_user_ids, num_negatives)
        pos_item_ids = np.repeat(pos_item_ids, num_negatives)

        pos_user_ids = torch.from_numpy(pos_user_ids).long().to(self.device)
        pos_item_ids = torch.from_numpy(pos_item_ids).long().to(self.device)
        neg_user_ids = torch.from_numpy(neg_user_ids).long().to(self.device)
        neg_item_ids = torch.from_numpy(neg_item_ids).long().to(self.device)

        print("starts loop")
        for epoch in range(1, self.epochs + 1):
            self.optimizer.zero_grad()

            # forward prop
            user_emb, item_emb = self.model(self.adj_norm)

            u_pos = user_emb[pos_user_ids]    # (N, embedding_dim)
            i_pos = item_emb[pos_item_ids]    # (N, embedding_dim)
            u_neg = user_emb[neg_user_ids]    # (N, embedding_dim)
            i_neg = item_emb[neg_item_ids]    # (N, embedding_dim)

            pos_scores = torch.sum(u_pos * i_pos, dim=1)  # (N,)
            neg_scores = torch.sum(u_neg * i_neg, dim=1)  # (N,)

            # BPR loss：-log(sigmoid(pos - neg))
            loss = -torch.mean(torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-10))

            loss.backward()
            self.optimizer.step()

            self.logger.info(f"Epoch {epoch}/{self.epochs}, Loss: {loss.item():.4f}")
            print(f"Epoch {epoch}/{self.epochs}, Loss: {loss.item():.4f}")
        

    def predict(self) -> pd.DataFrame:
        self.model.eval()
        with torch.no_grad():
            user_emb, item_emb = self.model(self.adj_norm)
            scores = torch.matmul(user_emb, item_emb.t()).cpu().numpy()
        
        test_users = self.test_df['user_idx'].values
        test_items = self.test_df['item_idx'].values
        test_scores = scores[test_users, test_items]

        predictions = self.test_df.copy()
        predictions['score'] = test_scores

        return predictions[['user_idx', 'item_idx', 'score']]


    def recommend_k(self) -> Dict[str, float]:
        pass


def main():

    model = ClusGCN(
        size="100k",
        embedding_dim=64,
        num_layers=5,
        learning_rate=0.01,
        epochs=50,
        device='cpu'
    )

    model.prepare_training_data()
    model.train()

    predictions = model.predict()
    print(predictions.head())

    # # 为指定用户生成 Top-K 推荐
    # user_id = '196'  # 示例用户 ID，确保该 ID 存在于数据集中
    # top_k = 10
    # recommendations = model.recommend_k(user_id, k=top_k)
    # print(f"\n为用户 '{user_id}' 推荐的 Top-{top_k} 物品：{recommendations}")

    # # 评估模型
    # metrics = model.evaluate_metrics(k=top_k)
    # print("\n评估指标：")
    # for metric, value in metrics.items():
    #     print(f"{metric}: {value:.4f}")


if __name__ == "__main__":
    main()