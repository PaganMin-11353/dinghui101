import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# from .base_model import BaseModel
from utils.dataloader import DataLoader
from utils.metrics import get_top_k_items
from utils.metrics import calculate_rating_metrics, calculate_ranking_metrics
# from sklearn.model_selection import train_test_split

import pandas as pd
import numpy as np
from typing import Optional, Set, Tuple, Dict, Union
from typeguard import typechecked

import torch
import torch.nn as nn
from torch_geometric.data import Data
import logging

class TransformerLayer(nn.Module):
    def __init__(self, embedding_dim, num_heads=8):
        super(TransformerLayer, self).__init__()
        self.num_heads = num_heads
        self.attention = nn.MultiheadAttention(embedding_dim, num_heads)
    
    def forward(self, user_embedding_concat, item_embedding_concat):
        # Reshape user and item embeddings for the MultiheadAttention layer
        user_embedding_concat = user_embedding_concat.unsqueeze(1)  
        item_embedding_concat = item_embedding_concat.unsqueeze(1)  

        # Calculate attention scores
        updated_user_embedding, _ = self.attention(user_embedding_concat, item_embedding_concat, item_embedding_concat)

        # Remove batch dimension
        updated_user_embedding = updated_user_embedding.squeeze(1)

        return updated_user_embedding
    
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.fc1 = nn.Linear(self.input_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = nn.ReLU()(x)
        x = self.fc2(x)
        x = nn.ReLU()(x)

        return x
    

class Recommender(nn.Module):
    def __init__(self, embedding_dim, user_input_dim, item_input_dim, num_heads=8):
        super(Recommender, self).__init__()

        self.embedding_dim = embedding_dim
        self.user_input_dim = user_input_dim
        self.item_input_dim = item_input_dim
        self.num_heads = num_heads

        # Project the meta-path and content-based input embeddings into the same embedding as the pre-trained ones
        self.mlp_u = MLP(input_dim=self.user_input_dim, hidden_dim= 512, output_dim=self.embedding_dim)
        self.mlp_i = MLP(input_dim=self.item_input_dim, hidden_dim= 512, output_dim=self.embedding_dim)

        # To be applied on the concatenation of pre-trained and feature-aware embeddings (2 * embedding_dim)
        self.transformer_layer = TransformerLayer(embedding_dim = 2 * self.embedding_dim, num_heads=self.num_heads)

    def forward(self, user_input_emb, item_input_emb, user_pretrained_embedding, item_pretrained_embedding):
        # Project meta-path and content-based input embeddings to embedding_dim
        user_input_emb_2 = self.mlp_u(user_input_emb)
        item_input_emb_2 = self.mlp_i(item_input_emb)

        # Concat pretrained and input embedding -> dim = 2 * embedding_dim
        user_embedding = torch.cat((user_pretrained_embedding, user_input_emb_2), dim = -1)
        item_embedding = torch.cat((item_pretrained_embedding, item_input_emb_2), dim = -1)
        # user_embedding = user_input_emb_2
        # item_embedding = item_input_emb_2
        # Updated user embedding via transformation layer
        updated_user_embedding = self.transformer_layer(user_embedding, item_embedding)

        return updated_user_embedding, item_embedding

        
    

class RecommenderModel():
    def __init__(
        self,
        size: str,  # "100k" or "1m"
        train_set:pd.DataFrame,
        test_set:pd.DataFrame,
        num_layers: int = 3,
        embedding_dim: int = 64,
        learning_rate: float = 0.01,
        epochs: int = 10,
        batch_size: int = 1024,
        num_negatives: int =4,
        train_alpha: float = 1.0,
        paths = {"umam": "umam_embeddings.pt", 
                                                  "umdm":"umdm_embeddings.pt",
                                                  "umum":"umum_embeddings.pt",
                                                  "user_content": "user_content_based_embeddings.pt",
                                                  "item_content": "movie_genre_hot_embeddings.pt",
                                                  "user_pretrained": "pretrain_user_embeddings.pt",
                                                  "item_pretrained": "pretrain_item_embeddings.pt"},
        device: str = 'auto'  # 'auto', 'cuda', 'mps', or 'cpu'
    ) -> None:
        super().__init__()
        
        logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(name)s - [%(levelname)s]: %(message)s")
        self.logger = logging.getLogger("Recommender model")

        self.data_loader = DataLoader(size=size)
        self.train_set = train_set
        self.test_set = test_set

        if size not in self.data_loader.DATA_FORMATS:
            raise ValueError(f"Invalid size: {size}. Choose from {list(self.data_loader.DATA_FORMATS.keys())}.")

        self.size = size
        self.num_layers = num_layers
        self.embedding_dim = embedding_dim
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.num_negatives = num_negatives
        self.train_alpha = train_alpha
        self.paths = paths
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

        self.test_pre = None

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
        # print(ratings.head())
        return ratings

    def _load_items(self) -> pd.DataFrame:
        items = self.data_loader.load_items(process_title=True, process_year=True, process_genres=True, genres_as_binary=True)
        return items

    def _load_user_features(self) -> pd.DataFrame:
        user_features = self.data_loader.load_user_features(convert_age_to_range=True,convert_occupation_to_code=False)
        return user_features
    
    def load_input_embeddings(self, paths = {"umam": "umam_embeddings.pt", 
                                                  "umdm":"umdm_embeddings.pt",
                                                  "umum":"umum_embeddings.pt",
                                                  "user_content": "user_content_based_embeddings.pt",
                                                  "item_content": "movie_genre_hot_embeddings.pt",
                                                  "user_pretrained": "user_pretrained_embedding.pt",
                                                  "item_pretrained": "item_pretrained_embedding.pt"}):
        
        ### Content-based and meta-path features
        # Load the meta-path and content-based features of users
        umam_embeddings = torch.load(paths['umam'])
        umdm_embeddings = torch.load(paths['umdm'])
        umum_embeddings = torch.load(paths['umum'])
        user_content_embedding = torch.load(paths['user_content'])

        user_input_emb = {}
        for user_id in self.user2idx:
            user_input_emb[user_id] = torch.cat((umam_embeddings[user_id], umdm_embeddings[user_id], umum_embeddings[user_id], user_content_embedding[user_id]), dim = 1)
        
        del umam_embeddings, umdm_embeddings, umum_embeddings, user_content_embedding # save memory

        # Load the content-based features of items
        item_input_emb = torch.load(paths["item_content"])

        # Transform embeddings from dict to tensor
        self.user_input_emb = torch.cat(tuple([user_input_emb[i] for i in user_input_emb]), dim = 0).float()
        self.item_input_emb= torch.cat(tuple([item_input_emb[i] for i in item_input_emb]), dim = 0).float()

        ### Pretrained embedding
        self.user_pretrained_embedding = torch.load(paths["user_pretrained"])
        self.item_pretrained_embedding = torch.load(paths["item_pretrained"])


    def prepare_training_data(self) -> None:
        self.logger.info("Preparing data...")
        self.ratings = pd.concat([self.train_set, self.test_set], axis=0, ignore_index=True)

        user_list = self.ratings['user'].unique().tolist()
        item_list = self.ratings['item'].unique().tolist()
    
        self.user2idx = {user: idx for idx, user in enumerate(user_list)}
        self.idx2user = {idx: user for user, idx in self.user2idx.items()}

        self.item2idx = {item: idx for idx, item in enumerate(item_list)}
        self.idx2item = {idx: item for item, idx in self.item2idx.items()}

        self.num_users = len(user_list)
        self.num_items = len(item_list)

        self.ratings['user_idx'] = self.ratings['user'].map(self.user2idx)
        self.ratings['item_idx'] = self.ratings['item'].map(self.item2idx)

        train_df = self.ratings[:len(self.train_set)]
        test_df = self.ratings[len(self.train_set):]

        self.test_pre = test_df.copy()
        global_neg_set = set(zip(self.ratings['user_idx'], self.ratings['item_idx']))

        train_neg_df = self._generate_negative_samples(train_df, global_neg_set=global_neg_set)
        test_neg_df = self._generate_negative_samples(test_df, global_neg_set=global_neg_set)

        # merge pos neg, label: pos 1 neg 0
        train_full_df = pd.concat([
            train_df[['user_idx', 'item_idx', 'rating']].assign(label=1),
            train_neg_df[['user_idx', 'item_idx', 'rating', 'label']]
        ], ignore_index=True)

        test_full_df = pd.concat([
            test_df[['user_idx', 'item_idx', 'rating']].assign(label=1),
            test_neg_df[['user_idx', 'item_idx', 'rating', 'label']]
        ], ignore_index=True)

        self.train_df = train_full_df
        self.test_df = test_full_df

        self.load_input_embeddings()
        self._build_graph()
        self._init_model()
    
    def _generate_negative_samples(self, df: pd.DataFrame, global_neg_set: Optional[Set[tuple]] = None) -> pd.DataFrame:
        self.logger.info("Generating negative samples...") 
    
        if global_neg_set is None:
            global_neg_set = set(zip(df['user_idx'], df['item_idx']))
        
        users = df['user_idx'].unique()
        all_items = set(range(self.num_items))
        
        user_to_interacted_items = df.groupby('user_idx')['item_idx'].apply(set).to_dict()
        
        user_positive_counts = df['user_idx'].value_counts().to_dict()
        user_neg_counts = {user: count * self.num_negatives for user, count in user_positive_counts.items()}
        
        neg_samples = []
        
        for user, neg_count in user_neg_counts.items():
            interacted = user_to_interacted_items.get(user, set())
            available_items = list(all_items - interacted)
            
            if len(available_items) < neg_count:
                sampled_items = np.random.choice(available_items, size=neg_count, replace=True)
            else:
                sampled_items = np.random.choice(available_items, size=neg_count, replace=False)
            
            neg_samples.extend([{'user_idx': user, 'item_idx': int(item), 'rating': 0, 'label': 0} for item in sampled_items])
        
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

        src = np.concatenate([user_indices, item_indices])
        dst = np.concatenate([item_indices, user_indices])
        
        edge_index_np = np.stack([src, dst], axis=0).astype(np.int64)
        edge_index = torch.tensor(edge_index_np, dtype=torch.long)

        data = Data(edge_index=edge_index, num_nodes=self.num_users + self.num_items)

        data = data.to(self.device)

        # adj_norm = self.normalize_adj(data.edge_index, data.num_nodes)
        row, col = data.edge_index
        deg = torch.bincount(row, minlength=data.num_nodes).float()
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0.0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        edge_weight = norm

        adj_norm = torch.sparse_coo_tensor(data.edge_index, edge_weight, (data.num_nodes, data.num_nodes))
        self.adj_norm = adj_norm.coalesce().to(self.device)

    def _init_model(self):
        self.logger.info("initilizing model...")
        self.model = Recommender(
            embedding_dim=self.embedding_dim,
            user_input_dim=self.user_input_emb.shape[1],
            item_input_dim=self.item_input_emb.shape[1],
            num_heads = 8
        ).to(self.device)

        # adam + l2
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-5)

    def train(self):
        self.model.train()

        pos_df = self.train_df[self.train_df['label'] == 1]
        neg_df = self.train_df[self.train_df['label'] == 0]

        pos_user_ids = pos_df['user_idx'].values
        pos_item_ids = pos_df['item_idx'].values
        pos_ratings = pos_df['rating'].values
        
        neg_user_ids = neg_df['user_idx'].values
        neg_item_ids = neg_df['item_idx'].values

        pos_len = len(pos_user_ids)
        required_neg_samples = pos_len * self.num_negatives

        # 检查负样本数量是否足够
        actual_neg_samples = len(neg_user_ids)
        if actual_neg_samples < required_neg_samples:
            raise ValueError(f"Not enough negative samples: required {required_neg_samples}, but got {actual_neg_samples}")

        # 截取所需数量的负样本
        neg_user_ids = neg_user_ids[:required_neg_samples]
        neg_item_ids = neg_item_ids[:required_neg_samples]

        # 重复正样本以匹配负样本数量
        pos_user_ids_bpr = np.repeat(pos_user_ids, self.num_negatives)
        pos_item_ids_bpr = np.repeat(pos_item_ids, self.num_negatives)
        
        pos_user_ids_bpr = torch.from_numpy(pos_user_ids_bpr).long().to(self.device)
        pos_item_ids_bpr = torch.from_numpy(pos_item_ids_bpr).long().to(self.device)
        neg_user_ids_bpr = torch.from_numpy(neg_user_ids).long().to(self.device)
        neg_item_ids_bpr = torch.from_numpy(neg_item_ids).long().to(self.device)
        
        # For rating loss, use positive samples only
        pos_user_ids_reg = torch.from_numpy(pos_user_ids).long().to(self.device)
        pos_item_ids_reg = torch.from_numpy(pos_item_ids).long().to(self.device)
        pos_ratings_reg = torch.from_numpy(pos_ratings).float().to(self.device)

        for epoch in range(1, self.epochs + 1):
            self.optimizer.zero_grad()

            # forward
            user_emb, item_emb = self.model(self.user_input_emb, self.item_input_emb, self.user_pretrained_embedding, self.item_pretrained_embedding)
        
            # BPR Loss
            u_pos = user_emb[pos_user_ids_bpr]    # (N, embedding_dim)
            i_pos = item_emb[pos_item_ids_bpr]    # (N, embedding_dim)
            u_neg = user_emb[neg_user_ids_bpr]    # (N, embedding_dim)
            i_neg = item_emb[neg_item_ids_bpr]    # (N, embedding_dim)
            
            pos_scores = torch.sum(u_pos * i_pos, dim=1)  # (N,)
            neg_scores = torch.sum(u_neg * i_neg, dim=1)  # (N,)
            
            # BPR loss
            bpr_loss = -torch.mean(torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-10))
            
            # MSE Loss for rating prediction
            pred_ratings = torch.sum(user_emb[pos_user_ids_reg] * item_emb[pos_item_ids_reg], dim=1)  # (N,)
            mse_loss = nn.functional.mse_loss(pred_ratings, pos_ratings_reg)
            
            # Combine losses
            loss = bpr_loss #+ self.train_alpha * mse_loss

            loss.backward()
            self.optimizer.step()

            self.logger.info(f"Epoch {epoch}/{self.epochs}, BPR Loss: {loss.item():.4f}")
            if epoch % 10:
                print(f"Epoch {epoch}/{self.epochs}, BPR Loss: {loss.item():.4f}")

            # see evaluation result of this epoch
            if epoch % 10 == 0:
                predictions = self.predict()
                ratings = calculate_rating_metrics(self.test_pre, predictions)
                top_k_scores = self.recommend_k()
                rankings = calculate_ranking_metrics(self.test_pre, top_k_scores, 10)
                print(f"Epoch {epoch}/{self.epochs}, Recall@K: {rankings['Recall@k']}")
    
    def _predict_scores(self) -> torch.Tensor:
        self.model.eval()
        with torch.no_grad():
            user_emb, item_emb = self.model(self.user_input_emb, self.item_input_emb, self.user_pretrained_embedding, self.item_pretrained_embedding)
            scores = torch.matmul(user_emb, item_emb.t())
        return scores

    def predict(self) -> pd.DataFrame:
        scores = self._predict_scores()
        
        test_users = self.test_df['user_idx'].values
        test_items = self.test_df['item_idx'].values
        test_scores = scores[test_users, test_items].cpu().numpy()

        predictions = self.test_df.copy()
        predictions['prediction'] = test_scores

        predictions['user'] = predictions['user_idx'].map(self.idx2user)
        predictions['item'] = predictions['item_idx'].map(self.idx2item)

        predictions['rating'] = predictions.apply(
            lambda row: row['rating'] if row['label'] == 1 else 0, axis=1)

        return predictions[['user', 'item', 'prediction', 'rating']]


    def recommend_k(self, k:int=10) -> pd.DataFrame:
        scores = self._predict_scores().cpu().numpy()

        user_indices = np.repeat(np.arange(self.num_users), self.num_items)
        item_indices = np.tile(np.arange(self.num_items), self.num_users)
        all_scores = scores.flatten()

        all_predictions = pd.DataFrame({
            'user_idx': user_indices,
            'item_idx': item_indices,
            'prediction': all_scores
        })

        interacted = self.train_df[self.train_df['label'] == 1][['user_idx', 'item_idx']].drop_duplicates()

        all_predictions = all_predictions.merge(
            interacted.assign(interacted=True),
            on=['user_idx', 'item_idx'],
            how='left'
        )
        all_predictions = all_predictions[all_predictions['interacted'].isna()].drop(columns=['interacted'])

        top_k_items = get_top_k_items(
            dataframe=all_predictions.copy(),
            col_user="user_idx",
            col_rating="prediction",
            k=k
        )

        top_k_items['user'] = top_k_items['user_idx'].map(self.idx2user)
        top_k_items['item'] = top_k_items['item_idx'].map(self.idx2item)

        return top_k_items[['user', 'item', 'prediction']]