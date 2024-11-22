import ast
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class GeneralGNN(nn.Module):
    def __init__(self, name, settings, init_user_embedding_path, init_item_embedding_path):
        super().__init__()
        self.name = name

        # Hyperparameters and settings
        assert hasattr(settings, "embedding_size"), "settings must have 'embedding_size'"
        assert hasattr(settings, "learning_rate"), "settings must have 'learning_rate'"
        assert hasattr(settings, "num_items"), "settings must have 'num_items'"
        assert hasattr(settings, "num_users"), "settings must have 'num_users'"
        self.embedding_size = settings.embedding_size
        self.learning_rate = settings.learning_rate
        self.num_items = settings.num_items
        self.num_users = settings.num_users

        # Transformer encoder structure
        self.dropout_rate = settings.dropout_rate
        self.num_heads = settings.num_heads
        self.d_ff = settings.d_ff
        self.num_blocks = settings.num_blocks

        # Embedding matrices
        self.user_embeddings = nn.Embedding(self.num_users + 1, self.embedding_size)
        self.item_embeddings = nn.Embedding(self.num_items + 1, self.embedding_size)

        # Initialize padding embedding (last row for padding)
        with torch.no_grad():
            self.user_embeddings.weight[-1].fill_(0)
            self.item_embeddings.weight[-1].fill_(0)

        # Encoder for multi-head attention
        self.encoder = Encoder(
            embedding_size=self.embedding_size,
            num_blocks=self.num_blocks,
            num_heads=self.num_heads,
            d_ff=self.d_ff,
            dropout_rate=self.dropout_rate
        )

        # Trainable weight matrices for second and third-order aggregation
        self.first_order_refine_weight = nn.Parameter(self.glorot([self.embedding_size, self.embedding_size]))
        self.second_order_refine_weight = nn.Parameter(self.glorot([2 * self.embedding_size, self.embedding_size]))
        self.third_order_refine_weight = nn.Parameter(self.glorot([3 * self.embedding_size, self.embedding_size]))
        self.first_order_sage_weight = nn.Parameter(self.glorot([self.embedding_size, self.embedding_size]))
        self.second_order_sage_weight = nn.Parameter(self.glorot([self.embedding_size, self.embedding_size]))
        self.third_order_sage_weight = nn.Parameter(self.glorot([self.embedding_size, self.embedding_size]))

        # Parameters for agent networks
        self.second_order_agent = self.create_agent_network(self.embedding_size)
        self.third_order_agent = self.create_agent_network(self.embedding_size)

        # Optimizer (you may need separate optimizers for different components if required)
        self.optimizer = torch.optim.Adagrad(self.parameters(), lr=self.learning_rate)

        # index and id mappings
        init_user_embedding_df = pd.read_csv(init_user_embedding_path)
        init_item_embedding_df = pd.read_csv(init_item_embedding_path)

        user_embeddings = torch.tensor(init_user_embedding_df['embedding'].apply(ast.literal_eval).tolist(), dtype=torch.float32)
        item_embeddings = torch.tensor(init_item_embedding_df['embedding'].apply(ast.literal_eval).tolist(), dtype=torch.float32)

        self.user_embeddings.weight.data = user_embeddings
        self.item_embeddings.weight.data = item_embeddings

        self.idx2user = init_user_embedding_df['user'].to_dict()
        self.user2idx = {user:idx for idx,user in self.idx2user.items()}

        self.idx2item = init_item_embedding_df['item'].to_dict()
        self.item2idx = {item:idx for idx,item in self.idx2item.items()}

    def create_agent_network(self, state_size):
        """
        Creates an agent network for second or third-order tasks.
        """
        return nn.Sequential(
            nn.Linear(state_size, self.embedding_size),
            nn.ReLU(),
            nn.Linear(self.embedding_size, 1),
            nn.Sigmoid()
        )
    

    def reload_embedding(self, new_user_embedding_path="path.csv", new_item_embedding_path="path.csv"):
        new_user_embedding_df = pd.read_csv(new_user_embedding_path)
        new_item_embedding_df = pd.read_csv(new_item_embedding_path)

        user_embeddings = torch.tensor(new_user_embedding_df['embedding'].apply(ast.literal_eval).tolist(), dtype=torch.float32)
        item_embeddings = torch.tensor(new_item_embedding_df['embedding'].apply(ast.literal_eval).tolist(), dtype=torch.float32)

        self.user_embeddings.weight.data = user_embeddings
        self.item_embeddings.weight.data = item_embeddings

        self.idx2user = new_user_embedding_df['user'].to_dict()
        self.user2idx = {user:idx for idx,user in self.idx2user.items()}

        self.idx2item = new_item_embedding_df['item'].to_dict()
        self.item2idx = {item:idx for idx,item in self.idx2item.items()}
        

    def forward(self, target_ids, support_1st, support_2nd=None, support_3rd=None, task="user", aggregation="GAT"):
        """
        Forward pass for the GeneralGNN model.

        Args:
            target_ids (Tensor): IDs of target users or items. Shape: [batch_size].
            support_1st (Tensor): First-order neighbors (items/users). Shape: [batch_size, num_neighbors_1st].
            support_2nd (Tensor, optional): Second-order neighbors (users/items). Shape: [batch_size, num_neighbors_2nd].
            support_3rd (Tensor, optional): Third-order neighbors (items/users). Shape: [batch_size, num_neighbors_3rd].
            task (str): Task type ("user" or "item").
            aggregation (str): Aggregation method ("GAT", "GraphSAGE", "FBNE", etc.).

        Returns:
            Tensor: Predicted embeddings for the target users/items. Shape: [batch_size, embedding_size].
        """
        # Get embeddings for first-order neighbors
        if task == "user":
            # First-order neighbors are items for user tasks'
            support_1st_idx = [self.item2idx[itemid] for itemid in support_1st]
            first_order_embeddings = self.item_embeddings(torch.Tensor(support_1st_idx).type(torch.long))  # Shape: [batch_size, num_neighbors_1st, embedding_size]
        elif task == "item":
            # First-order neighbors are users for item tasks
            support_1st_idx = [self.user2idx[userid] for userid in support_1st]
            first_order_embeddings = self.user_embeddings(torch.Tensor(support_1st_idx).type(torch.long))  # Shape: [batch_size, num_neighbors_1st, embedding_size]
        else:
            raise ValueError("Task must be 'user' or 'item'")

        # Aggregate first-order embeddings
        # print(self.first_order_weight.size())
        # print(first_order_embeddings.size())
        # print(self.aggregate_graphsage(first_order_embeddings).unsqueeze(dim=0).size()) #64
        # print(self.aggregate_graphsage(first_order_embeddings).unsqueeze(dim=0))

        first_order_agg = torch.matmul(self.aggregate_graphsage(first_order_embeddings).unsqueeze(dim=0), self.first_order_sage_weight)
        refine_weight = self.first_order_refine_weight

        # Handle second-order neighbors
        if support_2nd is not None:
            if task == "user":
                # Second-order neighbors are users for user tasks
                support_2nd_idx = [self.user2idx[userid] for userid in support_2nd]
                second_order_embeddings = self.user_embeddings(torch.Tensor(support_2nd_idx).type(torch.long))  # Shape: [batch_size, num_neighbors_2nd, embedding_size]
            elif task == "item":
                # Second-order neighbors are items for item tasks
                support_2nd_idx = [self.item2idx[itemid] for itemid in support_2nd]
                second_order_embeddings = self.item_embeddings(torch.Tensor(support_2nd_idx).type(torch.long))  # Shape: [batch_size, num_neighbors_2nd, embedding_size]
            
            second_order_agg = torch.matmul(self.aggregate_graphsage(second_order_embeddings), self.second_order_sage_weight).unsqueeze(dim = 0) # from (64) to (1,64)
            
            # Combine first and second-order aggregations
            combined_1st_2nd = torch.cat([first_order_agg, second_order_agg], dim=1)  # Shape: [batch_size, 2 * embedding_size]
            refine_weight = self.second_order_refine_weight
        else:
            combined_1st_2nd = first_order_agg

        # Handle third-order neighbors
        if support_3rd is not None:
            if task == "user":
                # Third-order neighbors are items for user tasks
                support_3rd_idx = [self.item2idx[itemid] for itemid in support_3rd]
                third_order_embeddings = self.item_embeddings(torch.Tensor(support_3rd_idx).type(torch.long))  # Shape: [batch_size, num_neighbors_3rd, embedding_size]
            elif task == "item":
                # Third-order neighbors are users for item tasks
                support_3rd_idx = [self.user2idx[userid] for userid in support_3rd]
                third_order_embeddings = self.user_embeddings(torch.Tensor(support_3rd_idx).type(torch.long))  # Shape: [batch_size, num_neighbors_3rd, embedding_size]

            third_order_agg = torch.matmul(self.aggregate_graphsage(third_order_embeddings), self.third_order_sage_weight).unsqueeze(dim=0)
            # Combine first, second, and third-order aggregations
            combined_1st_2nd_3rd = torch.cat([combined_1st_2nd, third_order_agg], dim=1)  # Shape: [batch_size, 3 * embedding_size]
            refine_weight = self.third_order_refine_weight
        else:
            combined_1st_2nd_3rd = combined_1st_2nd

        # Final embedding transformation
        if task == "user":
            targt_idx = [self.user2idx[target_ids]]
            target_embeddings = self.user_embeddings(torch.Tensor(targt_idx).type(torch.long))  # Shape: [batch_size, embedding_size]
        elif task == "item":
            targt_idx = [self.item2idx[target_ids]]
            target_embeddings = self.item_embeddings(torch.Tensor(targt_idx).type(torch.long))  # Shape: [batch_size, embedding_size]

        # Apply a transformation layer (e.g., MLP) to refine embeddings
        refined_embedding = self.refine_embedding(combined_1st_2nd_3rd, target_embeddings, refine_weight)

        #print(refined_embedding.size())

        return refined_embedding

    def _1st_user_task(self, support_item, target_user, training_phase):
        """
        First-order user task in PyTorch.
        Computes the user embedding by aggregating information from first-order neighbors (items).
        
        Args:
            support_item: Tensor of shape [batch_size, num_neighbors], indices of items.
            target_user: Tensor of shape [batch_size, embedding_size], target user embeddings.
            training_phase: Boolean indicating whether the model is in training mode.
        
        Returns:
            final_support_encode_user_task: Tensor of shape [batch_size, embedding_size], aggregated user embeddings.
            cosine_similarity: Tensor of shape [batch_size], cosine similarity between predicted and target embeddings.
            loss_user_task: Scalar, the loss value.
        """
        # Create aggregated user embedding from neighbors (items)
        final_support_encode_user_task = self._create_aggregator_network_user_task(support_item, training_phase)  # [batch_size, embedding_size]
        
        # Cosine similarity between predicted user embeddings and target user embeddings
        cosine_similarity = F.cosine_similarity(final_support_encode_user_task, target_user, dim=1)  # [batch_size]
        
        # Define loss (negative mean cosine similarity)
        loss_user_task = -torch.mean(cosine_similarity)
        
        return final_support_encode_user_task, cosine_similarity, loss_user_task

    def _1st_item_task(self):
        """
        First-order item task in PyTorch.
        Computes the item embedding by aggregating information from first-order neighbors (users).
        """
        # Load pretrained user embeddings
        self.pre_train_user_ebd = torch.tensor(np.load(self.settings.pre_train_user_ebd_path), dtype=torch.float32)
        
        # Create aggregated item embedding from neighbors (users)
        final_support_encode_item_task = self._create_aggregator_network_item_task('active')  # [batch_size, embedding_size]
        
        # Cosine similarity between predicted item embeddings and target item embeddings
        cosine_similarity = F.cosine_similarity(final_support_encode_item_task, self.target_item, dim=1)  # [batch_size]
        
        # Define loss (negative mean cosine similarity)
        loss_item_task = -torch.mean(cosine_similarity)
        
        return final_support_encode_item_task, cosine_similarity, loss_item_task


    def _2nd_user_task(self, name, support_user_2nd, support_item_1st):
        """
        Second-order user task in PyTorch.
        Aggregates second-order neighbors (users and items) to compute user embeddings.
        """
        if name == 'GraphSAGE':
            # Lookup embeddings for second-order neighbors (users)
            support_ori_ebd_2nd = self.user_embeddings(support_user_2nd)  # [batch_size, n2, embedding_size]
            support_encode_2nd = torch.mean(self.encoder(support_ori_ebd_2nd), dim=1)  # [batch_size, embedding_size]
            ori_2nd_ebd = torch.mean(support_ori_ebd_2nd, dim=1)  # [batch_size, embedding_size]

            # Lookup embeddings for first-order neighbors (items)
            support_ori_ebd_1st = self.item_embeddings(support_item_1st)  # [batch_size, n1, embedding_size]
            ori_1st_ebd = torch.mean(support_ori_ebd_1st, dim=1)  # [batch_size, embedding_size]

            # Concatenate embeddings and apply transformations
            aggregate_2nd = torch.cat([support_encode_2nd, ori_2nd_ebd, ori_1st_ebd], dim=1)  # [batch_size, 3 * embedding_size]
            refined_first_neigh_ebd = torch.matmul(aggregate_2nd, self.second_order_weight)  # [batch_size, embedding_size]
            refined_target_ebd = torch.matmul(refined_first_neigh_ebd, self.first_order_weight)  # [batch_size, embedding_size]

            # Final prediction for second-order user task
            predict_u_2nd = refined_target_ebd
            cosine_similarity = F.cosine_similarity(predict_u_2nd, self.target_user, dim=1)
            loss_2nd_user = -torch.mean(cosine_similarity)

            return predict_u_2nd, cosine_similarity, loss_2nd_user


    def _2nd_item_task(self, name, support_item_2nd, support_user_1st):
        """
        Second-order item task in PyTorch.
        Aggregates second-order neighbors (items and users) to compute item embeddings.
        """
        if name == 'GraphSAGE':
            # Lookup embeddings for second-order neighbors (items)
            support_ori_ebd_2nd = self.item_embeddings(support_item_2nd)  # [batch_size, n2, embedding_size]
            support_encode_2nd = torch.mean(self.encoder(support_ori_ebd_2nd), dim=1)  # [batch_size, embedding_size]
            ori_2nd_ebd = torch.mean(support_ori_ebd_2nd, dim=1)  # [batch_size, embedding_size]

            # Lookup embeddings for first-order neighbors (users)
            support_ori_ebd_1st = self.user_embeddings(support_user_1st)  # [batch_size, n1, embedding_size]
            ori_1st_ebd = torch.mean(support_ori_ebd_1st, dim=1)  # [batch_size, embedding_size]

            # Concatenate embeddings and apply transformations
            aggregate_2nd = torch.cat([support_encode_2nd, ori_2nd_ebd, ori_1st_ebd], dim=1)  # [batch_size, 3 * embedding_size]
            refined_first_neigh_ebd = torch.matmul(aggregate_2nd, self.second_order_weight)  # [batch_size, embedding_size]
            refined_target_ebd = torch.matmul(refined_first_neigh_ebd, self.first_order_weight)  # [batch_size, embedding_size]

            # Final prediction for second-order item task
            predict_i_2nd = refined_target_ebd
            cosine_similarity = F.cosine_similarity(predict_i_2nd, self.target_item, dim=1)
            loss_2nd_item = -torch.mean(cosine_similarity)

            return predict_i_2nd, cosine_similarity, loss_2nd_item

    def _3rd_user_task(self, name, support_item_3rd, support_user_2nd_, support_item_1st_):
        """
        Third-order user task in PyTorch.
        Aggregates third-order neighbors (items and users) to compute user embeddings.
        """
        if name == 'GraphSAGE':
            # Third-order embeddings
            support_ori_ebd_3rd = self.item_embeddings(support_item_3rd)
            support_encode_3rd = torch.mean(self.encoder(support_ori_ebd_3rd), dim=1)  # [batch_size, embedding_size]
            ori_3rd_ebd  = torch.mean(support_ori_ebd_3rd, dim=1) # [batch_size, embedding_size]

            # Second-order embeddings
            support_ori_ebd_2nd = self.user_embeddings(support_user_2nd_)
            support_encode_2nd = torch.mean(self.encoder(support_ori_ebd_2nd), dim=1)  # [batch_size, embedding_size]
            ori_2nd_ebd = torch.mean(support_ori_ebd_2nd, dim=1)  # [batch_size, embedding_size]

            # First-order embeddings
            support_ori_ebd_1st = self.item_embeddings(support_item_1st_)
            ori_1st_ebd = torch.mean(support_ori_ebd_1st, dim=1)  # [batch_size, embedding_size]

            # Aggregate third-order to second-order
            aggregate_3rd = torch.cat([support_encode_3rd, ori_3rd_ebd, ori_2nd_ebd], dim=1)  # [batch_size, 3 * embedding_size]
            refined_second_neigh_ebd = torch.matmul(aggregate_3rd, self.third_order_weight)  # [batch_size, embedding_size]

            # Aggregate second-order to first-order
            aggregate_2nd = torch.cat([refined_second_neigh_ebd, support_encode_2nd, ori_1st_ebd], dim=1)  # [batch_size, 3 * embedding_size]
            refined_first_neigh_ebd = torch.matmul(aggregate_2nd, self.second_order_weight)  # [batch_size, embedding_size]
            refined_target_ebd = torch.matmul(refined_first_neigh_ebd, self.first_order_weight)  # [batch_size, embedding_size]

            # Final prediction for third-order user task
            predict_u_3rd = refined_target_ebd
            cosine_similarity = F.cosine_similarity(predict_u_3rd, self.target_user, dim=1)
            loss_3rd_user = -torch.mean(cosine_similarity)
            return predict_u_3rd, cosine_similarity, loss_3rd_user 


    def _3rd_item_task(self, name, support_user_3rd, support_item_2nd_, support_user_1st_):
        """
        Third-order item task in PyTorch.
        Aggregates third-order neighbors (users and items) to compute item embeddings.
        """
        if name == 'GraphSAGE':
            # Lookup embeddings for third-order neighbors (users)
            support_ori_ebd_3rd = self.user_embeddings(support_user_3rd)  # [batch_size, n3, embedding_size]
            ori_3rd_ebd = torch.mean(support_ori_ebd_3rd, dim=1)  # [batch_size, embedding_size]
            support_encode_3rd = self.encoder(support_ori_ebd_3rd)  # Apply encoder [batch_size, n3, embedding_size] -> [batch_size, embedding_size]

            # Lookup embeddings for second-order neighbors (items)
            support_ori_ebd_2nd = self.item_embeddings(support_item_2nd_)  # [batch_size, n2, embedding_size]
            ori_2nd_ebd = torch.mean(support_ori_ebd_2nd, dim=1)  # [batch_size, embedding_size]
            support_encode_2nd = self.encoder(support_ori_ebd_2nd)  # Apply encoder [batch_size, n2, embedding_size] -> [batch_size, embedding_size]

            # Lookup embeddings for first-order neighbors (users)
            support_ori_ebd_1st = self.user_embeddings(support_user_1st_)  # [batch_size, n1, embedding_size]
            ori_1st_ebd = torch.mean(support_ori_ebd_1st, dim=1)  # [batch_size, embedding_size]

            # Combine embeddings for third-order aggregation
            aggregate_3rd = torch.cat([support_encode_3rd, ori_3rd_ebd, ori_2nd_ebd], dim=1)  # [batch_size, 3 * embedding_size]
            refined_second_neigh_ebd = torch.matmul(aggregate_3rd, self.third_order_weight)  # [batch_size, embedding_size]

            # Aggregate second-order to first-order
            aggregate_2nd = torch.cat([refined_second_neigh_ebd, support_encode_2nd, ori_1st_ebd], dim=1)  # [batch_size, 3 * embedding_size]
            refined_first_neigh_ebd = torch.matmul(aggregate_2nd, self.second_order_weight)  # [batch_size, embedding_size]
            refined_target_ebd = torch.matmul(refined_first_neigh_ebd, self.first_order_weight)  # [batch_size, embedding_size]

            # Final prediction for third-order item task
            predict_i_3rd = refined_target_ebd
            cosine_similarity = F.cosine_similarity(predict_i_3rd, self.target_item, dim=1)
            loss_3rd_item = -torch.mean(cosine_similarity)

            return predict_i_3rd, cosine_similarity, loss_3rd_item
    ################ BELOW ARE HELPER FUNCTIONS ####################
    def aggregate_gat(self, neighbor_embeddings):
        """
        Perform attention-weighted aggregation for GAT.
        Args:
            neighbor_embeddings (Tensor): Neighbor embeddings. Shape: [batch_size, num_neighbors, embedding_size].
        Returns:
            Tensor: Aggregated embedding. Shape: [batch_size, embedding_size].
        """
        attention_weights = torch.nn.functional.softmax(neighbor_embeddings.mean(dim=-1), dim=1)  # [batch_size, num_neighbors]
        aggregated = torch.sum(attention_weights.unsqueeze(-1) * neighbor_embeddings, dim=1)  # [batch_size, embedding_size]
        return aggregated

    def aggregate_graphsage(self, neighbor_embeddings):
        """
        Perform mean pooling aggregation for GraphSAGE.
        Args:
            neighbor_embeddings (Tensor): Neighbor embeddings. Shape: [batch_size, num_neighbors, embedding_size].
        Returns:
            Tensor: Aggregated embedding. Shape: [batch_size, embedding_size].
        """
        aggregated_embedding = torch.mean(neighbor_embeddings, dim=0)
        return aggregated_embedding # [batch_size, embedding_size]



    def refine_embedding(self, combined_embeddings, target_embeddings, weight):
        """
        Refines the embeddings by applying a transformation layer.
        Args:
            combined_embeddings (Tensor): Combined neighbor embeddings. Shape: [batch_size, N * embedding_size].
            target_embeddings (Tensor): Target embeddings. Shape: [batch_size, embedding_size].
        Returns:
            Tensor: Refined embeddings. Shape: [batch_size, embedding_size].
        """
        # Apply transformation to the combined embedding
        refined = torch.mm(combined_embeddings, weight)  # [batch_size, embedding_size]
        # refined = torch.mm(combined_embeddings, self.second_order_weight)  # [batch_size, embedding_size]
        refined += target_embeddings  # Residual connection
        return F.relu(refined)


    def _create_aggregator_network_user_task(self, support_item, training_phase):
        """
        Creates the aggregator network for the first-order user task.
        Aggregates embeddings of items (first-order neighbors) to compute user embeddings.
        
        Args:
            support_item: Tensor of shape [batch_size, num_neighbors], indices of items.
            training_phase: Boolean indicating whether the model is in training mode.
            
        Returns:
            final_support_encode_user_task: Tensor of shape [batch_size, embedding_size], aggregated user embeddings.
        """
        # Lookup embeddings for support items (first-order neighbors)
        support_ebd = self.item_embeddings(support_item)  # [batch_size, num_neighbors, embedding_size]

        # Apply encoding using the Encoder (multi-head attention mechanism)
        support_encoded = self.encoder(support_ebd, training_phase)  # [batch_size, num_neighbors, embedding_size]

        # Aggregate embeddings by averaging across neighbors
        final_support_encode_user_task = torch.mean(support_encoded, dim=1)  # [batch_size, embedding_size]

        return final_support_encode_user_task


    def _create_aggregator_network_item_task(self, support_user, training_phase):
        """
        Creates the aggregator network for the first-order item task.
        Aggregates embeddings of users (first-order neighbors) to compute item embeddings.
        
        Args:
            support_user: Tensor of shape [batch_size, num_neighbors], indices of users.
            training_phase: Boolean indicating whether the model is in training mode.
            
        Returns:
            final_support_encode_item_task: Tensor of shape [batch_size, embedding_size], aggregated item embeddings.
        """
        # Lookup embeddings for support users (first-order neighbors)
        support_ebd = self.user_embeddings(support_user)  # [batch_size, num_neighbors, embedding_size]

        # Apply encoding using the Encoder (multi-head attention mechanism)
        support_encoded = self.encoder(support_ebd, training_phase)  # [batch_size, num_neighbors, embedding_size]

        # Aggregate embeddings by averaging across neighbors
        final_support_encode_item_task = torch.mean(support_encoded, dim=1)  # [batch_size, embedding_size]

        return final_support_encode_item_task


    def glorot(self, shape, name=None):
        """
        Glorot & Bengio (AISTATS 2010) initialization for weight matrices.
        
        Args:
            shape (tuple): Shape of the weight matrix (e.g., (input_dim, output_dim)).
            name (str, optional): Name for the weight variable (not used in PyTorch but kept for compatibility).

        Returns:
            torch.Tensor: Initialized weight matrix.
        """
        init_range = np.sqrt(6.0 / (shape[0] + shape[1]))
        return torch.empty(shape).uniform_(-init_range, init_range)

            
class Encoder(nn.Module):
    def __init__(self, embedding_size, num_blocks, num_heads, d_ff, dropout_rate):
        """
        Encoder module for multi-head attention and feedforward transformations.

        Args:
            embedding_size (int): Size of embeddings.
            num_blocks (int): Number of attention blocks.
            num_heads (int): Number of attention heads.
            d_ff (int): Hidden layer size in feedforward network.
            dropout_rate (float): Dropout rate.
        """
        super().__init__()
        self.embedding_size = embedding_size
        self.num_blocks = num_blocks
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.dropout_rate = dropout_rate

        # Define multi-head attention and feedforward layers for each block
        self.attention_blocks = nn.ModuleList([
            nn.MultiheadAttention(embed_dim=embedding_size, num_heads=num_heads, dropout=dropout_rate)
            for _ in range(num_blocks)
        ])
        self.feedforward_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embedding_size, d_ff),
                nn.ReLU(),
                nn.Linear(d_ff, embedding_size),
                nn.Dropout(dropout_rate)
            )
            for _ in range(num_blocks)
        ])
        self.layer_norms_attention = nn.ModuleList([nn.LayerNorm(embedding_size) for _ in range(num_blocks)])
        self.layer_norms_ff = nn.ModuleList([nn.LayerNorm(embedding_size) for _ in range(num_blocks)])

    def forward(self, input):
        """
        Forward pass for the encoder.

        Args:
            input (Tensor): Input tensor of shape [batch_size, num_neighbors, embedding_size].

        Returns:
            enc (Tensor): Output tensor of shape [batch_size, num_neighbors, embedding_size].
        """
        # Scale the input embeddings
        enc = input * (self.embedding_size ** 0.5)  # [b, n, e]

        # Process through each block
        for i in range(self.num_blocks):
            # Multi-head self-attention
            enc_transposed = enc.permute(1, 0, 2)  # Convert to [n, b, e] for PyTorch MultiheadAttention
            attn_output, _ = self.attention_blocks[i](enc_transposed, enc_transposed, enc_transposed)
            attn_output = attn_output.permute(1, 0, 2)  # Convert back to [b, n, e]
            # Add and normalize
            enc = self.layer_norms_attention[i](enc + attn_output)

            # Feedforward
            ff_output = self.feedforward_blocks[i](enc)
            # Add and normalize
            enc = self.layer_norms_ff[i](enc + ff_output)

        return enc  # [b, n, e]