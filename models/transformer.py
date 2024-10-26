from .base_model import BaseModel
from utils.dataloader import load_data_df, load_item_df, load_user_features, DATA_FORMAT

import os
import numpy as np
import pandas as pd
from typeguard import typechecked
from typing import Tuple, Dict, Union, List
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, BertConfig
from transformers import get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


class BERT4Rec(nn.Module):
    def __init__(
        self,
        num_items: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        max_position_embeddings: int = 20,
        dropout: float = 0.1,
    ):
        """
        Initialize the BERT4Rec model.

        Parameters:
        - num_items (int): Number of unique items.
        - hidden_size (int): Hidden size of BERT.
        - num_layers (int): Number of transformer layers.
        - num_heads (int): Number of attention heads.
        - max_position_embeddings (int): Maximum sequence length.
        - dropout (float): Dropout probability.
        """
        super(BERT4Rec, self).__init__()
        config = BertConfig(
            vocab_size=num_items + 3,  # Including [PAD], [BOS], [EOS]
            hidden_size=hidden_size,
            num_hidden_layers=num_layers,
            num_attention_heads=num_heads,
            intermediate_size=hidden_size * 4,
            max_position_embeddings=max_position_embeddings,
            hidden_dropout_prob=dropout,
            attention_probs_dropout_prob=dropout,
            type_vocab_size=1,
            output_attentions=False,
            output_hidden_states=False,
        )
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_items + 3)

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass of the BERT4Rec model.

        Parameters:
        - input_ids (torch.Tensor): Input token IDs.
        - attention_mask (torch.Tensor): Attention mask.

        Returns:
        - torch.Tensor: Logits for each position and item.
        """
        outputs = self.bert(
            input_ids=input_ids, attention_mask=attention_mask
        )
        last_hidden_state = outputs.last_hidden_state
        pooled_output = self.dropout(last_hidden_state)
        logits = self.classifier(pooled_output)
        return logits

    def predict_next_item(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Predict the next item given a sequence.

        Parameters:
        - input_ids (torch.Tensor): Input token IDs.
        - attention_mask (torch.Tensor): Attention mask.

        Returns:
        - torch.Tensor: Logits for the next item.
        """
        logits = self.forward(input_ids, attention_mask)
        next_logits = logits[:, -1, :]
        return next_logits


class ItemTokenizer:
    def __init__(self, num_items: int):
        """
        Initialize the ItemTokenizer with the number of items.

        Parameters:
        - num_items (int): Number of unique items.
        """
        self.num_items = num_items
        self.pad_token_id = 0
        self.bos_token_id = 1
        self.eos_token_id = 2
        self.vocab_size = num_items + 3

    def encode_plus(
        self,
        sequence: List[int],
        add_special_tokens: bool = True,
        max_length: int = None,
        padding: str = "max_length",
        truncation: bool = True,
        return_attention_mask: bool = True,
        return_tensors: str = "pt",
    ) -> Dict[str, torch.Tensor]:
        """
        Encode a sequence of items with optional padding and truncation.

        Parameters:
        - sequence (List[int]): List of item indices.
        - add_special_tokens (bool): Whether to add special tokens [BOS] and [EOS].
        - max_length (int): Maximum sequence length.
        - padding (str): Padding strategy ('max_length' or None).
        - truncation (bool): Whether to truncate sequences longer than max_length.
        - return_attention_mask (bool): Whether to return the attention mask.
        - return_tensors (str): Type of tensors to return ('pt' for PyTorch).

        Returns:
        - Dict[str, torch.Tensor]: Dictionary containing 'input_ids' and 'attention_mask'.
        """
        if not sequence:
            sequence = [self.pad_token_id]

        if add_special_tokens:
            tokens = [self.bos_token_id] + sequence + [self.eos_token_id]
        else:
            tokens = sequence

        if padding == "max_length" and max_length is not None:
            pad_length = max_length - len(tokens)
            if pad_length > 0:
                tokens += [self.pad_token_id] * pad_length
            elif truncation:
                tokens = tokens[:max_length]

        attention_mask = [1] * len(tokens)
        if padding == "max_length" and max_length is not None:
            attention_mask += [0] * (max_length - len(attention_mask))

        if return_tensors == "pt":
            tokens = torch.tensor(tokens, dtype=torch.long)
            attention_mask = torch.tensor(attention_mask, dtype=torch.long)

        return {
            "input_ids": tokens,
            "attention_mask": attention_mask if return_attention_mask else None,
        }


class RecDataset(Dataset):
    def __init__(
        self,
        inputs: List[List[int]],
        targets: List[int],
        tokenizer: ItemTokenizer,
        max_length: int,
    ):
        """
        Initialize the RecDataset with input sequences and target items.

        Parameters:
        - inputs (List[List[int]]): List of input item sequences.
        - targets (List[int]): List of target items.
        - tokenizer (ItemTokenizer): Tokenizer for encoding sequences.
        - max_length (int): Maximum sequence length.
        """
        self.inputs = inputs
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get the item at the specified index.

        Parameters:
        - idx (int): Index of the item.

        Returns:
        - Dict[str, torch.Tensor]: Dictionary containing 'input_ids', 'attention_mask', and 'labels'.
        """
        seq = self.inputs[idx]
        target = self.targets[idx]

        # Adjust item indices by adding 1 to account for [PAD], [BOS], [EOS]
        adjusted_seq = [item + 1 for item in seq]

        encoding = self.tokenizer.encode_plus(
            sequence=adjusted_seq,
            add_special_tokens=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(target, dtype=torch.long),
        }
    

@typechecked
class BERT4RecModel(BaseModel):
    def __init__(
        self,
        size: str,  # "100k" or "1m"
        max_seq_length: int = 20,
        batch_size: int = 32,
        num_epochs: int = 5,
        learning_rate: float = 2e-5,
        device: str = "auto",  # 'auto', 'cuda', 'mps', or 'cpu'
    ) -> None:
        """
        Initialize the BERT4RecModel with specified hyperparameters and dataset size.

        Parameters:
        - size (str): Dataset size, "100k" or "1m".
        - max_seq_length (int): Maximum sequence length.
        - batch_size (int): Batch size for training.
        - num_epochs (int): Number of training epochs.
        - learning_rate (float): Learning rate for optimizer.
        - device (str): Device to run the model on ('auto', 'cuda', 'mps', 'cpu').
        """
        super().__init__()

        # Validate dataset size
        if size not in DATA_FORMAT:
            raise ValueError(
                f"Invalid size: {size}. Choose from {list(DATA_FORMAT.keys())}."
            )

        self.size = size
        self.max_seq_length = max_seq_length
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.device = self._get_device(device)

        # Initialize LabelEncoders
        self.user_encoder = LabelEncoder()
        self.item_encoder = LabelEncoder()

        # Load and prepare data
        self._load_and_prepare_data()

        # Initialize tokenizer and datasets
        self.tokenizer = ItemTokenizer(num_items=self.num_items)
        self._create_datasets()

        # Create dataloaders
        self.train_loader = DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0
        )
        self.val_loader = DataLoader(
            self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0
        )

        # Initialize model
        self.model = BERT4Rec(num_items=self.num_items).to(self.device)

        # Define loss function and optimizer
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.learning_rate
        )

        # Learning rate scheduler
        total_steps = len(self.train_loader) * self.num_epochs
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer, num_warmup_steps=0, num_training_steps=total_steps
        )

    def _get_device(self, device_str: str) -> torch.device:
        """
        Determine the device to use for computation.

        Parameters:
        - device_str (str): 'auto', 'cuda', 'mps', or 'cpu'.

        Returns:
        - torch.device: The device to use.
        """
        if device_str == "auto":
            if torch.cuda.is_available():
                print("Using CUDA (GPU)")
                return torch.device("cuda")
            elif torch.backends.mps.is_available():
                print("Using MPS (Apple Silicon GPU)")
                return torch.device("mps")
            else:
                print("Using CPU")
                return torch.device("cpu")
        elif device_str == "cuda":
            if torch.cuda.is_available():
                print("Using CUDA (GPU)")
                return torch.device("cuda")
            else:
                raise ValueError("CUDA is not available on this machine.")
        elif device_str == "mps":
            if torch.backends.mps.is_available():
                print("Using MPS (Apple Silicon GPU)")
                return torch.device("mps")
            else:
                raise ValueError("MPS is not available on this machine.")
        elif device_str == "cpu":
            print("Using CPU")
            return torch.device("cpu")
        else:
            raise ValueError(
                "Invalid device option. Choose from 'auto', 'cuda', 'mps', or 'cpu'."
            )

    def _load_and_prepare_data(self) -> None:
        """
        Load and preprocess the dataset.
        """
        # Load train and test data
        train_df = load_data_df(self.size, split="train")
        test_df = load_data_df(self.size, split="test")

        # Combine and encode data
        combined_df = pd.concat([train_df, test_df], axis=0)
        combined_df["user"] = self.user_encoder.fit_transform(combined_df["userId"])
        combined_df["item"] = self.item_encoder.fit_transform(combined_df["movieId"])

        self.num_users = combined_df["user"].nunique()
        self.num_items = combined_df["item"].nunique()

        # Split back into train and test
        train_df = combined_df.iloc[: len(train_df)].copy()
        test_df = combined_df.iloc[len(train_df) :].copy()

        # Sort and create sequences
        train_df = train_df.sort_values(["user", "timestamp"])
        user_sequences = train_df.groupby("user")["item"].apply(list).reset_index()

        # Create input sequences and targets
        input_sequences = []
        target_items = []

        for seq in user_sequences["item"]:
            if len(seq) < 2:
                continue
            for i in range(1, len(seq)):
                start = max(i - (self.max_seq_length - 1), 0)
                input_seq = seq[start:i]
                target = seq[i]
                input_sequences.append(input_seq)
                target_items.append(target)

        # Split into train and validation sets
        (
            self.train_inputs,
            self.val_inputs,
            self.train_targets,
            self.val_targets,
        ) = train_test_split(
            input_sequences, target_items, test_size=0.1, random_state=42
        )
        print(f"Training set size: {len(self.train_inputs)}")
        print(f"Validation set size: {len(self.val_inputs)}")

    def _create_datasets(self) -> None:
        """
        Create training and validation datasets.
        """
        self.train_dataset = RecDataset(
            inputs=self.train_inputs,
            targets=self.train_targets,
            tokenizer=self.tokenizer,
            max_length=self.max_seq_length,
        )
        self.val_dataset = RecDataset(
            inputs=self.val_inputs,
            targets=self.val_targets,
            tokenizer=self.tokenizer,
            max_length=self.max_seq_length,
        )

    def train_epoch(
        self, data_loader: DataLoader, criterion: nn.Module, optimizer: torch.optim.Optimizer
    ) -> Tuple[float, float]:
        """
        Train the model for one epoch.

        Parameters:
        - data_loader (DataLoader): DataLoader for training data.
        - criterion (nn.Module): Loss function.
        - optimizer (torch.optim.Optimizer): Optimizer.

        Returns:
        - Tuple[float, float]: Average loss and accuracy for the epoch.
        """
        self.model.train()
        losses = []
        correct = 0
        total = 0

        for batch in tqdm(data_loader, desc="Training"):
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)

            optimizer.zero_grad()
            logits = self.model(input_ids=input_ids, attention_mask=attention_mask)
            last_logits = logits[:, -1, :]  # Get logits for the last token

            loss = criterion(last_logits, labels)
            loss.backward()
            optimizer.step()
            self.scheduler.step()

            losses.append(loss.item())

            _, preds = torch.max(last_logits, dim=1)
            correct += torch.sum(preds == labels)
            total += labels.size(0)

        avg_loss = np.mean(losses)
        accuracy = correct.float() / total
        return avg_loss, accuracy.item()

    def eval_model(
        self, data_loader: DataLoader, criterion: nn.Module
    ) -> Tuple[float, float]:
        """
        Evaluate the model.

        Parameters:
        - data_loader (DataLoader): DataLoader for validation data.
        - criterion (nn.Module): Loss function.

        Returns:
        - Tuple[float, float]: Average loss and accuracy for the evaluation.
        """
        self.model.eval()
        losses = []
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Evaluating"):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                logits = self.model(input_ids=input_ids, attention_mask=attention_mask)
                last_logits = logits[:, -1, :]  # Get logits for the last token

                loss = criterion(last_logits, labels)
                losses.append(loss.item())

                _, preds = torch.max(last_logits, dim=1)
                correct += torch.sum(preds == labels)
                total += labels.size(0)

        avg_loss = np.mean(losses)
        accuracy = correct.float() / total
        return avg_loss, accuracy.item()

    def get_topk_accuracy(
        self, model: nn.Module, data_loader: DataLoader, device: torch.device, k: int = 10
    ) -> float:
        """
        Compute the Top-K accuracy of the model.

        Parameters:
        - model (nn.Module): Trained model.
        - data_loader (DataLoader): DataLoader for validation data.
        - device (torch.device): Device to perform computations.
        - k (int): Top-K items to consider.

        Returns:
        - float: Top-K accuracy.
        """
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in tqdm(data_loader, desc=f"Computing Top-{k} Accuracy"):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                logits = model.predict_next_item(input_ids=input_ids, attention_mask=attention_mask)
                topk_preds = torch.topk(logits, k, dim=1).indices  # (batch_size, k)

                # Adjust labels to match the token ids (labels +1)
                # Since the labels in the dataset are original item indices, and the model's output includes [PAD], [BOS], [EOS], the target label should also be shifted
                labels_adjusted = labels + 1  # [BOS] is 1, [EOS] is 2

                # Compare the true label with the top-k predictions
                correct += torch.sum(topk_preds == labels_adjusted.unsqueeze(1))
                total += labels.size(0)

        topk_acc = correct.float() / total
        return topk_acc.item()

    def train_model(self) -> None:
        """
        Train the BERT4Rec model across multiple epochs and evaluate on validation data.
        """
        for epoch in range(self.num_epochs):
            print(f"\nEpoch {epoch + 1}/{self.num_epochs}")
            print("-" * 10)

            train_loss, train_acc = self.train_epoch(
                data_loader=self.train_loader,
                criterion=self.criterion,
                optimizer=self.optimizer,
            )
            print(f"Training loss: {train_loss:.4f}, Training accuracy: {train_acc:.4f}")

            val_loss, val_acc = self.eval_model(
                data_loader=self.val_loader,
                criterion=self.criterion,
            )
            print(f"Validation loss: {val_loss:.4f}, Validation accuracy: {val_acc:.4f}")

        # After training, compute Top-K accuracy
        topk = 10
        topk_acc = self.get_topk_accuracy(
            model=self.model,
            data_loader=self.val_loader,
            device=self.device,
            k=topk,
        )
        print(f"Validation Top-{topk} accuracy: {topk_acc:.4f}")

    def predict(self, user: Union[int, str], item: Union[int, str]) -> float:
        """
        Predict the interaction score between a user and an item.

        Parameters:
        - user (int or str): The user identifier.
        - item (int or str): The item identifier.

        Returns:
        - float: The predicted interaction score.
        """
        self.model.eval()
        with torch.no_grad():
            if isinstance(user, str):
                try:
                    user_idx = self.user_encoder.transform([user])[0]
                except ValueError:
                    print("User not in training data.")
                    return 0.0
            elif isinstance(user, int):
                if user < self.num_users:
                    user_idx = user
                else:
                    print("User index out of range.")
                    return 0.0
            else:
                print("Invalid user type.")
                return 0.0

            if isinstance(item, str):
                try:
                    item_idx = self.item_encoder.transform([item])[0]
                except ValueError:
                    print("Item not in training data.")
                    return 0.0
            elif isinstance(item, int):
                if item < self.num_items:
                    item_idx = item
                else:
                    print("Item index out of range.")
                    return 0.0
            else:
                print("Invalid item type.")
                return 0.0

            # Prepare input sequence with the item as the next token
            # Here, we assume that we have access to the user's last sequence
            # For simplicity, we'll use the entire sequence as input
            # In practice, you might need to retrieve the user's history

            # Placeholder: retrieve the user's last sequence
            user_history = self.train_df[self.train_df["user"] == user_idx]
            if user_history.empty:
                print("No history for the user.")
                return 0.0
            last_seq = user_history.sort_values("timestamp")["item"].tolist()
            if not last_seq:
                print("User history is empty.")
                return 0.0

            # Limit the sequence to max_seq_length -1
            input_seq = last_seq[-(self.max_seq_length - 1) :] if len(last_seq) >= self.max_seq_length - 1 else last_seq
            input_seq = input_seq[-(self.max_seq_length - 1) :]  # Ensure max length

            # Encode the input sequence
            adjusted_seq = [item + 1 for item in input_seq]
            encoding = self.tokenizer.encode_plus(
                sequence=adjusted_seq,
                add_special_tokens=True,
                max_length=self.max_seq_length,
                padding="max_length",
                truncation=True,
                return_attention_mask=True,
                return_tensors="pt",
            )

            input_ids = encoding["input_ids"].to(self.device)
            attention_mask = encoding["attention_mask"].to(self.device)

            # Predict the next item logits
            logits = self.model.predict_next_item(input_ids=input_ids, attention_mask=attention_mask)
            # Apply softmax to get probabilities
            probabilities = torch.softmax(logits, dim=1)
            # Get the probability of the target item
            target_prob = probabilities[0, item_idx + 1].item()  # +1 for [BOS]

            return target_prob