import torch
import torch.nn as nn
import torch.optim as optim

def train_first_order_task(model, train_data, valid_data, epochs, device, task="user"):
    """
    Train GeneralGNN for first-order tasks (user or item) without batching.
    
    Args:
        model (nn.Module): The GNN model.
        train_data (tuple): (target_ids, support_1st, support_2nd, support_3rd, oracle_embeddings).
        valid_data (tuple): (target_ids, support_1st, support_2nd, support_3rd, oracle_embeddings).
        epochs (int): Number of epochs.
        device (torch.device): Device to run the training on.
        task (str): "user" or "item".
    """
    optimizer = optim.Adagrad(model.parameters(), lr=model.learning_rate)
    loss_fn = nn.CosineEmbeddingLoss()
    model.to(device)

    target_ids_train, support_1st_train, _, _, oracle_embeddings_train = train_data
    target_ids_valid, support_1st_valid, _, _, oracle_embeddings_valid = valid_data

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()

        # 前向传播
        predicted_embeddings = model(
            target_ids_train, support_1st_train, None, None, task=task
        )

        # 计算损失
        target = torch.ones(predicted_embeddings.size(0), device=device)
        loss = loss_fn(predicted_embeddings, oracle_embeddings_train, target)

        # 反向传播和优化
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch}/{epochs} - First-Order {task} Task: Train Loss = {loss.item():.4f}")

        # 验证
        validate_task(model, valid_data, loss_fn, device, task, "First-Order")

def train_second_order_task(model, train_data, valid_data, epochs, device, task="user"):
    """
    Train GeneralGNN for second-order tasks (user or item) without batching.
    
    Args:
        model (nn.Module): The GNN model.
        train_data (tuple): (target_ids, support_1st, support_2nd, support_3rd, oracle_embeddings).
        valid_data (tuple): (target_ids, support_1st, support_2nd, support_3rd, oracle_embeddings).
        epochs (int): Number of epochs.
        device (torch.device): Device to run the training on.
        task (str): "user" or "item".
    """
    optimizer = optim.Adagrad(model.parameters(), lr=model.learning_rate)
    loss_fn = nn.CosineEmbeddingLoss()
    model.to(device)

    target_ids_train, support_1st_train, support_2nd_train, _, oracle_embeddings_train = train_data
    target_ids_valid, support_1st_valid, support_2nd_valid, _, oracle_embeddings_valid = valid_data

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()

        # 前向传播
        predicted_embeddings = model(
            target_ids_train, support_1st_train, support_2nd_train, None, task=task
        )

        # 计算损失
        target = torch.ones(predicted_embeddings.size(0), device=device)
        loss = loss_fn(predicted_embeddings, oracle_embeddings_train, target)

        # 反向传播和优化
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch}/{epochs} - Second-Order {task} Task: Train Loss = {loss.item():.4f}")

        # 验证
        validate_task(model, valid_data, loss_fn, device, task, "Second-Order")

def train_third_order_task(model, train_data, valid_data, epochs, device, task="user"):
    """
    Train GeneralGNN for third-order tasks (user or item) without batching.
    
    Args:
        model (nn.Module): The GNN model.
        train_data (tuple): (target_ids, support_1st, support_2nd, support_3rd, oracle_embeddings).
        valid_data (tuple): (target_ids, support_1st, support_2nd, support_3rd, oracle_embeddings).
        epochs (int): Number of epochs.
        device (torch.device): Device to run the training on.
        task (str): "user" or "item".
    """
    optimizer = optim.Adagrad(model.parameters(), lr=model.learning_rate)
    loss_fn = nn.CosineEmbeddingLoss()
    model.to(device)

    target_ids_train, support_1st_train, support_2nd_train, support_3rd_train, oracle_embeddings_train = train_data
    target_ids_valid, support_1st_valid, support_2nd_valid, support_3rd_valid, oracle_embeddings_valid = valid_data

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()

        # 前向传播
        predicted_embeddings = model(
            target_ids_train, support_1st_train, support_2nd_train, support_3rd_train, task=task
        )

        # 计算损失
        target = torch.ones(predicted_embeddings.size(0), device=device)
        loss = loss_fn(predicted_embeddings, oracle_embeddings_train, target)

        # 反向传播和优化
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch}/{epochs} - Third-Order {task} Task: Train Loss = {loss.item():.4f}")

        # 验证
        validate_task(model, valid_data, loss_fn, device, task, "Third-Order")

def validate_task(model, valid_data, loss_fn, device, task, order):
    """
    Validate GeneralGNN without batching.
    
    Args:
        model (nn.Module): The GNN model.
        valid_data (tuple): (target_ids, support_1st, support_2nd, support_3rd, oracle_embeddings).
        loss_fn (nn.Module): The loss function.
        device (torch.device): Device to run the validation on.
        task (str): "user" or "item".
        order (str): "First-Order", "Second-Order", or "Third-Order".
    """
    model.eval()
    target_ids_valid, support_1st_valid, support_2nd_valid, support_3rd_valid, oracle_embeddings_valid = valid_data

    try:
        predicted_embeddings = model(
            target_ids_valid, support_1st_valid, support_2nd_valid, support_3rd_valid, task=task
        )
    except IndexError as e:
        print(f"IndexError during validation: {e}")
        print(f"Support_1st IDs: {support_1st_valid}")
        if support_2nd_valid is not None:
            print(f"Support_2nd IDs: {support_2nd_valid}")
        if support_3rd_valid is not None:
            print(f"Support_3rd IDs: {support_3rd_valid}")
        raise

    target = torch.ones(predicted_embeddings.size(0), device=device)
    loss = loss_fn(predicted_embeddings, oracle_embeddings_valid, target)

    print(f"Validation {order} {task} Task: Loss = {loss.item():.4f}")