import torch
import torch.nn as nn
import torch.optim as optim
import ast

def train_first_order_task(model, train_data, epochs, device, task="user"):
    optimizer = optim.Adagrad(model.parameters(), lr=model.learning_rate)
    loss_fn = nn.CosineEmbeddingLoss()
    model.to(device)

    if (task == "user"):
        target_ids_train = train_data["userid"].tolist()
    else:
        target_ids_train = train_data["itemid"].tolist()
    support_1st_train = train_data["1st_order"].tolist()
    oracle_embeddings_train = torch.tensor(train_data['oracle_embedding'].tolist(), dtype=torch.float32)

    # target_ids_train = target_ids_train.to(device)
    # support_1st_train = support_1st_train.to(device)
    # support_2nd = support_2nd.to(device)
    # support_3rd = support_3rd.to(device)
    # oracle_embeddings_train = oracle_embeddings_train.to(device)
    # target_ids_valid, support_1st_valid, _, _, oracle_embeddings_valid = valid_data

    for epoch in range(1, epochs + 1): 
        model.train()

        # 前向传播
        for i in range(0, len(target_ids_train)):
            predicted_embeddings = model(target_ids_train[i], support_1st_train[i], None, None, task=task)

        # 计算损失
        target = torch.ones(predicted_embeddings.size(0), device=device)
        loss = loss_fn(predicted_embeddings, oracle_embeddings_train, target)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch}/{epochs} - First-Order {task} Task: Train Loss = {loss.item():.4f}")

        # # 验证
        # validate_task(model, valid_data, loss_fn, device, task, "First-Order")

def train_second_order_task(model, train_data, epochs, device, task="user"):
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

    if (task == "user"):
        target_ids_train = train_data["userid"].tolist()
    else:
        target_ids_train = train_data["itemid"].tolist()
    support_1st_train = train_data["1st_order"].tolist()
    support_2nd_train = train_data["2nd_order"].tolist()
    oracle_embeddings_train = torch.tensor(train_data['oracle_embedding'].tolist(), dtype=torch.float32)
    # target_ids_valid, support_1st_valid, support_2nd_valid, _, oracle_embeddings_valid = valid_data

    # target_ids_train = target_ids_train.to(device)
    # support_1st_train = support_1st_train.to(device)
    # support_2nd_train = support_2nd_train.to(device)
    # support_3rd = support_3rd.to(device)
    oracle_embeddings_train = oracle_embeddings_train.to(device)

    for epoch in range(1, epochs + 1):
        model.train()

        # 前向传播
        for i in range(0, len(target_ids_train)):
            predicted_embeddings = model(target_ids_train[i], support_1st_train[i], support_2nd_train[i], None, task=task)

        # 计算损失
        target = torch.ones(predicted_embeddings.size(0), device=device)
        loss = loss_fn(predicted_embeddings, oracle_embeddings_train, target)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch}/{epochs} - Second-Order {task} Task: Train Loss = {loss.item():.4f}")

        # # 验证
        # validate_task(model, valid_data, loss_fn, device, task, "Second-Order")

def train_third_order_task(model, train_data, epochs, device, task="user"):
    optimizer = optim.Adagrad(model.parameters(), lr=model.learning_rate)
    loss_fn = nn.CosineEmbeddingLoss()
    model.to(device)

    if (task == "user"):
        target_ids_train = train_data["userid"].tolist()
    else:
        target_ids_train = train_data["itemid"].tolist()
    support_1st_train = train_data["1st_order"].tolist()
    support_2nd_train = train_data["2nd_order"].tolist()
    support_3rd_train = train_data["3rd_order"].tolist()
    oracle_embeddings_train = torch.tensor(train_data['oracle_embedding'].tolist(), dtype=torch.float32)
    # target_ids_valid, support_1st_valid, support_2nd_valid, support_3rd_valid, oracle_embeddings_valid = valid_data

    # target_ids_train = target_ids_train.to(device)
    # support_1st_train = support_1st_train.to(device)
    # support_2nd_train = support_2nd_train.to(device)
    # support_3rd_train = support_3rd_train.to(device)
    oracle_embeddings_train = oracle_embeddings_train.to(device)

    for epoch in range(1, epochs + 1):
        model.train()

        # 前向传播
        for i in range(0, len(target_ids_train)):
            predicted_embeddings = model(target_ids_train[i], support_1st_train[i], support_2nd_train[i], support_3rd_train[i], task=task)

        # 计算损失
        target = torch.ones(predicted_embeddings.size(0), device=device)
        loss = loss_fn(predicted_embeddings, oracle_embeddings_train, target)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch}/{epochs} - Third-Order {task} Task: Train Loss = {loss.item():.4f}")

        # # 验证
        # validate_task(model, valid_data, loss_fn, device, task, "Third-Order")

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