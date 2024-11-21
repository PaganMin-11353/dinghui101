import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm


def train_first_order_task(model, train_loader, valid_loader, epochs, device, task="user"):
    """
    Train GeneralGNN for first-order tasks (user or item).
    """
    optimizer = optim.Adagrad(model.parameters(), lr=model.learning_rate)
    loss_fn = nn.CosineEmbeddingLoss()
    model.to(device)

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} - Training First-Order {task} Task"):
            target_ids, support_1st, _, _, oracle_embeddings = batch
            target_ids = target_ids.to(device)
            support_1st = support_1st.to(device)
            oracle_embeddings = oracle_embeddings.to(device)

            predicted_embeddings = model(
                target_ids, support_1st, None, None, task=task
            )

            target = torch.ones(predicted_embeddings.size(0), device=device)
            loss = loss_fn(predicted_embeddings, oracle_embeddings, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{epochs} - First-Order {task} Task: Train Loss = {avg_train_loss:.4f}")

        validate_task(model, valid_loader, loss_fn, device, task, "First-Order")


def train_second_order_task(model, train_loader, valid_loader, epochs, device, task="user"):
    """
    Train GeneralGNN for second-order tasks (user or item).
    """
    optimizer = optim.Adagrad(model.parameters(), lr=model.learning_rate)
    loss_fn = nn.CosineEmbeddingLoss()
    model.to(device)

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} - Training Second-Order {task} Task"):
            target_ids, support_1st, support_2nd, _, oracle_embeddings = batch
            target_ids = target_ids.to(device)
            support_1st = support_1st.to(device)
            support_2nd = support_2nd.to(device)
            oracle_embeddings = oracle_embeddings.to(device)

            predicted_embeddings = model(
                target_ids, support_1st, support_2nd, None, task=task
            )

            target = torch.ones(predicted_embeddings.size(0), device=device)
            loss = loss_fn(predicted_embeddings, oracle_embeddings, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{epochs} - Second-Order {task} Task: Train Loss = {avg_train_loss:.4f}")

        validate_task(model, valid_loader, loss_fn, device, task, "Second-Order")


def train_third_order_task(model, train_loader, valid_loader, epochs, device, task="user"):
    """
    Train GeneralGNN for third-order tasks (user or item).
    """
    optimizer = optim.Adagrad(model.parameters(), lr=model.learning_rate)
    loss_fn = nn.CosineEmbeddingLoss()
    model.to(device)

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} - Training Third-Order {task} Task"):
            target_ids, support_1st, support_2nd, support_3rd, oracle_embeddings = batch
            target_ids = target_ids.to(device)
            support_1st = support_1st.to(device)
            support_2nd = support_2nd.to(device)
            support_3rd = support_3rd.to(device)
            oracle_embeddings = oracle_embeddings.to(device)

            predicted_embeddings = model(
                target_ids, support_1st, support_2nd, support_3rd, task=task
            )

            target = torch.ones(predicted_embeddings.size(0), device=device)
            loss = loss_fn(predicted_embeddings, oracle_embeddings, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{epochs} - Third-Order {task} Task: Train Loss = {avg_train_loss:.4f}")

        validate_task(model, valid_loader, loss_fn, device, task, "Third-Order")


def validate_task(model, valid_loader, loss_fn, device, task, order):
    """
    Validation logic for tasks.
    """
    model.eval()
    valid_loss = 0.0
    with torch.no_grad():
        for batch in valid_loader:
            target_ids, support_1st, support_2nd, support_3rd, oracle_embeddings = batch
            target_ids = target_ids.to(device)
            support_1st = support_1st.to(device)
            support_2nd = support_2nd.to(device) if support_2nd is not None else None
            support_3rd = support_3rd.to(device) if support_3rd is not None else None
            oracle_embeddings = oracle_embeddings.to(device)

            try:
                predicted_embeddings = model(target_ids, support_1st, support_2nd, support_3rd, task=task)
            except IndexError as e:
                print(f"IndexError during validation: {e}")
                print(f"Support_1st IDs: {support_1st}")
                if support_2nd is not None:
                    print(f"Support_2nd IDs: {support_2nd}")
                if support_3rd is not None:
                    print(f"Support_3rd IDs: {support_3rd}")
                raise

            target = torch.ones(predicted_embeddings.size(0), device=device)
            loss = loss_fn(predicted_embeddings, oracle_embeddings, target)
            valid_loss += loss.item()

    avg_valid_loss = valid_loss / len(valid_loader)
    print(f"Validation {order} {task} Task: Loss = {avg_valid_loss:.4f}")
