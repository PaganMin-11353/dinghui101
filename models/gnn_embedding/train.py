import torch
from torch.utils.data import DataLoader
from GeneralGNN import GeneralGNN
from train_helper import (
    train_first_order_task,
    train_second_order_task,
    train_third_order_task,
)
from dataset import UserDataset, ItemDataset  # Define your dataset classes
from settings import Settings  # A settings file or object for configurations

def main():
    # Load settings
    settings = Settings()

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load datasets
    train_user_dataset = UserDataset(settings.oracle_training_file_user_task)
    valid_user_dataset = UserDataset(settings.oracle_valid_file_user_task)

    train_item_dataset = ItemDataset(settings.oracle_training_file_item_task)
    valid_item_dataset = ItemDataset(settings.oracle_valid_file_item_task)

    # Create DataLoaders
    train_user_loader = DataLoader(train_user_dataset, batch_size=settings.batch_size, shuffle=True)
    valid_user_loader = DataLoader(valid_user_dataset, batch_size=settings.batch_size, shuffle=False)

    train_item_loader = DataLoader(train_item_dataset, batch_size=settings.batch_size, shuffle=True)
    valid_item_loader = DataLoader(valid_item_dataset, batch_size=settings.batch_size, shuffle=False)

    # Instantiate the model
    model = GeneralGNN(name="GraphSAGE", settings=settings)

    # Define training parameters
    num_epochs = settings.epochs

    # Train for user tasks
    print("Training user tasks...")
    train_first_order_task(
        model=model,
        train_loader=train_user_loader,
        valid_loader=valid_user_loader,
        epochs=num_epochs,
        device=device,
        task="user",
    )

    train_second_order_task(
        model=model,
        train_loader=train_user_loader,
        valid_loader=valid_user_loader,
        epochs=num_epochs,
        device=device,
        task="user",
    )

    train_third_order_task(
        model=model,
        train_loader=train_user_loader,
        valid_loader=valid_user_loader,
        epochs=num_epochs,
        device=device,
        task="user",
    )

    # Train for item tasks
    print("Training item tasks...")
    train_first_order_task(
        model=model,
        train_loader=train_item_loader,
        valid_loader=valid_item_loader,
        epochs=num_epochs,
        device=device,
        task="item",
    )

    train_second_order_task(
        model=model,
        train_loader=train_item_loader,
        valid_loader=valid_item_loader,
        epochs=num_epochs,
        device=device,
        task="item",
    )

    train_third_order_task(
        model=model,
        train_loader=train_item_loader,
        valid_loader=valid_item_loader,
        epochs=num_epochs,
        device=device,
        task="item",
    )

    print("Training completed.")

if __name__ == "__main__":
    main()
