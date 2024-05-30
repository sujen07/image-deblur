import os
import torch
from blur_models import *
from load_data import load_data
import time
from argparse import ArgumentParser
import wandb



default_batch_size = 1
default_learning_rate = 0.0001
default_num_epochs = 1000
default_model_name = 'model.pth'
default_out_dir = 'out'
default_wandb_log = False
defualt_init_resume = False



# Check for command line args for hyperparameters
def parse_args():
    parser = ArgumentParser()
    # Use the default values defined above
    parser.add_argument("--batch_size", type=int, default=default_batch_size)
    parser.add_argument("--num_epochs", type=int, default=default_num_epochs)
    parser.add_argument("--model_name", type=str, default=default_model_name)
    parser.add_argument("--out_dir", type=str, default=default_out_dir)
    parser.add_argument("--wandb_log", type=bool, default=default_wandb_log)
    parser.add_argument("--lr", type=bool, default=default_learning_rate)
    parser.add_argument("--resume", type=bool, default=defualt_init_resume)
    return parser.parse_args()


def get_val_loss(model, val_loader, loss, device):
    total_loss = 0
    model.eval()
    with torch.no_grad():
        for val_lr, val_hr in val_loader:
            output = model(val_lr.to(device))
            total_loss += loss(output, val_hr.to(device)).item()
    total_loss = total_loss / len(val_loader)
    return total_loss




def train(batch_size, learning_rate, num_epochs, model_path, wandb_log, resume):
    print(f"Training configuration:\n"
          f"Model Path: {model_path}\n"
          f"Epochs: {num_epochs}\n"
          f"Batch Size: {batch_size}\n"
         )

    train_dir = 'data/train'
    val_dir = 'data/validation'
    train_loader, val_loader = load_data(batch_size, train_dir, val_dir)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device('cpu')
    print(f'Using Device: {device}')

    model = ALGNet()
    if resume:
        model.load_state_dict(torch.load(model_path))
        
    model = model.to(device)
    loss = DeblurLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


    # Training loop
    for epoch in range(num_epochs):
        start = time.time()
        running_loss = 0.0
        batch_idx = 1
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            deblur_loss = loss(outputs, targets)
            deblur_loss.backward()
            optimizer.step()

            running_loss += deblur_loss.item() * inputs.size(0)
            print(f"Epoch {epoch+1}/{num_epochs}, Batch [{batch_idx} / {len(train_loader)}] Loss: {deblur_loss.item():.4f}")
            batch_idx += 1
            torch.save(model.state_dict(), model_path)
        end = time.time()

        if epoch % 20 == 0:
            val_loss = get_val_loss(model, val_loader, loss, device)
            train_loss = running_loss / num_epochs
            one_epoch_time = end - start
            if wandb_log:
                wandb.log({"Validation Loss": val_loss, "Training Loss": train_loss})
            print(f'Train Loss: {train_loss}, Val: {val_loss}, Time for Epoch {epoch+1}: {one_epoch_time} Seconds')
            
        
    


if __name__ == '__main__':
    args = parse_args()

    if not os.path.exists(args.out_dir):
        os.mkdir(args.out_dir)
    model_path = os.path.join(args.out_dir, args.model_name)
    if args.wandb_log:
        wandb.login()
        config = vars(args)
        run = wandb.init(
            project="image-deblur",
            # Track hyperparameters and run metadata
            config=config
        )
    train(args.batch_size, args.lr, args.num_epochs, model_path, args.wandb_log, args.resume)