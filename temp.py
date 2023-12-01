import random                                                       # for random number generation
import matplotlib.pyplot as plt                                     # for plotting 
import os                                                           # for file handling
import torch                                                        # for deep learning functionality
from prettytable import PrettyTable                                 # for table formatting
import matplotlib.image as mpimg                                    # for image plotting
import pathlib as pl                                                # for path handling.
import shutil                                                       # for file handling
from torchreid import models, utils                                 # for deep learning functionality
import torchreid                                                    # for deep learning functionality
import torchvision.models as models                                 # for deep learning functionality
import torch.nn as nn                                               # for deep learning functionality
import sys                                                          # for system functionality
from vision_transformer import DINOHead                             # for deep learning functionality  
from torchvision import models as torchvision_models                # for deep learning functionality
from utils import MultiCropWrapper                                  # for deep learning functionality
from collections import OrderedDict                                 # for deep learning functionality   
from torchsummary import summary                                    # for model summary    
import argparse                                                     # for command line argument parsing     
import wandb                                                        # for logging training runs   
from icecream import ic                                             # for debugging  
ic.disable()                                                        # disable icecream debugging                 

def load_pretrained_dino_model(base_model, pretrained_model_path, device):
    # Check if the base model is OSNet
    if base_model == 'osnet':
        student = torchreid.models.build_model(
            name='osnet_x1_0',
            num_classes=0,  # Assuming we don't need the classifier for feature extraction
            pretrained=False
        )
        embed_dim = student.classifier.in_features
        student.classifier = nn.Identity()
    else:
        # For other models (like ResNet), use torchvision models
        student = torchvision_models.__dict__[base_model]()
        embed_dim = student.fc.weight.shape[1]
        student.fc = nn.Identity()  # Replace the fully connected layer with Identity
    ic(student)
    
    # Wrap the student model with MultiCropWrapper and DINOHead
    student = MultiCropWrapper(student, DINOHead(
        embed_dim,
        65536,  # Output dimension for DINOHead
        use_bn=False,
        norm_last_layer=True,
    ))
    ic(student)

    # Load the pretrained model weights
    saved_state_dict = torch.load(pretrained_model_path, map_location=device)
    new_state_dict = OrderedDict()
    for k, v in saved_state_dict["student"].items():
        name = k[7:]  # Remove "module." prefix if present
        new_state_dict[name] = v
    student.load_state_dict(new_state_dict)
    
    return student

def extract_vanilla_model(pretrained_model, model_type, datamanager, unfreeze_last_n):
    if model_type == 'osnet':
        model = torchreid.models.build_model(
            name='osnet_x1_0',
            num_classes=datamanager.num_train_pids,
            pretrained=False,
            loss='softmax',
        )
        state_dict_model = model.state_dict()
        
    else:
        model = torchvision_models.__dict__[model_type]()
        state_dict_model = model.state_dict()
        
    # Copy the weights from the pretrained model
    state_dict_pretrained = pretrained_model.state_dict()
    for name, param in state_dict_model.items():
        if name in state_dict_pretrained:
            state_dict_model[name] = state_dict_pretrained[name]

    model.load_state_dict(state_dict_model)
    
    # Handle unfreezing of layers
    if unfreeze_last_n == -1:
        for param in model.parameters():        # Make all layers trainable
            param.requires_grad = True
    elif unfreeze_last_n == 0:
        for param in model.parameters():        # Make all layers frozen
            param.requires_grad = False
    else:   
        for param in model.parameters():
            param.requires_grad = False         # Unfreeze the last n layers

        children = list(model.children())
        num_children = len(children)
        for i, child in enumerate(children):
            if i >= num_children - unfreeze_last_n:
                for param in child.parameters():
                    param.requires_grad = True
   
    return model

def create_base_models(base_model, datamanager, device):
    if base_model == 'osnet':
        model = torchreid.models.build_model(
            name='osnet_x1_0',
            num_classes=datamanager.num_train_pids,
            pretrained=False,
            loss='softmax',
        )
    else:
        model = torchvision_models.__dict__[base_model]()

    return model.to(device)
    
def setup_datamanager(dataset_dir, args):
    datamanager = torchreid.data.ImageDataManager(
        root=dataset_dir,
        sources='market1501',
        targets='market1501',
        height=256,
        width=128,
        batch_size_train=args.batch_size_train,
        batch_size_test=args.batch_size_test,
        transforms=['random_flip', 'random_crop'],
        combineall=False,
    )
    return datamanager

class WandbLogger(object):
    def __init__(self, args):
        wandb.init(name=args.wanb_name)
        wandb.config.update(args)

    def log_metrics(self, epoch, train_metrics, val_metrics):
        metrics = {
            'epoch': epoch,
            'train_loss': train_metrics['loss'],
            'train_acc': train_metrics['acc'],
            'val_mAP': val_metrics['mAP'],
            'val_rank1': val_metrics['rank1'],
            'val_rank5': val_metrics['rank5'],
            'val_rank10': val_metrics['rank10'],
            'val_rank20': val_metrics['rank20'],
        }
        wandb.log(metrics)
        
def main():
    parser = argparse.ArgumentParser(description='Fine-tune Models')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--pretrained_model_path', type=str, default='./pretrained_models/osnet_100k.pth', help='Path to pretrained DINO model')
    parser.add_argument('--pretrained', action='store_true', help='Use pretrained model')
    parser.add_argument('--unfreeze_last_n', type=int, default=-1, help='Number of last layers to unfreeze')
    parser.add_argument('--batch_size_train', type=int, default=32, help='Training batch size for ImageDataManager')
    parser.add_argument('--batch_size_test', type=int, default=100, help='Testing batch size for ImageDataManager')
    parser.add_argument('--random_seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--arch', type=str, default='osnet', help='Architecture for DINO model')
    parser.add_argument('--optimizer', type=str, default='adam', help='Optimizer for training')
    parser.add_argument('--lr_scheduler', type=str, default='single_step', help='Learning rate scheduler for training')
    parser.add_argument('--eval_freq', type=int, default=1, help='Frequency of evaluation')
    parser.add_argument('--stepsize', type=int, default=20, help='Step size for learning rate scheduler')
    parser.add_argument('--use_wandb', action='store_true', help='Use wandb for logging.')
    parser.add_argument('--wandb_name', type=str, default='finetune model', help='Name of wandb project')
    args = parser.parse_args()
    
    # Logging
    if args.use_wandb:
        wandb_logger = WandbLogger(args)
        
    # Environment setup
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    table = PrettyTable()
    table.field_names = ["CUDA", "GPU", "Total Memory (MB)"]
    if torch.cuda.is_available():
        num_devices = torch.cuda.device_count()
        for i in range(num_devices):
            total_memory = torch.cuda.get_device_properties(i).total_memory / (1024 * 1024)
            table.add_row([f"CUDA:{i}", torch.cuda.get_device_name(i), f"{total_memory:.2f}"])
            print(table)
    else:
        print("No GPU available.")
    
    # Set random seed
    random_seed = args.random_seed
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    if device == 'cuda':
        torch.cuda.manual_seed(random_seed)

    # Set dataset directories
    dataset_dir = pl.Path("./datasets")                                          # set dataset directory
    pa100k_dir = dataset_dir / "PA-100K/imgs"                                   # set pa100k imgs directory
    market1501_dir = dataset_dir / "Market-1501-v15.09.15/"                     # set pretrained models directory
    logs_dir ='logs'                                                            # set logs directory
    
    # Setup Data Manager
    datamanager = setup_datamanager(dataset_dir, args)
    
    # Load and prepare models
    if args.pretrained:
        dino_model = load_pretrained_dino_model(args.arch, args.pretrained_model_path, device).to(device)
        print(f"Loaded pretrained DINO model from {args.pretrained_model_path} with {args.arch} architecture.")
        model = extract_vanilla_model(dino_model, args.arch, datamanager, args.unfreeze_last_n).to(device)
        # save model
        torch.save(model.state_dict(), f"./pretrained_models/pretrained_{args.arch}.pth")
        quit()
    else:
        model = create_base_models(args.arch, datamanager, device)
        print(f"Training {args.arch} model from scratch.")

    # Print model summary
    print("Model Summary:")
    summary(model, (3, 256, 128))
    
    # Model Complexity
    print("Model Complexity:")
    torchreid.utils.compute_model_complexity(model, (1, 3, 256, 128), verbose=True, only_conv_linear=False)
    
    # Optimizer, Scheduler, Engine
    optimizer = torchreid.optim.build_optimizer(model, optim=args.optimizer, lr=args.learning_rate)
    scheduler = torchreid.optim.build_lr_scheduler(optimizer, lr_scheduler=args.lr_scheduler, stepsize=args.stepsize)
    engine = torchreid.engine.ImageSoftmaxEngine(datamanager, model, optimizer=optimizer, scheduler=scheduler)

    # Run training
    loss_summary = engine.run(max_epoch=args.epochs, save_dir=logs_dir, eval_freq=args.eval_freq, print_freq=1, test_only=False)
    print(loss_summary)
    # train_metrics = engine.run(max_epoch=args.epochs, save_dir=logs_dir, eval_freq=args.eval_freq, print_freq=1, test_only=False)
    # print(train_metrics)
    # wandb_logger.log_metrics(args.epochs, train_metrics['loss'], train_metrics['acc'])
    
    # val_metrics = engine.run(max_epoch=1, save_dir=logs_dir, eval_freq=1, print_freq=1, test_only=True)
    # print(val_metrics)
    # wandb_logger.log_metrics(args.epochs, val_metrics['mAP'], val_metrics['rank1'], val_metrics['rank5'], val_metrics['rank10'], val_metrics['rank20'])
    
if __name__ == "__main__":
    main()

