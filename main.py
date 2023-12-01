import sys
import time
import os.path as osp
import argparse
import torch
import torch.nn as nn
import torchreid
from collections import OrderedDict
from utils import MultiCropWrapper
from vision_transformer import DINOHead
from torchvision import models as torchvision_models
from icecream import ic
ic.disable()


from torchreid.utils import (
    Logger, check_isfile, set_random_seed, collect_env_info,
    resume_from_checkpoint, load_pretrained_weights, compute_model_complexity
)

from default_config import (
    imagedata_kwargs, optimizer_kwargs, videodata_kwargs, engine_run_kwargs,
    get_default_config, lr_scheduler_kwargs
)

def load_pretrained_dino_model(base_model, pretrained_model_path):
    # Check if the base model is OSNet
    if base_model == 'osnet_x1_0':
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
    saved_state_dict = torch.load(pretrained_model_path)
    new_state_dict = OrderedDict()
    for k, v in saved_state_dict["student"].items():
        name = k[7:]  # Remove "module." prefix if present
        new_state_dict[name] = v
    student.load_state_dict(new_state_dict)
    
    return student

def extract_vanilla_model(pretrained_model, arch, datamanager, unfreeze_last_n):
    if arch == 'osnet_x1_0':
        model = torchreid.models.build_model(
            name='osnet_x1_0',
            num_classes=datamanager.num_train_pids,
            pretrained=False,
            loss='softmax',
        )
        state_dict_model = model.state_dict()
        
    else:
        model = torchvision_models.__dict__[arch]()
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

def build_datamanager(cfg):
    if cfg.data.type == 'image':
        return torchreid.data.ImageDataManager(**imagedata_kwargs(cfg))
    else:
        return torchreid.data.VideoDataManager(**videodata_kwargs(cfg))


def build_engine(cfg, datamanager, model, optimizer, scheduler):
    if cfg.data.type == 'image':
        if cfg.loss.name == 'softmax':
            engine = torchreid.engine.ImageSoftmaxEngine(
                datamanager,
                model,
                optimizer=optimizer,
                scheduler=scheduler,
                use_gpu=cfg.use_gpu,
                label_smooth=cfg.loss.softmax.label_smooth
            )

        else:
            engine = torchreid.engine.ImageTripletEngine(
                datamanager,
                model,
                optimizer=optimizer,
                margin=cfg.loss.triplet.margin,
                weight_t=cfg.loss.triplet.weight_t,
                weight_x=cfg.loss.triplet.weight_x,
                scheduler=scheduler,
                use_gpu=cfg.use_gpu,
                label_smooth=cfg.loss.softmax.label_smooth
            )

    else:
        if cfg.loss.name == 'softmax':
            engine = torchreid.engine.VideoSoftmaxEngine(
                datamanager,
                model,
                optimizer=optimizer,
                scheduler=scheduler,
                use_gpu=cfg.use_gpu,
                label_smooth=cfg.loss.softmax.label_smooth,
                pooling_method=cfg.video.pooling_method
            )

        else:
            engine = torchreid.engine.VideoTripletEngine(
                datamanager,
                model,
                optimizer=optimizer,
                margin=cfg.loss.triplet.margin,
                weight_t=cfg.loss.triplet.weight_t,
                weight_x=cfg.loss.triplet.weight_x,
                scheduler=scheduler,
                use_gpu=cfg.use_gpu,
                label_smooth=cfg.loss.softmax.label_smooth
            )

    return engine


def reset_config(cfg, args):
    if args.root:
        cfg.data.root = args.root
    if args.sources:
        cfg.data.sources = args.sources
    if args.targets:
        cfg.data.targets = args.targets
    if args.transforms:
        cfg.data.transforms = args.transforms


def check_cfg(cfg):
    if cfg.loss.name == 'triplet' and cfg.loss.triplet.weight_x == 0:
        assert cfg.train.fixbase_epoch == 0, \
            'The output of classifier is not included in the computational graph'


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--config-file', type=str, default='', help='path to config file'
    )
    parser.add_argument(
        '-s',
        '--sources',
        type=str,
        nargs='+',
        help='source datasets (delimited by space)'
    )
    parser.add_argument(
        '-t',
        '--targets',
        type=str,
        nargs='+',
        help='target datasets (delimited by space)'
    )
    parser.add_argument(
        '--transforms', type=str, nargs='+', help='data augmentation'
    )
    parser.add_argument(
        '--root', type=str, default='./datasets', help='path to data root'
    )
    parser.add_argument(
        'opts',
        default=None,
        nargs=argparse.REMAINDER,
        help='Modify config options using the command-line'
    )
    parser.add_argument('--pretrained_model_path', type=str, default='./pretrained_models/osnet_100k.pth', help='Path to pretrained DINO model')
    args = parser.parse_args()

    cfg = get_default_config()
    cfg.use_gpu = torch.cuda.is_available()
    if args.config_file:
        cfg.merge_from_file(args.config_file)
    reset_config(cfg, args)
    cfg.merge_from_list(args.opts)
    set_random_seed(cfg.train.seed)
    check_cfg(cfg)

    log_name = 'test.log' if cfg.test.evaluate else 'train.log'
    log_name += time.strftime('-%Y-%m-%d-%H-%M-%S')
    sys.stdout = Logger(osp.join(cfg.data.save_dir, log_name))

    print('Show configuration\n{}\n'.format(cfg))
    print('Collecting env info ...')
    print('** System info **\n{}\n'.format(collect_env_info()))

    if cfg.use_gpu:
        torch.backends.cudnn.benchmark = True

    datamanager = build_datamanager(cfg)

    if cfg.model.load_weights:
        dino_model = load_pretrained_dino_model(cfg.model.name, args.pretrained_model_path)
        print(f"Loaded pretrained DINO model from {args.pretrained_model_path} with {cfg.model.name} architecture.")
        model = extract_vanilla_model(dino_model, cfg.model.name, datamanager, cfg.model.unfreeze_last_n)
        pretrained_vanilla_model_path = f"./pretrained_models/pretrained_vanilla_{cfg.model.name}.pth"
        torch.save(model.state_dict(), pretrained_vanilla_model_path) 
        cfg['model']['load_weights'] = pretrained_vanilla_model_path
    else:
        print('Building model: {}'.format(cfg.model.name))
        model = torchreid.models.build_model(
            name=cfg.model.name,
            num_classes=datamanager.num_train_pids,
            loss=cfg.loss.name,
            pretrained=cfg.model.pretrained,
            use_gpu=cfg.use_gpu
        )
        
    num_params, flops = compute_model_complexity(
        model, (1, 3, cfg.data.height, cfg.data.width)
    )
    print('Model complexity: params={:,} flops={:,}'.format(num_params, flops))

    if cfg.model.load_weights and check_isfile(cfg.model.load_weights):
        load_pretrained_weights(model, cfg.model.load_weights)

    if cfg.use_gpu:
        model = nn.DataParallel(model).cuda()

    optimizer = torchreid.optim.build_optimizer(model, **optimizer_kwargs(cfg))
    scheduler = torchreid.optim.build_lr_scheduler(
        optimizer, **lr_scheduler_kwargs(cfg)
    )

    if cfg.model.resume and check_isfile(cfg.model.resume):
        cfg.train.start_epoch = resume_from_checkpoint(
            cfg.model.resume, model, optimizer=optimizer, scheduler=scheduler
        )

    print(
        'Building {}-engine for {}-reid'.format(cfg.loss.name, cfg.data.type)
    )
    engine = build_engine(cfg, datamanager, model, optimizer, scheduler)
    engine.run(**engine_run_kwargs(cfg))


if __name__ == '__main__':
    main()
