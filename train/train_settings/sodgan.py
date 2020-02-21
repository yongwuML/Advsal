import torch.nn as nn
import datetime
import os 
import torch
import torch.optim as optim
import torchvision.transforms
from torch.utils.data import DataLoader
from train.admin.environment import env_settings 
from train.dataset import ImageFolder, check_mkdir
from train.data import transforms
from train.model.sodgan import Net
from train import MultiGPU
from train.trainers import LTRTrainer


def run(settings):
    settings.description = 'SODGAN with default settings.'
    settings.train_dir = env_settings().msra10k_dir
    settings.ckpt = env_settings().workspace_dir
    settings.multi_gpu = False # multi-gpus training, we don't have done it for now, we will update.
    settings.iter_num = 5000 # Number of iters, 2000 for first training
    settings.batch_size = 8
    settings.diter_num = 10000 # Number of iters for Dcriminator, 0 for only train Generator, 10000 for second training.
    settings.last_iter = 2000 # 0 for first training, 2000 for second training
    settings.lr = 1e-3
    settings.lr_decay = 0.9
    settings.weight_decay = 5e-4
    settings.momentum = 0.9
    settings.snapshot = '2000' # None for first training, 2000 for second training
    settings.num_workers = 12                                    # Number of workers for image loading
    settings.normalize_mean = [0.485, 0.456, 0.406]             # Normalize mean (default pytorch ImageNet values)
    settings.normalize_std = [0.229, 0.224, 0.225]              # Normalize std (default pytorch ImageNet values)
    settings.log_path = os.path.join(settings.ckpt, str(datetime.datetime.now()) + '.txt')
    
    joint_transform = transforms.Compose([transforms.RandomCrop(300), transforms.RandomHorizontallyFlip(), transforms.RandomRotate(10)])
    img_transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), 
                                        torchvision.transforms.Normalize(mean=settings.normalize_mean, std=settings.normalize_std)])
    target_transform = torchvision.transforms.ToTensor()
    
    train_set = ImageFolder(settings.train_dir, joint_transform, img_transform, target_transform)
    train_loader = DataLoader(train_set, batch_size=settings.batch_size, num_workers=settings.num_workers, shuffle=True)
    objective = nn.BCELoss().cuda()
    
    net = Net().cuda().train()
    if settings.multi_gpu:
        net = MultiGPU(net, dim=1)

    optimizer = optim.SGD([
        {'params': [param for name, param in net.named_parameters() if name[-4:] == 'bias'],'lr': 2 * settings.lr},
        {'params': [param for name, param in net.named_parameters() if name[-4:] != 'bias'],'lr': settings.lr, 'weight_decay': settings.weight_decay}],
         momentum=settings.momentum)
    

    if len(settings.snapshot) > 0:
        print ('training resumes from ' + settings.snapshot)
        net.load_state_dict(torch.load(os.path.join(settings.ckpt, settings.snapshot + '.pth')))
        optimizer.load_state_dict(torch.load(os.path.join(settings.ckpt, settings.snapshot + '_optim.pth')))
        optimizer.param_groups[0]['lr'] = 2 * settings.lr
        optimizer.param_groups[1]['lr'] = settings.lr
        
    check_mkdir(settings.ckpt)
    open(settings.log_path, 'w').write(str(settings) + '\n\n')
    
    trainer = LTRTrainer(net, objective, optimizer, train_loader, settings, t=0)
    trainer.train(settings.diter_num) # 0 for first training
    
