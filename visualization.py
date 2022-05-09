import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from matplotlib import pyplot as plt
import copy

transform_train = transforms.Compose([
    # transforms.RandomCrop(32, padding=4),
    # transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
device = 'cuda' if torch.cuda.is_available() else 'cpu'
trainset = torchvision.datasets.CIFAR100(
    root='./baseline/data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=4, shuffle=True, num_workers=2)
alpha = 50
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
setup_seed(12)

# cutout
def rand_bbox(size, lam):
    s0 = size[0]
    s1 = size[1]
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int32(W * cut_rat)
    cut_h = np.int32(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    mask = np.ones((s0, s1, W, H), np.float32)
    mask[:, :, bbx1: bbx2, bby1: bby2] = 0.
    mask = torch.from_numpy(mask)
    return mask

def cutout_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    #y_a, y_b = y, y[index]
    mask = rand_bbox(x.size(), lam)

    mask = mask.to(device)
    x = x * mask
    #lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
    
    return x, y

# mixup
def mixup_data(x, y, alpha=50, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)
    
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def rand_bbox_cutmix(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int32(W * cut_rat)
    cut_h = np.int32(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

def cutmix_data(x, y, alpha=50, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    y_a, y_b = y, y[index]
    bbx1, bby1, bbx2, bby2 = rand_bbox_cutmix(x.size(), lam)
    x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
    
    return x, y_a, y_b, lam



for batch_idx, (inputs, targets) in enumerate(trainloader):
    inputs, targets = inputs.to(device), targets.to(device)

    
    inputs1, targets1 = copy.deepcopy(inputs), copy.deepcopy(targets)
    inputs2, targets2 = copy.deepcopy(inputs), copy.deepcopy(targets)
    inputs3, targets3 = copy.deepcopy(inputs), copy.deepcopy(targets)
    inputs_cutout, targets_cutout = cutout_data(inputs1, targets1, alpha)
    inputs_mixup, targets_a_mixup, targets_b_mixup, lam_mixup = mixup_data(inputs2, targets2, alpha)
    inputs_cutmix, targets_a_cutmix, targets_b_cutmix, lam_cutmix = cutmix_data(inputs3, targets3, alpha)
    print("inputs",inputs.shape, targets_a_mixup, targets_b_mixup, lam_mixup )
    inputs, inputs_cutout, inputs_mixup, inputs_cutmix = inputs.cpu().numpy(), inputs_cutout.cpu().numpy(), inputs_mixup.cpu().numpy(), inputs_cutmix.cpu().numpy()

        
    # create figure
    fig = plt.figure(figsize=(20, 14))
    
    # setting values to rows and column variables
    rows = 4
    columns = 3
    
    plot_index = 1
    for i in range(3):
        # Adds a subplot
        fig.add_subplot(rows, columns, plot_index)
        plot_index += 1
        # showing image
        plt.imshow(inputs[i].transpose(1,2,0))
        plt.axis('off')
        plt.title("Origin label "+str(targets[i].item()))
    for i in range(3):
        # Adds a subplot
        fig.add_subplot(rows, columns, plot_index)
        plot_index += 1
        # showing image
        plt.imshow(inputs_cutout[i].transpose(1,2,0))
        plt.axis('off')
        plt.title("Cut out label "+str(targets_cutout[i].item()))
    for i in range(3):
        # Adds a subplot
        fig.add_subplot(rows, columns, plot_index)
        plot_index += 1
        # showing image
        plt.imshow(inputs_mixup[i].transpose(1,2,0))
        plt.axis('off')
        plt.title("Mix up label "+str(targets_a_mixup[i].item())+' '+str(targets_b_mixup[i].item())+" ratio %.2f" %lam_cutmix)
    for i in range(3):
        # Adds a subplot
        fig.add_subplot(rows, columns, plot_index)
        plot_index += 1
        # showing image
        plt.imshow(inputs_cutmix[i].transpose(1,2,0))
        plt.axis('off')
        plt.title("Cut mix label "+str(targets_a_cutmix[i].item())+' '+str(targets_b_cutmix[i].item())+" ratio %.2f" %lam_cutmix)
        plt.savefig("example.png")
    break
