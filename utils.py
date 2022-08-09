def adjust_learning_rate(optimizer, base_lr, epoch, lr_decay_epoch):
    lr = base_lr * (0.1**(epoch//lr_decay_epoch))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def prepare_path(path):
    if not os.path.isdir(path):
        os.makedirs(path, 0o777)
    return path


def remove_checkpoints(save_path):
    save_path = os.path.join(save_path, 'saved_models')
    saved_models = sorted(os.listdir(save_path))[:-1]
    for saved_model in saved_models:
        os.remove(os.path.join(save_path, saved_model))


def save_args(args):
    file_name = os.path.join(args.save_path, 'arguments.txt')
    with open(file_name, 'w') as f:
        for n, v in args.__dict__.items():
            f.write('{0}\n{1}\n\n'.format(n, v))


def save_checkpoint(state, is_best, save_path):
    save_path = os.path.join(save_path, 'saved_models')
    save_path = prepare_path(save_path)
    file_name = '%d.pth' % (state['epoch'])
    best_file_name = 'best_' + file_name
    file_path = os.path.join(save_path, file_name)
    best_file_path = os.path.join(save_path, best_file_name)
    torch.save(state, file_path)
    # Remove previous best model
    if is_best:
        saved_models = os.listdir(save_path)
        for saved_model in saved_models:
            if saved_model.startswith('best'):
                os.remove(os.path.join(save_path, saved_model))
        shutil.copyfile(file_path, best_file_path)


def send_data_dict_to_gpu(data, device):
    for k, v in data.items():
        if isinstance(v, torch.Tensor):
            data[k] = v.detach().to(device)
        elif isinstance(v, list):
            data_list = []
            for i in range(len(v)):
                data_list.append(v[i].detach().to(device))
            data[k] = data_list
    return data


class AverageMeter(object):
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
