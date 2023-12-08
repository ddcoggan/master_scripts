import torch.optim as optim

def get_optimizer(model,T):

    if T.optimizer_name == 'SGD':
        optimizer = optim.SGD(params=model.parameters(),
                              lr=T.learning_rate,
                              momentum=T.momentum,
                              weight_decay=T.weight_decay)
    elif T.optimizer_name == 'Adam':
        optimizer = optim.Adam(params=model.parameters(),
                               lr=T.learning_rate,
                               weight_decay=T.weight_decay)
    elif T.optimizer_name == 'Lion':
        from lion_pytorch import Lion
        optimizer = Lion(params=model.parameters(),
                         lr=T.learning_rate,
                         weight_decay=T.weight_decay)

    optimizer.param_groups[0]['initial_lr'] = T.learning_rate

    return optimizer


def get_scheduler(optimizer, T):

    if T.scheduler == 'StepLR':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=T.step_size, last_epoch=T.checkpoint)
    if T.scheduler == 'ReduceLROnPlateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 'min', patience=T.patience)

    return scheduler



