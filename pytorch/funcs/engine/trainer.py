import torch
import time

from funcs.optim.loss import cal_loss
from funcs.optim.performance import cal_performance
from funcs.utils import *
    



def train_epoch(model,
                train_dataloader,
                val_dataloader,
                optimizer,
                device,
                epoch,
                log,
                tb_log,
                display_iter,
                tfboard_iter,
                val_iter,
                save_dir,
                cfg):
    total_loss = AverageMeter()
    total_acc = AverageMeter()

    max_iter = len(train_dataloader)
    # train loop
    for iteration, batch in enumerate(train_dataloader):
        tb_log.update_iter(max_iter * (epoch - 1) + iteration)

        if cfg.train_method == 'dist':
            inp = batch[0].to(device, non_blocking=True)
            tgt = batch[1].to(device, non_blocking=True)
        else:
            inp = batch[0].to(device)
            tgt = batch[1].to(device)

        optimizer.zero_grad()
        pred = model(inp, tgt)
        loss = cal_loss(pred, tgt)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grads_clip)
        optimizer.step_and_update_lr(epoch)

        total_loss.add(loss.item())

    return total_loss.avg, total_acc.avg
        

def validation(model, dataloader, device, cfg):
    model.eval()
    with torch.no_grad():
        for iteration, batch in enumerate(dataloader):
            if cfg.train_method == 'dist':
                inp = batch[0].to(device, non_blocking=True)
                tgt = batch[1].to(device, non_blocking=True)
            else:
                inp = batch[0].to(device)
                tgt = batch[1].to(device)
            pred = model(inp, tgt)
            loss = cal_loss(pred, tgt)
            acc = cal_performance(pred, tgt)
    model.train()

    return loss, acc

def fit(model,
        train_dataloader,
        val_dataloader,
        optimizer,
        device,
        start_epoch,
        end_epoch,
        log,
        tb_log,
        save_iter,
        display_iter,
        tfboard_iter,
        val_iter,
        save_dir,
        cfg):
    best_val=0
    best_epoch = 0
    best_iter = 0
    for epoch in range(start_epoch, end_epoch + 1):
        if cfg.train_method == 'dist':
            train_dataloader.sampler.set_epoch(epoch)
            val_dataloader.sampler.set_epoch(epoch)
        start_time = time.time()
        train_loss, train_acc = train_epoch(
            model=model,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            log=log,
            tb_log=tb_log,
            save_iter=save_iter,
            display_iter=display_iter,
            tfboard_iter=tfboard_iter,
            val_iter=val_iter,
            save_dir=save_dir,
            cfg=cfg,
            best_val=best_val,
            best_epoch=best_epoch,
            best_iter=best_iter
        )
        log.info('  - (Training)   loss: {loss: 8.5f}, accuracy: {acc:3.3f} %, time: {time:3.3f} min'
                .format(loss=train_loss, accu=100 * train_acc, time=(time.time() - start_time) / 60))

        # validation
        start_time = time.time()
        log.info('Starting per epoch validation ...')
        val_loss, val_acc = validation(
            model=model,
            val_dataloader=val_dataloader,
            device=device,
            cfg=cfg
        )
        log.info('  - (Validation)   loss: {loss: 8.5f}, accuracy: {acc:3.3f} %, time: {time:3.3f} min'
                .format(loss=val_loss, accu=100 * val_acc, time=(time.time() - start_time) / 60))
        
        # update best validation performance
        if val_acc > best_val:
            best_val = val_acc
            # save model
            if cfg.train_method != 'dist' or device == 0:
                log.info('Saving model : best_val in epoch:{epoch}')
                save_dict = {
                    'model': model.state_dict(),
                    'optimizer': optimizer,
                    'epoch': epoch,
                    'best_val': best_val
                }
                torch.save(save_dict, f'{save_dir}/best_val_epoch_{epoch:02d}.pth')
                log.info('Saved!')