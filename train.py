import os.path
import torch
import torch.nn as nn
from mmdet.models import cross_entropy

from diceloss import SoftDiceLoss
from eval import  eval_model

# scaler = torch.cuda.amp.GradScaler(init_scale=2056)

def train_model(model, logger, train_dataloader, eval_dataloader, device, working_dir, configuration, num_epochs=50, lr=0.0001):
    loss_weight = torch.tensor(configuration.class_weight).to(device)
    criterion = nn.CrossEntropyLoss(weight=loss_weight)
    soft_dice_loss = SoftDiceLoss(configuration.class_num)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    total_train_step = 0
    total_test_step = 0
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[num_epochs//2, num_epochs//1],  # 在epoch 30和80时调整
        gamma=0.1
    )

    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight)  # 使用Xavier初始化参数

    for epoch in range(num_epochs):
        model.train() #
        for images, labels in train_dataloader:
            images = images.to(device)
            labels = labels.to(device)
            # print("image.shape:",images.shape,", labels.shape:", labels.shape)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                outputs= model(images)
                # dice_loss = soft_dice_loss(outputs, labels) * 2
                labels = labels.squeeze(1).long()
                # print("output.shape:",outputs.shape)
                loss = criterion(outputs, labels)
                # loss = torch.add(dice_loss, cross_entropy_loss)
            # scaler.scale(loss).backward()
            # scaler.step(optimizer)
            # scaler.update()
            loss.backward()
            optimizer.step()
            total_train_step += 1

            if (total_train_step + 1) % 10 == 0:
                # print(f'Epoch [{epoch + 1}/{num_epochs}], train step: {total_train_step}, loss: {loss.item():.4f}, '
                #       f'dice loss: {dice_loss.item():.4f}, cross entropy loss: {cross_entropy_loss.item():.4f}')
                print(f'Epoch [{epoch + 1}/{num_epochs}], train step: {total_train_step}, loss: {loss.item():.4f}')
                # logger.add_scalar('train cross loss', cross_entropy_loss.item(), total_train_step)
                # logger.add_scalar('train dice loss', dice_loss.item(), total_train_step)
                logger.add_scalar('train loss', loss.item(), total_train_step)
                for name,param in model.named_parameters():
                    if param.grad is not None:
                        # print(f"{name} grad: {param.grad.norm().item()}")
                        logger.add_scalar(f'{name} grad', param.grad.norm().item(), total_train_step)

        scheduler.step()
        eval_model(model, logger, eval_dataloader, device, total_test_step)
        total_test_step += 1
        if total_test_step % 1 == 0:
            # model_name = configuration.model_name + str(total_test_step) + ".pth"
            model_name = "model_" + str(total_test_step) + ".pth"
            model_path = os.path.join(working_dir, model_name)
            torch.save(model.state_dict(), model_path)




