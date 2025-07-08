import os.path
import torch
import torch.nn as nn
from eval import  eval_model

scaler = torch.cuda.amp.GradScaler()
def train_model(model, logger, train_dataloader, eval_dataloader, device, working_dir, model_name, num_epochs=50, lr=0.001):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    total_train_step = 0
    total_test_step = 0
    for epoch in range(num_epochs):
        model.train() #
        for images, labels in train_dataloader:
            images = images.to(device)
            labels = labels.to(device)
            # print("image.shape:",images.shape,", labels.shape:", labels.shape)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                outputs= model(images)
                labels = labels.squeeze(1).long()
                # print("output.shape:",outputs.shape)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_train_step += 1
            if (total_train_step + 1) % 10 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], train step: {total_train_step}, loss: {loss.item():.4f}')
                logger.add_scalar('train loss', loss.item(), total_train_step)
        eval_model(model, logger, eval_dataloader, device, total_test_step)
        model_path = os.path.join(working_dir, model_name)
        if (total_test_step + 1) % 10 == 0:
            torch.save(model.state_dict(), model_path)
        total_test_step += 1




