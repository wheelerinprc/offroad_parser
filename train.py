import torch
import torch.nn as nn

scaler = torch.cuda.amp.GradScaler()
def train_model(model, logger, dataloader, device, num_epochs=10, lr=0.001):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    total_train_step = 0
    for epoch in range(num_epochs):
        model.train() #
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                outputs= model(images)
                print("output.shape:",outputs.shape)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_train_step += 1
            if total_train_step % 100 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], train step: {total_train_step}, loss: {loss.item():.4f}')
                logger.add_scalar('train loss', loss.item(), total_train_step)


