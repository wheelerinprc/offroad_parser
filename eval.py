import torch
import torch.nn as nn

def eval_model(model, logger, dataloader, device, test_step):
    total_test_loss = 0
    test_count = 0
    criterion = nn.CrossEntropyLoss()
    model.eval()

    with torch.no_grad():
        for image, label in dataloader:
            image = image.to(device)
            label = label.to(device)
            output = model(image)
            label = label.squeeze(1).long()
            loss = criterion(output, label)
            total_test_loss += loss.item()
            test_count += 1
        print("Loss on train data: {}".format(total_test_loss / test_count))
        logger.add_scalar('Test loss', total_test_loss / test_count, test_step)