import torch
import torch.nn as nn
from diceloss import SoftDiceLoss
soft_dice_loss = SoftDiceLoss(21, backprop=False)

def eval_model(model, logger, dataloader, device, test_step):
    total_test_loss = 0
    total_dice_loss = 0
    test_count = 0
    criterion = nn.CrossEntropyLoss()
    model.eval()

    with torch.no_grad():
        for image, label in dataloader:
            image = image.to(device)
            label = label.to(device)
            output = model(image)

            #calculate mIoU
            dice_loss = soft_dice_loss(output, label)
            total_dice_loss += dice_loss

            label = label.squeeze(1).long()
            loss = criterion(output, label)
            total_test_loss += loss.item()
            test_count += 1
        print(f"Loss on train data - CrossEntropy Loss: {total_test_loss / test_count}, Dice Loss: {total_dice_loss / test_count}")
        logger.add_scalar('Test loss', total_test_loss / test_count, test_step)
        logger.add_scalar('Test mIoU', total_dice_loss / test_count, test_step)