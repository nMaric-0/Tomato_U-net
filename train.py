from Unet import UNet
import torch
from tqdm import tqdm
from torchmetrics.classification import BinaryJaccardIndex
from torchvision.utils import save_image
from puts import timestamp_seconds
from utils import init_logger, parse_args, prepr_data, pil_loader
import os



def train(opt, train_data, test_data, LOGGER):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = UNet().to(device)

    # Initilaize empty tensors for image batches and ground truth labels
    images = torch.empty(size=(opt.batchsize, 3, opt.imagesize[0], opt.imagesize[1]), dtype=torch.float32, device=device)
    mask = torch.empty(size=(opt.batchsize,1, opt.imagesize[0], opt.imagesize[1] ), dtype=torch.float32, device=device)

    # Initialize the loss function
    l_ce = torch.nn.BCEWithLogitsLoss()

    # Initialize the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.999))

    best_acc=0

    LOGGER.info(f"Batch size: {opt.batchsize}, epochs: {opt.epochs}")
    for epoch in range(opt.epochs):
        print("Training epoch: {}".format(epoch+1))
        # Set the CNN to train mode
        model.train()

        avg_loss = 0
        # Iterate through all batches
        for i_batch, input in tqdm(enumerate(train_data), total=len(train_data)):
        #for images, masks in tqdm(train_data):
            images.copy_(input[0]) # copy the batch in images
            mask.copy_(input[1]) # copy ground truth labels in gt

            optimizer.zero_grad() # Set all gradients to 0
            predictions = model(images) # Feedforward
            loss=l_ce(predictions, mask) # Calculate the error of the current batch
            avg_loss+=loss
            loss.backward() # Calculate gradients with backpropagation
            optimizer.step() # optimize weights for the next batch
        print("Epoch {}: average loss: {}". format(epoch+1, avg_loss/len(train_data)))

        print("Testing the model...")
        # Set the CNN to test mode
        model.eval()

        acc_sum = 0
        count = 0
        with torch.no_grad():
            for i_batch, input in tqdm(enumerate(test_data), total=len(test_data)):
                test_images=input[0].to(device)
                test_masks=input[1].to(device)

                predicted_masks = model(test_images)
                output = torch.sigmoid(predicted_masks)
                save_image(output, "output_image/img" + str(count)+ ".png")
                metric = BinaryJaccardIndex(threshold=0.6).to(device)
                acc_batch = metric(output, test_masks)
                acc_sum = acc_sum + acc_batch
                count = count + 1



        # Check what is the number of correctly classified samples
        print(acc_sum)
        acc = acc_sum/count# Calculate the total accuracy of the model
        print("Epoch {}: average accuracy: {}". format(epoch+1, acc))

        LOGGER.info(
                    f"[{epoch:03d}]: loss: {avg_loss/len(train_data)} - test acc: {acc}"
                )
        # save network weights when the accuracy is great than the best_acc
        if acc>best_acc:
            torch.save({'epoch': epoch, 'state_dict': model.state_dict()}, './output/UNet_weights_epoch_'+ epoch +'_.pth') #best network weights during traingin
            torch.save({'epoch': epoch, 'state_dict': model.state_dict()}, './output/UNet_weights.pth') #network weights saved for easier testing
            best_acc = acc
            LOGGER.info(f"Best Test Accuracy: {best_acc}")
            LOGGER.info(f"Best Model Saved: ./output/UNet_weights.pth")
        
        print("Average accuracy: {}     Best accuracy: {}".format(acc, best_acc))

def main():
    opt = parse_args()
    os.makedirs("output", exist_ok=True)
    os.makedirs("output_image", exist_ok=True)
    LOGGER = init_logger("output/" + f"{timestamp_seconds()}.log")

    train_data, test_data = prepr_data(opt, LOGGER)

    train(opt, train_data, test_data, LOGGER)
    LOGGER.info(f"Finished!!!")

if __name__ == "__main__":
    main()
