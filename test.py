from Unet import UNet
import torch
from tqdm import tqdm
from torchmetrics.classification import BinaryJaccardIndex
from torchvision.utils import save_image
from puts import timestamp_seconds
from utils import init_logger, parse_args, prepr_data, pil_loader
import os


def test(opt, test_data, LOGGER, model_path):

    device = "cuda:0"
    model = UNet().to(device)

    print("Testing the model...")
    # Set the CNN to test mode
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    LOGGER.info("Model loaded")

    acc_sum = 0
    count = 0
    with torch.no_grad():
        for i_batch, input in tqdm(enumerate(test_data), total=len(test_data)):
            test_images=input[0].to(device)
            test_masks=input[1].to(device)

            predicted_masks = model(test_images)
            output = torch.sigmoid(predicted_masks)
            image_print = torch.cat((test_images,output),1)
            save_image(image_print, "output_image/img" + str(count)+ ".png")
            save_image(output, "output_image/img" + str(count)+ "_Out.png")
            save_image(test_images, "output_image/img" + str(count)+ "_Org.png")
            save_image(test_masks, "output_image/img" + str(count)+ "_Mask.png")
            metric = BinaryJaccardIndex(threshold=0.5).to(device)
            acc_batch = metric(output, test_masks)
            acc_sum = acc_sum + acc_batch
            count = count + 1


    # Check what is the number of correctly classified samples
    print(acc_sum)
    acc = acc_sum/count# Calculate the total accuracy of the model

    LOGGER.info(
                f"[- test acc: {acc}]"
            )
    

def main():
    opt = parse_args()
    os.makedirs("output", exist_ok=True)
    os.makedirs("output_image", exist_ok=True)
    LOGGER = init_logger("output/" + f"{timestamp_seconds()}.log")
    _, test_data = prepr_data(opt, LOGGER)

    model_path = "./output/UNet_weights.pth"

    test(opt, test_data, LOGGER, model_path)

if __name__ == "__main__":
    main()