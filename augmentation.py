import albumentations as A
import cv2
import os
from tqdm import tqdm


def create_augmentated_images(folder1, folder2):
    """Script to crate augmentated images and enrich the datased"""

    transformFlip = A.Compose([
        A.RandomResizedCrop(always_apply=False, p=1.0, height=512, width=384, scale=(0.8, 1.0), ratio=(0.75, 1.0), interpolation=0),
        A.HorizontalFlip(always_apply=True)
    ])

    transformFlip_Vertical = A.Compose([
        A.HorizontalFlip(always_apply=True),
        A.RandomResizedCrop(always_apply=False, p=1.0, height=512, width=384, scale=(0.8, 1.0), ratio=(0.75, 1.0), interpolation=0)
    ])

    transformCLAHE = A.Compose([
        A.CLAHE(always_apply=True, clip_limit=(1, 58), tile_grid_size=(4, 4))
    ])

    transform_Elastic = A.Compose([
        A.ElasticTransform(always_apply=False, p=1.0, alpha=4.08, sigma=50.0, alpha_affine=50.0, interpolation=0, border_mode=3, value=(0, 0, 0), mask_value=None, approximate=False, same_dxdy=False)
    ])

    transformRandomBrightnessContrast = A.Compose([
        A.RandomBrightnessContrast(always_apply=False, p=1.0, brightness_limit=(-0.36, 0.33), contrast_limit=(-0.51, 0.5), brightness_by_max=True),
        A.RandomGamma(always_apply=False, p=1.0, gamma_limit=(30, 158))
    ])

    transformRandomGridShuffle = A.Compose([
        A.RandomGridShuffle(always_apply=False, p=1.0, grid=(3, 3))
    ])

    transformRandomCrop = A.Compose([
        A.RandomSizedCrop(always_apply=False, p=1.0, min_max_height=(106, 500), height=512, width=384, w2h_ratio=0.75, interpolation=0)
    ])
    for filename in tqdm(os.listdir(folder1)):
        image = cv2.imread(os.path.join(folder1,filename))
        mask = cv2.imread(os.path.join(folder2,filename))
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if image.shape[0]<image.shape[1]:
            image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        if mask.shape[0]<mask.shape[1]:
            mask = cv2.rotate(mask, cv2.ROTATE_90_COUNTERCLOCKWISE)



        transformed = transformFlip(image = image, mask = mask)
        transformed_image = transformed['image']
        transformed_mask = transformed['mask']
        os.makedirs("laboro_tomato/train/images/Flip_horizontal", exist_ok=True)
        os.makedirs("laboro_tomato/train/masks/Flip_horizontal", exist_ok=True)
        cv2.imwrite("laboro_tomato/train/images/Flip_horizontal/FlipHorizontal_"+filename, transformed_image)
        cv2.imwrite("laboro_tomato/train/masks/Flip_horizontal/FlipHorizontal_"+filename, transformed_mask)

        transformed = transformFlip_Vertical(image = image, mask = mask)
        transformed_image = transformed['image']
        transformed_mask = transformed['mask']
        os.makedirs("laboro_tomato/train/images/Flip_vertical", exist_ok=True)
        os.makedirs("laboro_tomato/train/masks/Flip_vertical", exist_ok=True)
        cv2.imwrite("laboro_tomato/train/images/Flip_vertical/Flip_vertical_"+filename, transformed_image)
        cv2.imwrite("laboro_tomato/train/masks/Flip_vertical/Flip_vertical_"+filename, transformed_mask)

        transformed = transformCLAHE(image = image, mask = mask)
        transformed_image = transformed['image']
        transformed_mask = transformed['mask']
        os.makedirs("laboro_tomato/train/images/CLAHE", exist_ok=True)
        os.makedirs("laboro_tomato/train/masks/CLAHE", exist_ok=True)
        cv2.imwrite("laboro_tomato/train/images/CLAHE/CLAHE_"+filename, transformed_image)
        cv2.imwrite("laboro_tomato/train/masks/CLAHE/CLAHE_"+filename, transformed_mask)

        transformed = transform_Elastic(image = image, mask = mask)
        transformed_image = transformed['image']
        transformed_mask = transformed['mask']
        os.makedirs("laboro_tomato/train/images/Elastic_transform", exist_ok=True)
        os.makedirs("laboro_tomato/train/masks/Elastic_transform", exist_ok=True)
        cv2.imwrite("laboro_tomato/train/images/Elastic_transform/Elastic_transform_"+filename, transformed_image)
        cv2.imwrite("laboro_tomato/train/masks/Elastic_transform/Elastic_transform_"+filename, transformed_mask)

        transformed = transform_Elastic(image = image, mask = mask)
        transformed_image = transformed['image']
        transformed_mask = transformed['mask']
        os.makedirs("laboro_tomato/train/images/Elastic_transform", exist_ok=True)
        os.makedirs("laboro_tomato/train/masks/Elastic_transform", exist_ok=True)
        cv2.imwrite("laboro_tomato/train/images/Elastic_transform/Elastic_transform2_"+filename, transformed_image)
        cv2.imwrite("laboro_tomato/train/masks/Elastic_transform/Elastic_transform2_"+filename, transformed_mask)

        transformed = transformRandomBrightnessContrast(image = transformed_image, mask = transformed_mask)
        transformed_image = transformed['image']
        transformed_mask = transformed['mask']
        os.makedirs("laboro_tomato/train/images/Random_contrast", exist_ok=True)
        os.makedirs("laboro_tomato/train/masks/Random_contrast", exist_ok=True)
        cv2.imwrite("laboro_tomato/train/images/Random_contrast/Random_contrast_"+filename, transformed_image)
        cv2.imwrite("laboro_tomato/train/masks/Random_contrast/Random_contrast_"+filename, transformed_mask)

        transformed = transformRandomBrightnessContrast(image = image, mask = mask)
        transformed_image = transformed['image']
        transformed_mask = transformed['mask']
        os.makedirs("laboro_tomato/train/images/Random_contrast", exist_ok=True)
        os.makedirs("laboro_tomato/train/masks/Random_contrast", exist_ok=True)
        cv2.imwrite("laboro_tomato/train/images/Random_contrast/Random_contrast2_"+filename, transformed_image)
        cv2.imwrite("laboro_tomato/train/masks/Random_contrast/Random_contrast2_"+filename, transformed_mask)

        transformed = transformRandomGridShuffle(image = image, mask = mask)
        transformed_image = transformed['image']
        transformed_mask = transformed['mask']
        os.makedirs("laboro_tomato/train/images/Random_grid_shuffle", exist_ok=True)
        os.makedirs("laboro_tomato/train/masks/Random_grid_shuffle", exist_ok=True)
        cv2.imwrite("laboro_tomato/train/images/Random_grid_shuffle/Random_grid_shuffle_"+filename, transformed_image)
        cv2.imwrite("laboro_tomato/train/masks/Random_grid_shuffle/Random_grid_shuffle_"+filename, transformed_mask)

        transformed = transformRandomGridShuffle(image = image, mask = mask)
        transformed_image = transformed['image']
        transformed_mask = transformed['mask']
        os.makedirs("laboro_tomato/train/images/Random_grid_shuffle", exist_ok=True)
        os.makedirs("laboro_tomato/train/masks/Random_grid_shuffle", exist_ok=True)
        cv2.imwrite("laboro_tomato/train/images/Random_grid_shuffle/Random_grid_shuffle2_"+filename, transformed_image)
        cv2.imwrite("laboro_tomato/train/masks/Random_grid_shuffle/Random_grid_shuffle2_"+filename, transformed_mask)

        transformed = transformRandomCrop(image = image, mask = mask)
        transformed_image = transformed['image']
        transformed_mask = transformed['mask']
        os.makedirs("laboro_tomato/train/images/Random_sized_crop", exist_ok=True)
        os.makedirs("laboro_tomato/train/masks/Random_sized_crop", exist_ok=True)
        cv2.imwrite("laboro_tomato/train/images/Random_sized_crop/Random_sized_crop_"+filename, transformed_image)
        cv2.imwrite("laboro_tomato/train/masks/Random_sized_crop/Random_sized_crop_"+filename, transformed_mask)

        transformed = transformRandomCrop(image = image, mask = mask)
        transformed_image = transformed['image']
        transformed_mask = transformed['mask']
        os.makedirs("laboro_tomato/train/images/Random_sized_crop", exist_ok=True)
        os.makedirs("laboro_tomato/train/masks/Random_sized_crop", exist_ok=True)
        cv2.imwrite("laboro_tomato/train/images/Random_sized_crop/Random_sized_crop2_"+filename, transformed_image)
        cv2.imwrite("laboro_tomato/train/masks/Random_sized_crop/Random_sized_crop2_"+filename, transformed_mask)



if __name__ == "__main__":
    create_augmentated_images("laboro_tomato/train/images/tomato", "laboro_tomato/train/masks/tomato")



