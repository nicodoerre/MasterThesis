import os
import random
import shutil	

def split_dataset(source_folder, train_folder, valid_folder,test_folder, train_ratio = 0.8, test_count=100):

    '''
    Splits the dataset into training, validation, and test sets.
    Parameters:
    - source_folder: Folder containing the original images.
    - train_folder: Folder to save training images.
    - valid_folder: Folder to save validation images.
    - test_folder: Folder to save test images.
    - train_ratio: Proportion of the dataset to include in the training set.
    - test_count: Number of images to include in the test set.
    '''

    if not os.path.exists(source_folder):
        raise Exception("Source folder does not exist")
    
    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(valid_folder, exist_ok=True)
    os.makedirs(test_folder, exist_ok=True)
    image_files = [f for f in os.listdir(source_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]
    random.shuffle(image_files)
    split_index = int(len(image_files)*train_ratio)
    train_files = image_files[:split_index]
    valid_files = image_files[split_index:]
    for file in train_files:
        shutil.move(os.path.join(source_folder, file), os.path.join(train_folder, file))
    for file in valid_files:
        shutil.move(os.path.join(source_folder, file), os.path.join(valid_folder, file))
        
    valid_files_post = [f for f in os.listdir(valid_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]
    test_files = random.sample(valid_files_post, test_count)
    for file in test_files:
        shutil.move(os.path.join(valid_folder, file), os.path.join(test_folder, file))
        
    print(f"Dataset split completed:")
    print(f"- {len(train_files)} images in '{train_folder}'")
    print(f"- {len(valid_files_post) - test_count} images in '{valid_folder}'")
    print(f"- {test_count} images in '{test_folder}'")   
     
def main():
    split_dataset('dataset_generation/generated_plots', 'dataset_generation/train', 'dataset_generation/valid','dataset_generation/test', train_ratio = 0.8)

if __name__ == "__main__":
    main()
