import numpy as np
import os
from PIL import Image

#for each image vector in matrix X we associate an output y for the expected genre
#y can take the following values:
    #0 = Architecture
    #1 = Cookbooks
    #2 = Fantasy
    #3 = Romance
    #4 = Sience fiction

def process_and_save():
    """
    takes the images from the dataset folder and turns it into a matrix of numbers (trainable data)
    creates an npz file
    """
    outputs = sorted([i for i in os.listdir('dataset') if os.path.isdir(os.path.join('dataset', i))])

    X_list = []
    y_list = []
    total_images = 0

    for val, genre in enumerate(outputs):
        genre_dir = os.path.join('dataset', genre)
        files = os.listdir(genre_dir)
        print(f"Processing {genre}...")

        for f in files:
            path = os.path.join(genre_dir, f)

            #ensure format is RGB
            img = Image.open(path).convert('RGB')

            img = img.resize((160, 256))

            img_array = np.array(img)

            #normalizing data to [0,1]
            array_normalized = img_array / 255.0

            #turn into single dimension vector
            vect = array_normalized.flatten()

            X_list.append(vect) #dimensions 150x122880 
            y_list.append(val) #dimensions 150
            total_images += 1

    X = np.array(X_list) 
    y = np.array(y_list)

    print(f"\nDone! Processed {total_images} images.")
    print(f"Matrix Shape: {X.shape}")

    #save into a compressed file
    np.savez_compressed('numpy_data', X=X, y=y, class_names=outputs)
    print(f"Saved to '{'numpy_data'}'")


if __name__ == "__main__":
    process_and_save()
