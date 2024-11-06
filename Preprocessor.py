#import random
import cv2
import numpy as np

from DataLoader import Batch

class Preprocessor:

    def __init__ (self) -> None:
        self.img_size = [2000, 150]


    def process_img(self, img: np.ndarray) -> np.ndarray:
        if img is None:
            #some error happened while reading the file and img is empty
            img = np.zeros(self.img_size[::-1])
        
        resized = cv2.resize(img, (self.img_size))

        #convert to range -1, 1
        resized = resized / 255 - 0.5
        return resized
            
    
    def processBatch(self, batch: Batch) -> Batch:
        res_imgs = [self.process_img(img) for img in batch.img]
        return Batch(res_imgs, batch.gt_text, len(res_imgs))



def main():
    import matplotlib.pyplot as plt

    img = cv2.imread('Training Data\\lines\\a01\\a01-000u\\a01-000u-06.png', cv2.IMREAD_GRAYSCALE)
    img_aug = Preprocessor().process_img(img)
    plt.subplot(121)
    plt.imshow(img_aug, cmap='gray')
    #plt.subplot(122)
    #plt.imshow(img_aug, cmap='gray', vmin=0, vmax=1)
    plt.show()


if __name__ == '__main__':
    main()