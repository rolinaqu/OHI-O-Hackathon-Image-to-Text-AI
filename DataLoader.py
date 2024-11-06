import numpy as np
import cv2
import random
from collections import namedtuple

Sample = namedtuple('Sample', 'gt_text, file_path')
Batch = namedtuple('Batch', 'img, gt_text, batch_size')

class DataLoader:
    def __init__ (self) -> None:
        # assert data_dir.exists()
        data_split = 0.95
        self.samples = []
        self.curr_set = []
        self.batch_size = 5

        # temporarily hard coding, replace with argument later
        data_dir = 'C:\\Users\\plano\\Documents\\1-SCHOOL STUFF\\2024-2025 Year 3\\Hackathon OHIO 2024\\Training Data' 
        f = open(data_dir + '\\' + 'ascii\lines.txt')
        #chars = set()
        for line in f:
            line = line.strip()
            if not line or line[0] == '#':
                continue
            line_split = line.split(' ')
            assert len(line_split) >= 9

            # splits first filename (ex 'a01-000u-03' into a list of strings)
            #ex ['ao1', '000u', '03' ]
            file_name_split = line_split[0].split('-')

            #first directory ('ao1'), string type
            file_name_subdir1 = file_name_split[0]

            #second directory ('000u')
            file_name_subdir2 = file_name_split[0] + '-' + file_name_split[1]

            #name of file ('03.png')
            file_base_name = line_split[0] + '.png'

            #full path to specified file on windows machine ('...\Training Data\lines\a01-000u-03)
            file_name = data_dir + '\\' + 'lines' + '\\' + file_name_subdir1 + '\\' + file_name_subdir2 + '\\' + file_base_name

            #outputs version of line with spaces instead of | 
            #A|MOVE|to|stop|Mr.|Gaitskell|from -> A MOVE to stop Mr. Gaitskell from
            gt_text = line_split[8].replace('|', ' ')
            self.samples.append(Sample(gt_text, file_name))

        #95 - 5 split
        split_idx = int(data_split * len(self.samples))

        #train_samples = list of Sample tuples(gt_text, filename) (String, String) types
        #(String line, String filename)
        self.train_samples = self.samples[:split_idx]
        self.validation_samples = self.samples[split_idx:]

        # put words into lists
        #self.train_words = [x.gt_text for x in self.train_samples]
        #self.validation_words = [x.gt_text for x in self.validation_samples]


        #starts with training set
        self.train_set()

    def train_set(self) -> None:
        """Switch to randomly chosen subset of training set."""
        self.data_augmentation = True
        self.curr_idx = 0
        random.shuffle(self.train_samples)
        self.samples = self.train_samples
        self.curr_set = 'train'

    def validation_set(self) -> None:
        self.curr_idx = 0
        self.samples = self.validation_samples
        self.curr_set = 'val'

    def _get_img(self, i: int) -> np.ndarray:
        #reads in image and returns ndarray
        img = cv2.imread(self.samples[i].file_path, cv2.IMREAD_GRAYSCALE)

        return img
    

    def get_next(self) -> Batch:
        # gets next element - BATCH SIZE OF 5
        batch_range = range(self.curr_idx, min(self.curr_idx + self.batch_size, len(self.samples)))

        #iterates over each index in batch range and makes list of images
        #list comprehensions yay
        imgs = [self._get_img(i) for i in batch_range]
        gt_texts = [self.samples[i].gt_text for i in batch_range]
        
        #increments by batch size 
        self.curr_idx = self.curr_idx + self.batch_size

        return Batch(imgs, gt_texts, self.batch_size)
    


    def has_next(self) -> bool:
        """Is there a next element?"""
        if self.curr_set == 'train':
            return self.curr_idx + self.batch_size <= len(self.samples)  # train set: only full-sized batches
        else:
            return self.curr_idx < len(self.samples)  # val set: allow last batch to be smaller








