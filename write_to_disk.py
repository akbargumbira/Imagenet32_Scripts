# coding=utf-8
import os
import getpass
import time

from PIL import Image

from test import load_data
from utils import get_label_dict


get_synset_dict = {v: k for k, v in get_label_dict().items()}
base_pickled_train_dir = '/home/%s/dev/data/imagenet/downsampled/Imagenet32_train' % getpass.getuser()
base_output_dir = '/home/%s/dev/data/imagenet/downsampled/Imagenet32_train/out' % getpass.getuser()

start_time = time.strftime("%Y%m%d-%H%M%S")
print("Start time: %s" % start_time)

counter = 0
for i in range(10):
    file_path = 'train_data_batch_%s' % (i + 1)
    full_path = os.path.join(base_pickled_train_dir, file_path)
    x, y = load_data(full_path)

    for j in range(x.shape[0]):
        counter += 1
        image = Image.fromarray(x[j])
        synset = get_synset_dict[y[j]]
        output_dir_path = os.path.join(base_output_dir, synset)
        if not os.path.exists(output_dir_path):
            os.makedirs(output_dir_path)

        image.save(
            os.path.join(output_dir_path, '%s_%s.JPEG' % (synset, counter)))

    print('Saved %s images...' % counter)

end_time = time.strftime("%Y%m%d-%H%M%S")
print("End time: %s" % end_time)
