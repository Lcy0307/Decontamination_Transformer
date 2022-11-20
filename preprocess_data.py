from PIL import Image
import numpy as np
from pathlib import Path

def preprocess_image(filelist, dataset = 'imagenet',file_format = '.jpeg'):
    file_list = []
    with open(filelist, 'r') as out:
        for line in out:
            file_list.append(line.rstrip("\n"))
    file_np = np.asarray(file_list)
    i = 0
    with open(f'./data_load/pre_{dataset}.txt', 'a') as out:
        for file_name in file_np:
            img = Image.open(file_name)
            new_img = img.resize((256, 256),Image.BICUBIC)
            new_img.save(f'./data_load/{dataset}/{str(i+1).zfill(5)}{file_format}')
            out.write(f'./data_load/{dataset}/{str(i+1).zfill(5)}{file_format}' + '\n')
            i = i + 1

# If you want to reproduce our results on FFHQ, please preprocess the data like this.
# While training and testing on FFHQ, contaminant sampled from (CelebA-HQ, ImageNet)

# FFHQ preprocess
# ffhq text means your ffhq filelist.
# remeber to make directory for name 'ffhq'
def preprocess(tp = 'train'):
    # output_dir = Path(f'./data_load/ffhq_{tp}')
    # output_dir.mkdir(exist_ok=True)
    # print('FFHQ ...')
    # preprocess_image(f'./data_load/ffhq_{tp}.txt', dataset = f'ffhq_{tp}', file_format = '.png')

    # # remeber to make directory for name 'celebA'
    # output_dir = Path(f'./data_load/celebA_{tp}')
    # output_dir.mkdir(exist_ok=True)
    # print('celebA ...')
    # preprocess_image(f'./data_load/celebA_{tp}.txt', dataset = f'celebA_{tp}', file_format = '.png')

    # remeber to make directory for name 'imagenet'
    output_dir = Path(f'./data_load/imagenet_{tp}')
    output_dir.mkdir(exist_ok=True)
    print('imagenet ...')
    preprocess_image(f'./data_load/imagenet_{tp}.txt', dataset = f'imagenet_{tp}', file_format = '.jpeg')
    
if __name__ == "__main__":
    # Preprocessing on training, validation, testing
    # print('Preprocess training ...')
    # preprocess('train')
    print('Preprocess valdation ...')
    preprocess('val')
    print('Preprocess testing ...')
    preprocess('test')


