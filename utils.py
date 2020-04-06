import os
import pickle

import torch
import torch.nn.functional as F
import torch.utils.data as torch_data

import numpy as np
from skimage.measure import compare_psnr, compare_ssim

class fastMRIData(torch_data.Dataset):
    def __init__(self, path_to_source, path_to_sampled):
        super().__init__()

        self.path_to_source = path_to_source
        self.path_to_sampled = path_to_sampled

        source_images = set(os.listdir(path_to_source))
        sampled_images = set(os.listdir(path_to_sampled))
        intersected_images = source_images.intersection(sampled_images)
        images = sorted([img for img in intersected_images if img.endswith('npy')])

        self.images = images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        source_image = npy_load(os.path.join(self.path_to_source, self.images[idx]))
        sampled_image = npy_load(os.path.join(self.path_to_sampled, self.images[idx]))
        source_image = torch.from_numpy(source_image)
        sampled_image = torch.from_numpy(sampled_image)

        return source_image, sampled_image
class fastMRIData3D(torch_data.Dataset):
    def __init__(self, path_to_source, path_to_sampled):
        super().__init__()
        self.path_to_source = path_to_source
        self.path_to_sampled = path_to_sampled
        source_images = set(os.listdir(path_to_source))
        sampled_images = set(os.listdir(path_to_sampled))
        intersected_images = source_images.intersection(sampled_images)
        images = sorted([img for img in intersected_images if img.endswith('npy')])
        ids = set([image_name.split("_")[0] for image_name in images])
        images_dict = {}
        for _id in ids:
            images_dict[_id] = sorted([img.split("_")[1] for img in images if img.startswith(str(_id))])
        

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_name = self.images[idx]
        image_id, image_postfix = image_name.split("_")
        image_postfix = int(image_postfix)
        max_image_postfix = images_dict[image_id][-1]
        if image_postfix == 1:
            source_images = [
            npy_load(os.path.join(self.path_to_source, image_name)),
            npy_load(os.path.join(self.path_to_source, image_name)),
            npy_load(os.path.join(self.path_to_source, image_name)),
            npy_load(os.path.join(self.path_to_source, str(image_id) + "_" + str(image_postfix+1))),
            npy_load(os.path.join(self.path_to_source, str(image_id) + "_" + str(image_postfix+2)))
            ]
            sampled_images = [
            npy_load(os.path.join(self.path_to_sampled, image_name)),
            npy_load(os.path.join(self.path_to_sampled, image_name)),
            npy_load(os.path.join(self.path_to_sampled, image_name)),
            npy_load(os.path.join(self.path_to_sampled, str(image_id) + "_" + str(image_postfix+1))),
            npy_load(os.path.join(self.path_to_sampled, str(image_id) + "_" + str(image_postfix+2)))
            ]
        elif image_postfix == 2:
            source_images = [
            npy_load(os.path.join(self.path_to_source, str(image_id) + "_" + str(image_postfix-1))),
            npy_load(os.path.join(self.path_to_source, image_name)),
            npy_load(os.path.join(self.path_to_source, image_name)),
            npy_load(os.path.join(self.path_to_source, str(image_id) + "_" + str(image_postfix+1))),
            npy_load(os.path.join(self.path_to_source, str(image_id) + "_" + str(image_postfix+2)))
            ]
            sampled_images = [
            npy_load(os.path.join(self.path_to_sampled, str(image_id) + "_" + str(image_postfix-1))),
            npy_load(os.path.join(self.path_to_sampled, image_name)),
            npy_load(os.path.join(self.path_to_sampled, image_name)),
            npy_load(os.path.join(self.path_to_sampled, str(image_id) + "_" + str(image_postfix+1))),
            npy_load(os.path.join(self.path_to_sampled, str(image_id) + "_" + str(image_postfix+2)))
            ]
        elif image_postfix == max_image_postfix:
            source_images = [
            npy_load(os.path.join(self.path_to_source, str(image_id) + "_" + str(image_postfix-1))),
            npy_load(os.path.join(self.path_to_source, str(image_id) + "_" + str(image_postfix-2))),
            npy_load(os.path.join(self.path_to_source, image_name)),
            npy_load(os.path.join(self.path_to_source, image_name)),
            npy_load(os.path.join(self.path_to_source, image_name))
            ]
            sampled_images = [
            npy_load(os.path.join(self.path_to_sampled, str(image_id) + "_" + str(image_postfix-1))),
            npy_load(os.path.join(self.path_to_sampled, str(image_id) + "_" + str(image_postfix-2))),
            npy_load(os.path.join(self.path_to_sampled, image_name)),
            npy_load(os.path.join(self.path_to_sampled, image_name)),
            npy_load(os.path.join(self.path_to_sampled, image_name))
            ]
        elif image_postfix == max_image_postfix-1:
            source_images = [
            npy_load(os.path.join(self.path_to_source, str(image_id) + "_" + str(image_postfix-1))),
            npy_load(os.path.join(self.path_to_source, str(image_id) + "_" + str(image_postfix-2))),
            npy_load(os.path.join(self.path_to_source, image_name)),
            npy_load(os.path.join(self.path_to_source, image_name)),
            npy_load(os.path.join(self.path_to_source, str(image_id) + "_" + str(image_postfix+1)))
            ]
            sampled_images = [
            npy_load(os.path.join(self.path_to_sampled, str(image_id) + "_" + str(image_postfix-1))),
            npy_load(os.path.join(self.path_to_sampled, str(image_id) + "_" + str(image_postfix-2))),
            npy_load(os.path.join(self.path_to_sampled, image_name)),
            npy_load(os.path.join(self.path_to_sampled, image_name)),
            npy_load(os.path.join(self.path_to_sampled, str(image_id) + "_" + str(image_postfix+1)))
            ]
        else:
            source_images = [
            npy_load(os.path.join(self.path_to_source, str(image_id) + "_" + str(image_postfix-1))),
            npy_load(os.path.join(self.path_to_source, str(image_id) + "_" + str(image_postfix-2))),
            npy_load(os.path.join(self.path_to_source, image_name)),
            npy_load(os.path.join(self.path_to_source, str(image_id) + "_" + str(image_postfix+1)))
            npy_load(os.path.join(self.path_to_source, str(image_id) + "_" + str(image_postfix+2)))
            ]
            sampled_images = [
            npy_load(os.path.join(self.path_to_sampled, str(image_id) + "_" + str(image_postfix-1))),
            npy_load(os.path.join(self.path_to_sampled, str(image_id) + "_" + str(image_postfix-2))),
            npy_load(os.path.join(self.path_to_sampled, image_name)),
            npy_load(os.path.join(self.path_to_sampled, str(image_id) + "_" + str(image_postfix+1))),
            npy_load(os.path.join(self.path_to_sampled, str(image_id) + "_" + str(image_postfix+2)))
            ]
        source_images = torch.from_numpy(np.stack(source_images, axis=2))
        sampled_images = torch.from_numpy(np.stack(sampled_images, axis=2))
        return source_images, sampled_images

def compare_imgs(img_true, img_rec, verbose=True):

    assert img_true.shape == img_rec.shape
    if len(img_true.shape) == 3:
        assert img_true.shape[0] == 1

        img_true = img_true[0]
        img_rec = img_rec[0]

    img_true = img_true.numpy() if isinstance(img_true, torch.Tensor) else img_true
    img_rec = img_rec.numpy() if isinstance(img_rec, torch.Tensor) else img_rec

    mae = abs(img_true - img_rec).mean()
    psnr = compare_psnr(img_true, img_rec)
    ssim = compare_ssim(img_true, img_rec)

    if verbose:
        print('\tMAE\tPSNR\tSSIM')
        print(f'score\t{mae:.3f}\t{psnr:.3f}\t{ssim:.3f}')

    return [mae, psnr, ssim]


def npy_load(path):
    with open(path, 'rb') as f:
        return np.load(f)


def load_torch_model(model, filename):
    state_dict = torch.load(filename)
    model.load_state_dict(state_dict)
    return model


def dumps_train_params(**kwargs):
    str_params = [f'\t\'{k}\': {v}'.replace('\n', '' ) for k, v in kwargs.items()]
    str_params = ['Train params:'] + str_params
    
    return '\n'.join(str_params)


def write_train_params(log_file, **kwargs):
    with open(log_file, 'w') as f:
        params = dumps_train_params(**kwargs)
        f.write(params + '\n\n')


def save_history_loss(path_to_save, train_losses, val_losses=None):

    losses = {'Train': train_losses, 'Val': val_losses}

    with open(path_to_save, 'wb') as f:
        pickle.dump(losses, f)


def load_history_loss(path_to_history):
    with open(path_to_history, 'rb') as f:
        losses = pickle.load(f)
        
    return losses


def scale_MRI(image, low=2, high=98):

    lp, hp = np.percentile(image, (low, high))
    image_scaled = np.clip(image, lp, hp)

    image_scaled -= image_scaled.min()
    image_scaled /= image_scaled.max()
    image_scaled = image_scaled.astype(np.float32)

    return image_scaled


def scale_data(samples, path_to_save, threshold=5e-2):
    for s in samples:
        data = npy_load(s)
        data = scale_MRI(data)
        data = data[np.newaxis, :, :]
        data[np.where(data < threshold)] = 0

        name = os.path.split(s)[1]

        with open(os.path.join(path_to_save, name), 'wb') as f:
            np.save(f, data)
