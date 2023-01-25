import numpy as np
from torchvision import transforms


def transform(new_size, to_tensor=False):
    transform_list = []
    if to_tensor:
        transform_list.append(transforms.ToTensor())
    transform_list.append(transforms.Resize((new_size, new_size)))
    transform_list.append(transforms.RandomHorizontalFlip())
    transform = transforms.Compose(transform_list)
    return transform


class TransformCV2:
    ''' Transform function for TFRecords dataset iterator '''
    def __init__(self, batch_size, orig_dim=None, resize_dim=256):
        self.orig_dim = orig_dim
        self.resize_dim = resize_dim
        self.batch_size = batch_size

        self.transform = transform(resize_dim, to_tensor=True)
    
    def __call__(self, features):
        if self.orig_dim is not None: # For resize as TFRecords in bytes does not encode back to image shape
            features["image"] = features["image"].reshape(self.orig_dim, self.orig_dim, 3)

        new_features = np.empty((3, self.resize_dim, self.resize_dim))
        for i in range(3):
            new_features[i] = self.transform(features["image"][:,:,i])
        features["image"] = new_features[[0,2,1], :, :]
        return features