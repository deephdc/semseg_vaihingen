# imports
import numpy as np
import semseg_vaihingen.config as cfg
from sklearn.utils import shuffle


def print_matrix(z):
    for row in z:
        print(row)


def rotate_90(z):
    z = flip_up_down(z)
    return np.swapaxes(z, 0, 1)


def rotate_180(z):
    z = flip_left_right(z)
    z = flip_up_down(z)
    return z


def rotate_270(z):
    z = np.swapaxes(z, 0, 1)
    return z[::-1]


def flip_left_right(z):
    return np.fliplr(z)


def flip_up_down(z):
    return np.flipud(z)


def choose_augmentation(z, num):
    if num == 1:
        return rotate_90(z)
    if num == 2:
        return rotate_180(z)
    if num == 3:
        return rotate_270(z)
    if num == 4:
        return flip_up_down(z)
    if num == 5:
        return flip_left_right(z)


# apply one random augmentation to every image in the dataset:
def every_element_randomly_once(x, y):
    rands = np.random.randint(1, cfg.NUM_LABELS, x.shape[0])
    x_aug = np.zeros(x.shape, dtype=x.dtype)
    y_aug = np.zeros(y.shape, dtype=y.dtype)
    for i, r in enumerate(rands):
        x_aug[i, ...] = choose_augmentation(x[i, ...], r)
        y_aug[i, ...] = choose_augmentation(y[i, ...], r)
    return x_aug, y_aug


# apply five augmentations to every image in the dataset:
def every_element_five_augmentations(x, y):
    x_aug = np.zeros((x.shape[0]*5, x.shape[1], x.shape[2], x.shape[3]), dtype=x.dtype)
    y_aug = np.zeros((y.shape[0]*5, y.shape[1], y.shape[2]), dtype=y.dtype)
   

    counter=0
    for i in range (0,x.shape[0]):
      for index_aug in range (1,6):
        x_aug[counter, ...] = choose_augmentation(x[i, ...], index_aug)
        y_aug[counter, ...] = choose_augmentation(y[i, ...], index_aug)
        counter=counter+1
    
    print(counter)
    print(x_aug.shape[0],x_aug.shape[1],x_aug.shape[2],x_aug.shape[3])
  
    return x_aug, y_aug



# shuffle both arrays reproducible and in the same way:
def shuffle_4d_sample_wise(x, y):
    x, y = shuffle(x, y, random_state=0)
    return x, y


def test_implementation_basic():
    z = np.arange(1, 10).reshape((3, 3))
    print_matrix(z)
    print('')
    print_matrix(choose_augmentation(z, 1))
    print('')
    print_matrix(choose_augmentation(z, 2))
    print('')
    print_matrix(choose_augmentation(z, 3))
    print('')
    print_matrix(choose_augmentation(z, 4))
    print('')
    print_matrix(choose_augmentation(z, 5))


def test_implementation():
    x = np.random.randint(0, 5, (4, 5, 5, 3))
    y = x[:, :, :, 0]
    x_a, y_a = every_element_randomly_once(x, y)
    print((x.shape, x_a.shape, y.shape, y_a.shape))
    print_matrix(x[0, :, :, 0])
    print_matrix(x_a[0, :, :, 0])
    print_matrix(y[0, :, :])
    print_matrix(y_a[0, :, :])


def main():
    test_implementation_basic()


if __name__ == '__main__':
    main()
