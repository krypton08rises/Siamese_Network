import numpy as np
import loadimg
import numpy.random as rng


def get_batch(batch_size, s="train"):
    """
    Create batch of n pairs, half same class, half different class
    """
    """
    if s == 'train':
        X = X_train
        # categories = train_classes
    else:
        X = X_val
        categories = val_classes
        """

    num_people, num_example, w, h, channel = X_train.shape
# """"""

    # randomly sample several classes to use in the batch
    categories = rng.choice(num_people, size=(batch_size,), replace=False)              # shape = (12,)
    # initialize 2 empty arrays for the input image batch
    pairs = [np.zeros((batch_size, h, w, channel, 1)) for i in range(2)]                # shape = (2, 12
    # print(len(pairs))
    # print(len(pairs[0]))
    # print(len(pairs[0][0]))
    # print(len(pairs[0][0][0]))
    # initialize vector for the targets
    targets = np.zeros((batch_size,))

    # make one half of it '1's, so 2nd half of batch has same class
    targets[batch_size // 2:] = 1
    for i in range(batch_size):
        category = categories[i]
        idx_1 = rng.randint(0, num_example)
        pairs[0][i, :, :, :, :] = X_train[category, idx_1].reshape(h, w, channel, 1)
        idx_2 = rng.randint(0, num_example)

        # pick images of same class for 1st half, different for 2nd
        if i >= batch_size // 2:
            category_2 = category
        else:
            # add a random number to the category modulo n classes to ensure 2nd image has a different category
            category_2 = (category + rng.randint(1, num_people)) % num_people

        pairs[1][i, :, :, :] = X_train[category_2, idx_2].reshape(h, w, channel,  1)

    return pairs, targets


pairs, targets = get_batch(12, s = "train")
print(targets)
