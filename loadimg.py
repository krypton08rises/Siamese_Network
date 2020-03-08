import os
import cv2
import numpy as np
import numpy.random as rng
import time
import pickle
from keras import Model, Sequential
from keras.layers import Conv2D, ZeroPadding2D, Activation, Input, concatenate
from keras.layers.pooling import MaxPooling2D
from keras.layers.core import Lambda, Flatten, Dense
from keras.backend import abs
from keras.regularizers import l2
from keras.optimizers import Adam

# import keras as K
# from keras.layers.normalization import BatchNormalization
# from keras.layers.merge import Concatenate
# from keras.initializers import glorot_uniform
# from keras.layers.normalization import BatchNormalization
# from scipy.misc import imread                         # deprecated

def loadimgs(dir_path, n=0):
    '''
    path => Path of train directory or test directory
    path = "/images background/"
    '''
    X=[]
    y = []
    cat_dict = {}                                                           # dictionary of categories
    person_dict = {}                                                        # dictionary of person
    curr_y = n
    # we load every alphabet seperately so   we can isolate them later
    # for dir in os.listdir(path):                                             # for a  person in path, get the photos of that person
    #     # print("loading directory: " + dir)
    #     # print(curr_y)
    #     dir_path = os.path.join(path, dir)
    #     # every person has it's own column in the array, so  load seperately
    for person in os.listdir(dir_path):
        person_path = os.path.join(dir_path, person)
        # i=0
        person_dict[person] = [curr_y, None]
        category_images = []
        # print(person_path)
        for feet in os.listdir(person_path):
            image_path = os.path.join(person_path, feet)
            image = cv2.imread(image_path)
            image= cv2.resize(image,(125, 175), interpolation= cv2.INTER_AREA)
            category_images.append(image)
            y.append(curr_y)
            curr_y+=1

           # if i==0:
                # cv2.imshow('image', image)
                # print(image_path)

            # i+=1

        try :
           X.append(np.stack(category_images))
        except ValueError as e:
            print(e)
            print("error - category_images:", category_images)
        person_dict[person][1] = curr_y - 1

# category_images.append(image)
# y.append(curr_y)
    # print(y)
    y = np.vstack(y)
    X = np.stack(X)
    X = X[0:12, :, :, :, :]
    Y = y[0:12*8]
    # print(str(X_train.shape)+"   "+str(Y_train.shape))
    return X, Y, person_dict

save_path ="/home/kr08rises/My_siamese_implementation/data/"

X_train, Y_train, train_dict = loadimgs("/home/kr08rises/My_siamese_implementation/footprint_dataset/Images_background", n=0)
with open(os.path.join(save_path,"train.pickle"), "wb") as f:                                                           # saving training tensors on disk
    pickle.dump((X_train,train_dict),f)


X_val, Y_val, val_dict = loadimgs("/home/kr08rises/My_siamese_implementation/footprint_dataset/Images_evaluation", n=0)
with open(os.path.join(save_path,"val.pickle"), "wb") as f:
    pickle.dump((X_val,val_dict),f)

# print(X_train.shape)
# print(person_dict.keys())
# print(person_dict)


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
    pairs = [np.zeros((batch_size, h, w, channel)) for i in range(2)]                # shape = (2, 12
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
        pairs[0][i, :, :, :] = X_train[category, idx_1].reshape(h, w, channel)
        idx_2 = rng.randint(0, num_example)

        # pick images of same class for 1st half, different for 2nd
        if i >= batch_size // 2:
            category_2 = category
        else:
            # add a random number to the category modulo n classes to ensure 2nd image has a different category
            category_2 = (category + rng.randint(1, num_people)) % num_people

        pairs[1][i, :, :, :] = X_train[category_2, idx_2].reshape(h, w, channel)

    return pairs, targets

# pairs, targets = get_batch(12, s = "train")
# print(pairs.shape)
# print(targets)

# def generate(batch_size, s="train"):
#     """
#     a generator for batches, so model.fit_generator can be used.
#     """
#     # while True:
#         pairs, targets = get_batch(batch_size, s)
#         yield (pairs, targets)
#
# p, t = generate(12, s="train")
# print(pairs.shape+targets.shape)



def initialize_weights(shape, dtype=None, name=None):
    """
        The paper, http://www.cs.utoronto.ca/~gkoch/files/msc-thesis.pdf
        suggests to initialize CNN layer weights with mean as 0.0 and standard deviation of 0.01
    """
    return np.random.normal(loc=0.0, scale=1e-2, size=shape)


def initialize_bias(shape,dtype = None, name=None):
    """
        The paper, http://www.cs.utoronto.ca/~gkoch/files/msc-thesis.pdf
        suggests to initialize CNN layer bias with mean as 0.5 and standard deviation of 0.01
    """
    return np.random.normal(loc = 0.5, scale = 1e-2, size = shape)


def get_siamese_model(input_shape):
    """
        Model architecture
    """

    # Define the tensors for the two input images
    left_input = Input(input_shape)
    right_input = Input(input_shape)

    # Convolutional Neural Network
    model = Sequential()
    model.add(Conv2D(64, (10, 10), activation='relu', input_shape=input_shape,
                     kernel_initializer=initialize_weights, kernel_regularizer=l2(2e-4)))
    model.add(MaxPooling2D())
    model.add(Conv2D(128, (7, 7), activation='relu',
                     kernel_initializer=initialize_weights,
                     bias_initializer=initialize_bias, kernel_regularizer=l2(2e-4)))
    model.add(MaxPooling2D())
    model.add(Conv2D(128, (4, 4), activation='relu', kernel_initializer=initialize_weights,
                     bias_initializer=initialize_bias, kernel_regularizer=l2(2e-4)))
    model.add(MaxPooling2D())
    model.add(Conv2D(256, (4, 4), activation='relu', kernel_initializer=initialize_weights,
                     bias_initializer=initialize_bias, kernel_regularizer=l2(2e-4)))
    model.add(Flatten())
    model.add(Dense(4096, activation='sigmoid',
                    kernel_regularizer=l2(1e-3),
                    kernel_initializer=initialize_weights, bias_initializer=initialize_bias))

    # Generate the encodings (feature vectors) for the two images
    encoded_l = model(left_input)
    encoded_r = model(right_input)

    # Add a customized layer to compute the absolute difference between the encodings
    L1_layer = Lambda(lambda tensors: abs(tensors[0] - tensors[1]))
    L1_distance = L1_layer([encoded_l, encoded_r])

    # Add a dense layer with a sigmoid unit to generate the similarity score
    prediction = Dense(1, activation='sigmoid', bias_initializer=initialize_bias)(L1_distance)

    # Connect the inputs with the outputs
    siamese_net = Model(inputs=[left_input, right_input], outputs=prediction)

    # return the model
    return siamese_net


model = get_siamese_model((125, 175, 3))
model.summary()

optimizer = Adam(lr=0.00006)
model.compile(loss="binary_crossentropy",optimizer=optimizer)

with open(os.path.join(save_path, "train.pickle"), "rb") as f:
    (Xtrain, train_classes) = pickle.load(f)

print("Training Feet: \n")
print(list(train_classes.keys()))

with open(os.path.join(save_path, "val.pickle"), "rb") as f:
    (Xval, val_classes) = pickle.load(f)

print("Validation Feet:", end="\n\n")
print(list(val_classes.keys()))


def make_oneshot_task(N, s="val", language=None):
    """Create pairs of test image, support set for testing N way one-shot learning. """
    if s == 'train':
        X = Xtrain
        categories = train_classes
    else:
        X = Xval
        categories = val_classes
    n_classes, n_examples, w, h = X.shape

    indices = rng.randint(0, n_examples, size=(N,))
    if language is not None:  # if language is specified, select characters for that language
        low, high = categories[language]
        if N > high - low:
            raise ValueError("This language ({}) has less than {} letters".format(language, N))
        categories = rng.choice(range(low, high), size=(N,), replace=False)

    else:  # if no language specified just pick a bunch of random letters
        categories = rng.choice(range(n_classes), size=(N,), replace=False)
    true_category = categories[0]
    ex1, ex2 = rng.choice(n_examples, replace=False, size=(2,))
    test_image = np.asarray([X[true_category, ex1, :, :]] * N).reshape(N, w, h, 1)
    support_set = X[categories, indices, :, :]
    support_set[0, :, :] = X[true_category, ex2]
    support_set = support_set.reshape(N, w, h, 1)
    targets = np.zeros((N,))
    targets[0] = 1
    targets, test_image, support_set = shuffle(targets, test_image, support_set)
    pairs = [test_image, support_set]

    return pairs, targets


def test_oneshot(model, N, k, s="val", verbose=0):
    """Test average N way oneshot learning accuracy of a siamese neural net over k one-shot tasks"""
    n_correct = 0
    if verbose:
        print("Evaluating model on {} random {} way one-shot learning tasks ... \n".format(k, N))
    for i in range(k):
        inputs, targets = make_oneshot_task(N, s)
        probs = model.predict(inputs)
        if np.argmax(probs) == np.argmax(targets):
            n_correct += 1
    percent_correct = (100.0 * n_correct / k)
    if verbose:
        print("Got an average of {}% {} way one-shot learning accuracy \n".format(percent_correct, N))
    return percent_correct


evaluate_every = 50 # interval for evaluating on one-shot tasks
batch_size = 12
n_iter = 500 # No. of training iterations
N_way = 20 # how many classes for testing one-shot tasks
n_val = 250 # how many one-shot tasks to validate on
best = -1

model_path = './weights/'

print("Starting training process!")
print("-------------------------------------")
t_start = time.time()
for i in range(1, n_iter+1):
    (inputs, targets) = get_batch(batch_size)
    loss = model.train_on_batch(inputs, targets)
    if i % evaluate_every == 0:
        print("\n ------------- \n")
        print("Time for {0} iterations: {1} mins".format(i, (time.time()-t_start)/60.0))
        print("Train Loss: {0}".format(loss))
        val_acc = test_oneshot(model, N_way, n_val, verbose=True)
        model.save_weights(os.path.join(model_path, 'weights.{}.h5'.format(i)))
        if val_acc >= best:
            print("Current best: {0}, previous best: {1}".format(val_acc, best))
            best = val_acc
t_end = time.time
print("Time Passed : " + str(t_end-t_start))

