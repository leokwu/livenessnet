# USAGE
# python3.6 train_liveness.py --dataset dataset --model liveness.model --le le.pickle

# set the matplotlib backend so figures can be saved in the background
import matplotlib

matplotlib.use("Agg")

# import the necessary packages
from pyimagesearch.livenessnet import LivenessNet
from net.audioappraisenet import AudioAppraiseNet
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.utils import np_utils
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import pickle
import cv2
import os
import tensorflow as tf
from keras import backend as K
import keras
from keras.models import Model
# from keras.utils import multi_gpu_model

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
                help="path to input dataset")
ap.add_argument("-m", "--model", type=str, required=True,
                help="path to trained model")
ap.add_argument("-l", "--le", type=str, required=True,
                help="path to label encoder")
ap.add_argument("-p", "--plot", type=str, default="plot.png",
                help="path to output loss/accuracy plot")
args = vars(ap.parse_args())

# initialize the initial learning rate, batch size, and number of
# epochs to train for
INIT_LR = 1e-4
BS = 8
# EPOCHS = 50
EPOCHS = 1

# grab the list of images in our dataset directory, then initialize
# the list of data (i.e., images) and class images
print("[INFO] loading images...")
imagePaths = list(paths.list_images(args["dataset"]))
#print("imagePaths: ", imagePaths)
data = []
labels = []

for imagePath in imagePaths:
    # extract the class label from the filename, load the image and
    # resize it to be a fixed 32x32 pixels, ignoring aspect ratio
    label = imagePath.split(os.path.sep)[-2]
    # print(": ", imagePath.split(os.path.sep)[1])
    # print("os.path.sep: ", os.path.sep)
    # print("label: ", label)
    image = cv2.imread(imagePath)
    image = cv2.resize(image, (32, 32))

    # update the data and labels lists, respectively
    data.append(image)
    labels.append(label)

# convert the data into a NumPy array, then preprocess it by scaling
# all pixel intensities to the range [0, 1]
data = np.array(data, dtype="float") / 255.0

# encode the labels (which are currently strings) as integers and then
# one-hot encode them
le = LabelEncoder()
labels = le.fit_transform(labels)
labels = np_utils.to_categorical(labels, 2)

# partition the data into training and testing splits using 75% of
# the data for training and the remaining 25% for testing
(trainX, testX, trainY, testY) = train_test_split(data, labels,
                                                  test_size=0.25, random_state=42)

# construct the training image generator for data augmentation
aug = ImageDataGenerator(rotation_range=20, zoom_range=0.15,
                         width_shift_range=0.2, height_shift_range=0.2, shear_range=0.15,
                         horizontal_flip=True, fill_mode="nearest")

# initialize the optimizer and model
print("[INFO] compiling model...")
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
# model = LivenessNet.build(width=32, height=32, depth=3,
#                          classes=len(le.classes_))
# mobilenetv2
model = AudioAppraiseNet.build_mobilenetv2(width=128, height=128, depth=3,
                           classes=len(le.classes_), reg=l2(0.0004))
# model = multi_gpu_model(model, gpus=4)
model.compile(loss="binary_crossentropy", optimizer=opt,
              metrics=["accuracy"])

callbacks = [keras.callbacks.TensorBoard(log_dir='./logs', update_freq='epoch'),
             keras.callbacks.ModelCheckpoint(filepath='./checkpoints/livenessnet.{epoch:02d}.hdf5',
                                             verbose=0, save_best_only=False, save_weights_only=False, mode='auto',
                                             period=1)
             ]
# train the network
print("len(trainX) // BS: ", len(trainX) // BS)
print("[INFO] training network for {} epochs...".format(EPOCHS))
H = model.fit_generator(aug.flow(trainX, trainY, batch_size=BS),
                        validation_data=(testX, testY),
                        steps_per_epoch=len(trainX) // BS,
                        epochs=EPOCHS,
                        workers=10,
                        use_multiprocessing=True,
                        callbacks=callbacks)

# evaluate the network
print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=BS)
print(classification_report(testY.argmax(axis=1),
                            predictions.argmax(axis=1), target_names=le.classes_))

# save the network to disk
print("[INFO] serializing network to '{}'...".format(args["model"]))
model.save(args["model"])
model.save_weights('./weights/livenessnet_weights.h5')
#model.load_weights('./weights/livenessnet_weights.h5')
# print model info as json/yaml
# json_strig = model.to_json()
# print("json_strig: ", json_strig)
# yaml_string = model.to_yaml()
# print("yaml_string: ", yaml_string)

# output intermediate layer model result
# layer_name = 'dense_2'
# intermediate_layer_model = Model(inputs=model.input,
#                                     outputs=model.get_layer(layer_name).output)
# intermediate_output = intermediate_layer_model.predict(testX, batch_size=BS)
# print("intermediate_output: --------> ", intermediate_output)

# save the label encoder to disk
f = open(args["le"], "wb")
f.write(pickle.dumps(le))
f.close()

print("input is: ", model.input.op.name)
print("output is: ", model.output.op.name)
# save pb model
# sess = K.get_session()
# frozen_graph_def = tf.graph_util.convert_variables_to_constants(
#     sess,
#     sess.graph_def,
#    output_node_names=["activation_6/Softmax"])
# with tf.gfile.GFile('./model/livenessnet_model.pb', "wb") as f:
#    f.write(frozen_graph_def.SerializeToString())
# tf.train.write_graph(frozen_graph_def, 'model', 'livenessnet_model.pb', as_text=True)
# tf.train.write_graph(frozen_graph_def, 'model', 'livenessnet_model.pb', as_text=False)

# Training set accuracy--------------------------------
result = model.evaluate(trainX, trainY, batch_size=10)
print('\nTrain Acc:', result[1])
# print('\nTrain Los:', result[0])

# Testing set accuracy---------------------------------
result = model.evaluate(testX, testY, batch_size=10000)
print('\nTest Acc:', result[1])
# print('\nTest Los:', result[0])

# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, EPOCHS), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, EPOCHS), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, EPOCHS), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, EPOCHS), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(args["plot"])
