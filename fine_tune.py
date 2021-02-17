from os import listdir
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD


#https://www.pyimagesearch.com/2019/06/03/fine-tuning-with-keras-and-deep-learning/
def fine_tune_resnet(images,model):

    print("Fine tuning")

def get_files(folder):

    files = listdir(folder)
    return pd.DataFrame(data=files,columns=['img_name'])

def split_test_train(images,ratio_test=0.4,ratio_train=0.4,ratio_val=0.2):

    #Create new folders
    print("Splitting data")

def plot_training(history,N):
    # construct a plot that plots and saves the training history
    plt.style.use("ggplot")
    fig, ax1 = plt.subplots()
    ax1.plot(np.arange(0, N), history["loss"], label="train_loss",color='red')
    ax1.plot(np.arange(0, N), history["val_loss"], label="val_loss")
    ax1.set_xlabel("Epoch #")
    ax1.set_ylabel("Loss")
    ax2 = ax1.twinx()
    ax2.plot(np.arange(0, N), history["accuracy"], label="train_acc",color='black')
    ax2.plot(np.arange(0, N), history["val_accuracy"], label="val_acc")
    ax2.set_ylabel("Accuracy")
    #ax1.title("Training Loss and Accuracy")
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':

    #Set folder and site
    folder = "/Users/codykupf/Documents/Projects/river-cnn/"
    images = "{}Images/".format(folder)
    models = "{}Models/".format(folder)


    #Load model
    model = load_model("{}VGG_FineTune/Trial5".format(models))
    history_df=pd.read_csv("{}VGG_FineTune/Trial5.csv".format(models))

    plot_training(history_df,history_df.shape[0])

    '''

    #Get all the files
    img_files = get_files(images)
    print(img_files)

    num_classes = 200

    #Split the images into train, test, validate
    df_train = img_files.sample(n=num_classes,random_state=14)
    print("Generating training set")

    #Create data augmentation for random flipping and zooming
    dataAug = ImageDataGenerator(
        width_shift_range=0.4,
        horizontal_flip=True,
        vertical_flip=True,
        validation_split=0.4)

    trainGen = dataAug.flow_from_dataframe(
        dataframe=df_train,
        directory=images,
        x_col="img_name",
        y_col="img_name",
        class_mode="categorical",
        target_size=(224,224),
        batch_size=24,
        shuffle=True,
        seed=14,
        subset='training')

    valGen = dataAug.flow_from_dataframe(
        dataframe=df_train,
        directory=images,
        x_col="img_name",
        y_col="img_name",
        class_mode="categorical",
        target_size=(224,224),
        batch_size=24,
        shuffle=True,
        seed=14,
        subset='validation')

    #From https://www.pyimagesearch.com/2019/06/03/fine-tuning-with-keras-and-deep-learning/
    # load the VGG16 network, ensuring the head FC layer sets are left
    # off
    baseModel = VGG16(weights="imagenet", include_top=False,
                      input_tensor=Input(shape=(224, 224, 3)))

    # construct the head of the model that will be placed on top of the
    # the base model
    headModel = baseModel.output
    headModel = Flatten(name="flatten")(headModel)
    headModel = Dense(512, activation="relu")(headModel)
    headModel = Dropout(0.5)(headModel)
    headModel = Dense(num_classes, activation="softmax")(headModel)

    # place the head FC model on top of the base model (this will become
    # the actual model we will train)
    model = Model(inputs=baseModel.input, outputs=headModel)

    # loop over all layers in the base model and freeze them so they will
    # *not* be updated during the first training process
    for layer in baseModel.layers:
        layer.trainable = False

    print(model.summary())

    # compile our model (this needs to be done after our setting our
    # layers to being non-trainable
    print("[INFO] compiling model...")
    opt = SGD(lr=1e-4, momentum=0.9)
    model.compile(loss="categorical_crossentropy", optimizer=opt,
                  metrics=["accuracy"])

    # train the head of the network for a few epochs (all other layers
    # are frozen) -- this will allow the new FC layers to start to become
    # initialized with actual "learned" values versus pure random
    print("[INFO] training head...")
    BATCH_SIZE = 24
    num_epochs = 100
    # history = model.fit(
    #     trainGen, valGen,
    #     steps_per_epoch=num_classes // BATCH_SIZE,
    #     validation_steps=num_classes // BATCH_SIZE,
    #     epochs=num_epochs)
    history = model.fit(
        trainGen,validation_data=valGen,
        steps_per_epoch=trainGen.n/trainGen.batch_size,
        validation_steps=valGen.n/valGen.batch_size,
        epochs=num_epochs)

    #Save results
    model.save("{}VGG_FineTune/Trial5".format(models))
    history_df = pd.DataFrame(history.history)
    history_df.to_csv("{}VGG_FineTune/Trial5.csv".format(models))

    plot_training(history_df,num_epochs)
    

    '''

