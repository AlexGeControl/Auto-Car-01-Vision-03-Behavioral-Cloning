
# coding: utf-8

# ## Set Up Session

# In[ ]:


# Configuration:
from behavioral_cloning.utils.conf import Conf
# ETL:
import numpy as np
from behavioral_cloning.utils.dataset import Dataset
from sklearn.model_selection import train_test_split
# Preprocessing:
from behavioral_cloning.preprocessors import Preprocessor
# Modeling:
from keras.models import Sequential
from keras.layers.core import Lambda, Dropout, Flatten, Dense 
from keras.layers.convolutional import Cropping2D, Convolution2D, MaxPooling2D
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
# Visualization:
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')


# ## Load Configuration

# In[ ]:


conf = Conf('conf/conf.json')


# ## Training/Testing Split

# #### Load Dataset

# In[ ]:


X, y = [], []

preprocessor = Preprocessor()
for dataset_dir in conf.datasets:
    dataset = Dataset(
        dataset_dir,
        steering_correction = 0.16
    )
    for (images, steerings) in iter(dataset):
        # Crop & resize to reduce memory footprint:
        processed = preprocessor.transform(images)
        X.append(processed)
        y.append(steerings)

X = np.vstack(X)
y = np.hstack(y)


# #### Save for Future Access

# #### Split Training & Testing Subsets

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.20, 
    random_state=42
)


# ## Build Network

# #### Configurations

# In[ ]:


INPUT_SHAPE = (18, 64, 3)
N_OUTPUT = 1

BATCH_SIZE = 256
N_EPOCHES = 256


# #### Model Architecture

# In[ ]:


def resize(X):
    # Set up session:
    from keras.backend import tf as ktf
    
    return ktf.image.resize_images(X, (32, 64))

def standardize(X):
    # Set up session:
    from keras.backend import tf as ktf
    
    return ktf.map_fn(
        lambda x: ktf.image.per_image_standardization(x),
        X
    )


# In[ ]:


model = Sequential()

# Preprocessing:
model.add(
    Lambda(
        standardize,
        input_shape = INPUT_SHAPE
    )
)
# Convs:
model.add(
    Convolution2D(
        32, 3, 3, border_mode='same',
        activation='relu'
    )
)
model.add(
    Convolution2D(
        64, 3, 3, border_mode='same', 
        activation='relu'
    )
)
model.add(
    MaxPooling2D(pool_size=(2, 2))
)
model.add(
    Convolution2D(
        128, 3, 3, border_mode='same', 
        activation='relu'
    )
)
model.add(
    MaxPooling2D(pool_size=(2, 2))
)
model.add(
    Dropout(0.25)
)
# Flatten:
model.add(
    Flatten()
)
# Fully connected:
model.add(
    Dense(
        512, 
        activation='relu'
    )
)
model.add(
    Dense(
        256, 
        activation='relu'
    )
)
model.add(
    Dense(
        128, 
        activation='relu'
    )
)
model.add(
    Dropout(0.50)
)
# Output:
model.add(
    Dense(
        N_OUTPUT, 
        activation='linear'
    )
)

model.compile(
    loss='mean_squared_error',
    optimizer=Adam(lr=5e-4),
    metrics=['mae']
)


# #### Optimization

# In[ ]:


callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=8,verbose=0
    )
]


# In[ ]:


model.fit(
    X_train, y_train,
    batch_size = BATCH_SIZE,
    nb_epoch = N_EPOCHES,
    verbose = 1,
    shuffle=True, validation_split=0.1, callbacks=callbacks
)


# #### Evaluation

# In[ ]:


score = model.evaluate(X_test, y_test, verbose=0)

print('Test loss:', score[0])
print('Test mae:', score[1])


# #### Save for Deployment

# In[ ]:


model.save('model.h5')

