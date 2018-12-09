from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.core import Activation

def create_conv2d(input_shape=None):
	
	model = Sequential()

	model.add(Conv2D(filters=16, kernel_size=2, strides=1, padding='same', activation='relu', input_shape=input_shape))
	model.add(MaxPooling2D(pool_size=2, strides=2)) # Using 'valid' padding for max-pooling layer

	model.add(Conv2D(filters=32, kernel_size=2, strides=1, padding='same', activation='relu'))
	model.add(MaxPooling2D(pool_size=2, strides=2))

	model.add(Conv2D(filters=64, kernel_size=2, strides=1, padding='same', activation='relu'))
	model.add(MaxPooling2D(pool_size=2, strides=2))

	model.add(Activation('softmax'))
	
	model.summary()

def main():
	
	create_conv2d(input_shape=(32, 32, 3))

if __name__ == '__main__':
	main()

