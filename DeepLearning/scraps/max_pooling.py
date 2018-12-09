from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D

def build_max_pool(strides=None, pool_size=None, input_shape=None, padding='valid'):
	model = Sequential()
	model.add(MaxPooling2D(pool_size=pool_size, strides=strides, input_shape=input_shape, padding=padding))
	model.summary()

def main():
    build_max_pool(strides=2, pool_size=2, input_shape=(100, 100, 15), padding='same')

if __name__=='__main__':
    main()
