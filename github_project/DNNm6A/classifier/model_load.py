from keras.models import load_model

H5_root = r"human-kidney\H5"
name = "H_B"


H5_path = H5_root + r'\{}.h5'.format(name)
model_reloaded = load_model(H5_path)
