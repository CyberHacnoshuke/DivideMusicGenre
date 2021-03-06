from keras.models import load_model
import MakeModels
import numpy as np

model = load_model('model_whistle.h5')
test_mcff = MakeModels.load_a_file('/testWav/answer0_1.wav')
test_mcff_final = test_mcff.reshape(1, 20, 11, 1)
print(np.argmax(model.predict(test_mcff_final)))