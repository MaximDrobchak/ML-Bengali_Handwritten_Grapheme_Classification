from model import model
from load_preprocessing_data import get_training_dataset, get_validation_dataset
from constants import EPOCHS, STEP_PER_EPOCH, VALIDATION_STEP_PER_EPOCH, weight_path
from callbacks import callbacks_list

training_dataset = get_training_dataset()
validation_dataset = get_validation_dataset()

model.summary()

history = model.fit_generator(
    training_dataset,
    steps_per_epoch=STEP_PER_EPOCH,
    epochs=EPOCHS,
    validation_data=validation_dataset,
    validation_steps=VALIDATION_STEP_PER_EPOCH,
    callbacks=callbacks_list)


model.load_weights(weight_path)
model.save('model_tpu_gpu_cpu.h5')