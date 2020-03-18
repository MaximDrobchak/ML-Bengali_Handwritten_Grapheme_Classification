
import os
from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf
from IPython.display import clear_output
import IPython.display as display
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, Callback
from constants import STEP_PER_EPOCH, strategy, EPOCHS, weight_path
# Matplotlib config
plt.ioff()
plt.rc('image', cmap='gray_r')
plt.rc('grid', linewidth=1)
plt.rc('xtick', top=False, bottom=False, labelsize='large')
plt.rc('ytick', left=False, right=False, labelsize='large')
plt.rc('axes', facecolor='F8F8F8', titlesize="large", edgecolor='white')
plt.rc('text', color='a8151a')
plt.rc('figure', facecolor='F0F0F0', figsize=(16,9))
# Matplotlib fonts
MATPLOTLIB_FONT_DIR = os.path.join(os.path.dirname(plt.__file__), "mpl-data/fonts/ttf")

def plot_learning_rate(lr_func, epochs):
    xx = np.arange(epochs+1, dtype=np.float)
    y = [lr_decay(x) for x in xx]
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.set_xlabel('epochs')
    ax.set_title('Learning rate\ndecays from {:0.3g} to {:0.3g}'.format(y[0], y[-2]))
    ax.minorticks_on()
    ax.grid(True, which='major', axis='both', linestyle='-', linewidth=1)
    ax.grid(True, which='minor', axis='both', linestyle=':', linewidth=0.5)
    ax.step(xx,y, linewidth=3, where='post')
    display.display(fig)

class PlotTraining(Callback):
    def __init__(self, sample_rate=1, zoom=1):
        self.sample_rate = sample_rate
        self.step = 0
        self.zoom = zoom
        self.steps_per_epoch = STEP_PER_EPOCH

    def on_train_begin(self, logs={}):
        self.batch_history = {}
        self.batch_step = []
        self.epoch_history = {}
        self.epoch_step = []
        self.fig, self.axes = plt.subplots(3, 2, figsize=(19, 19))
        self.fig.subplots_adjust(wspace=0.3, hspace=0.25)
        plt.ioff()


    def on_batch_end(self, batch, logs={}):
        if (batch % self.sample_rate) == 0:
            self.batch_step.append(self.step)
            for k,v in logs.items():
              # do not log "batch" and "size" metrics that do not change
              # do not log training accuracy "acc"
                if k=='batch' or k=='size' or k == 'loss':# or k=='acc':
                    continue
                self.batch_history.setdefault(k, []).append(v)
        self.step += 1

    def on_epoch_end(self, epoch, logs={}):
        plt.close(self.fig)
        for axes in self.axes:
            axes[0].cla()
            axes[1].cla()

            axes[0].set_ylim(0, 1.2/self.zoom)
            axes[1].set_ylim(1-1/self.zoom/2, 1+0.1/self.zoom/2)

        self.epoch_step.append(self.step)
        for k,v in logs.items():
          # only log validation metrics
            if not k.startswith('val_') or k == 'val_loss':
                continue
            self.epoch_history.setdefault(k, []).append(v)

        display.clear_output(wait=True)

        for count, (k,v) in enumerate(self.batch_history.items()):
            if count <= 2:
                self.axes[count][0].plot(np.array(self.batch_step) / self.steps_per_epoch, v, label='{}: {:.3f}'.format(k, v[len(v)-1]))
            else:
                self.axes[count-3][1].plot(np.array(self.batch_step) / self.steps_per_epoch, v, label='{}: {:.3f}'.format(k, v[len(v)-1]))

        for count, (k,v) in enumerate(self.epoch_history.items()):
            if count <= 2:
                self.axes[count][0].plot(np.array(self.epoch_step) / self.steps_per_epoch, v, label='{}: {:.3f}'.format(k, v[epoch]), linewidth=3)
            else:
                self.axes[count-3][1].plot(np.array(self.epoch_step) / self.steps_per_epoch, v, label='{}: {:.3f}'.format(k, v[epoch]), linewidth=3)
        for axes in self.axes:
            axes[0].legend()
            axes[1].legend()
            axes[0].set_xlabel('epochs')
            axes[1].set_xlabel('epochs')
            axes[0].minorticks_on()
            axes[0].grid(True, which='major', axis='both', linestyle='-', linewidth=1)
            axes[0].grid(True, which='minor', axis='both', linestyle=':', linewidth=0.5)
            axes[1].minorticks_on()
            axes[1].grid(True, which='major', axis='both', linestyle='-', linewidth=1)
            axes[1].grid(True, which='minor', axis='both', linestyle=':', linewidth=0.5)
        display.display(self.fig)


if strategy.num_replicas_in_sync == 1: # single GPU or CPU
    start_lr = 0.001
    min_lr = 0.00001
    max_lr = 0.03
    rampup_epochs = 9
    sustain_epochs = 0
    exp_decay = .75
else: # TPU pod
    start_lr =  0.001
    min_lr = 0.0158
    max_lr = 0.01 * strategy.num_replicas_in_sync
    rampup_epochs = 9
    sustain_epochs = 0
    exp_decay = .666

def lr_decay(epoch):
    def lr(epoch, start_lr, min_lr, max_lr, rampup_epochs, sustain_epochs, exp_decay):
        if epoch < rampup_epochs:
            lr = (max_lr - start_lr)/rampup_epochs * epoch + start_lr
        elif epoch < rampup_epochs + sustain_epochs:
            lr = max_lr
        else:
            lr = (max_lr - min_lr) * exp_decay**(epoch-rampup_epochs-sustain_epochs) + min_lr
        return lr
    return lr(epoch, start_lr, min_lr, max_lr, rampup_epochs, sustain_epochs, exp_decay)

lr_decay_callback = LearningRateScheduler(lambda epoch: lr_decay(epoch), verbose=True)

rng = [i for i in range(EPOCHS)]
y = [lr_decay(x) for x in rng]
plt.plot(rng, [lr_decay(x) for x in rng])
print(y[0], y[-1])
# plot_learning_rate(lr_decay_callback, EPOCHS)

checkpoint = ModelCheckpoint(weight_path, monitor='val_head_root_loss', verbose=1, save_best_only=True, mode='min', save_weights_only=True)

plot_training = PlotTraining(sample_rate=10, zoom=1)
callbacks_list = [checkpoint, plot_training, lr_decay_callback]