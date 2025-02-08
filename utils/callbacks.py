# MIT License
# 
# Copyright (c) 2024 Romain Ilbert
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import logging
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler, ModelCheckpoint

from .model_utils import cosine_annealing


class CaptureWeightsCallback(tf.keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.penultimate_weights = None
        self.attention_weights_history = []
    
    def on_epoch_end(self, epoch, logs=None):
        if epoch % 5 == 0:
            last_attention_weights = self.model.get_last_attention_weights()
            if last_attention_weights is not None:
                self.attention_weights_history.append(last_attention_weights)
    
    def get_attention_weights_history(self):
        return self.attention_weights_history



def setup_callbacks(args, checkpoint_path, model):
    """
    Sets up and returns TensorFlow callbacks for use during model training. These callbacks include early stopping,
    learning rate scheduling, model checkpointing, and a custom callback for capturing model weights.

    This function is tailored to support flexible training configurations, allowing for dynamic adjustment of training
    behavior based on model performance and training progress.

    Parameters:
        args (argparse.Namespace): Parsed command-line arguments containing training configurations such as patience for early stopping,
                                   total training epochs, and initial learning rate.
        checkpoint_path (str): File path where model checkpoints will be saved. The best model according to validation loss is checkpointed.
        model (tf.keras.Model): The TensorFlow model being trained. Required for initializing the `CaptureWeightsCallback`.
        model_name (str): Name of the model being trained. This can be used to adjust callback behavior for different models.

    Returns:
        tuple: A tuple containing:
               - A list of TensorFlow callbacks configured for the training session.
               - An instance of `CaptureWeightsCallback`, which can be used post-training to access captured weights.

    Raises:
        Exception: If an error occurs in the setup of callbacks, an exception is logged and raised to prevent silent training failures.

    Example:
        >>> callbacks, capture_weights_callback = setup_callbacks(args, './model_checkpoints', model, 'my_model')
        This example demonstrates how to invoke `setup_callbacks` to obtain configured callbacks for training, including a custom
        weight capture callback for post-training analysis.
    """
    try:
        checkpoint_callback = ModelCheckpoint(
            filepath=checkpoint_path,
            monitor='val_loss',  
            verbose=1,
            save_best_only=True,
            save_weights_only=True,
        )

        early_stop_callback = EarlyStopping(
            monitor='val_loss',
            patience=args.patience,
            verbose=1,
        )

        lr_schedule_callback = LearningRateScheduler(
            lambda epoch: cosine_annealing(epoch, 5, args.learning_rate, 1e-6),
            verbose=1,
        )

        capture_weights_callback = CaptureWeightsCallback()

        callbacks = [checkpoint_callback, lr_schedule_callback, capture_weights_callback, early_stop_callback]
        
        if args.model == 'tsmixer':
            callbacks = [checkpoint_callback, lr_schedule_callback, early_stop_callback]
        return callbacks, capture_weights_callback
    except Exception as e:
        logging.error(f"Error setting up callbacks: {e}")
        raise