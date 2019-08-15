#import tensorboard
import os
import numpy as np
# import skimage
import tensorflow as tf
from tensorflow import keras
#
# def make_image(tensor):
#     """
#     Convert an numpy representation image to Image protobuf.
#     Copied from https://github.com/lanpa/tensorboard-pytorch/
#     """
#     from PIL import Image
#     x_size, y_size, channel = tensor.shape
#     image = Image.fromarray(tensor)
#     import io
#     output = io.BytesIO()
#     image.save(output, format='PNG')
#     image_string = output.getvalue()
#     output.close()
#     return tf.Summary.Image(height=x_size,
#                          width=y_size,
#                          colorspace=channel,
#                          encoded_image_string=image_string)

class TrainValTensorBoard(keras.callbacks.TensorBoard):
    def __init__(self, log_dir='./logs', write_images = False, seleted_layers = None,**kwargs):
        # Make the original `TensorBoard` log to a subdirectory 'training'
        training_log_dir = os.path.join(log_dir, 'training')
        super(TrainValTensorBoard, self).__init__(training_log_dir, **kwargs)

        # Log the validation metrics to a separate subdirectory
        self.val_log_dir = os.path.join(log_dir, 'validation')
        self.write_images = write_images
        self.selected_layers = seleted_layers
        # self.val_log_dir = log_dir.format('/validation')

    def set_model(self, model):
        # Setup writer for validation metrics
        self.val_writer = tf.summary.FileWriter(self.val_log_dir)
        super(TrainValTensorBoard, self).set_model(model)

    def on_epoch_end(self, epoch, logs=None):
        # Pop the validation logs and handle them separately with
        # `self.val_writer`. Also rename the keys so that they can
        # be plotted on the same figure with the training metrics
        logs = logs or {}
        val_logs = {k.replace('val_', 'epoch_'): v for k, v in logs.items() if k.startswith('val_')}
        for name, value in val_logs.items():
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value.item()
            summary_value.tag = name
            self.val_writer.add_summary(summary, epoch)

        # added by Zhiwei March-2019
        if self.write_images:
            if self.selected_layers is None:
                print('no layers were selected!')
            else:
                for layer in self.model.layers:
                    if layer.name in self.selected_layers:
                        outTensor = self.model.get_layer(layer.name).output
                        #print( self.sess.run(tf.shape(outTensor) ))
                        img = tf.cast(255*outTensor[15,np.newaxis, :, :, 2, 0, np.newaxis], tf.uint8)
                        tf.summary.image(layer.name, img)
                        image_merge = tf.summary.merge_all()

                        tensors = (self.model.inputs + self.model.targets + self.model.sample_weights)
                        val_data = self.validation_data
                        val_size = val_data[0].shape[0]
                        i = 0
                        while i < val_size:
                            step = min(self.batch_size, val_size - i)
                            if self.model.uses_learning_phase:
                                # do not slice the learning phase
                                batch_val = [x[i:i + step] for x in val_data[:-1]]
                                batch_val.append(val_data[-1])
                            else:
                                batch_val = [x[i:i + step] for x in val_data]
                            # assert len(batch_val) == len(tensors)
                            feed_dict = dict(zip(tensors, batch_val))
                            image_result = self.sess.run([image_merge], feed_dict=feed_dict)
                            summary_img = image_result[0]
                            self.val_writer.add_summary(summary_img, epoch)

        self.val_writer.flush()

        # Pass the remaining logs to `TensorBoard.on_epoch_end`
        logs = {k: v for k, v in logs.items() if not k.startswith('val_')}
        super(TrainValTensorBoard, self).on_epoch_end(epoch, logs)

    def on_train_end(self, logs=None):
        super(TrainValTensorBoard, self).on_train_end(logs)
        self.val_writer.close()