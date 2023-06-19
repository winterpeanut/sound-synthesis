#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 19 00:10:14 2023

@author: xinglidongsheng
"""

import tensorflow.compat.v1 as tf
import vggish_slim

# Define the VGGish model
def create_vggish_model(input_shape):
    # Create a placeholder for the input
    input_tensor = tf.placeholder(tf.float32, shape=input_shape)

    # Define the VGGish model architecture
    with tf.variable_scope("vggish"):
        # Set training to False to disable dropout
        vggish_slim.define_vggish_slim(training=False)

        # Retrieve the embeddings layer
        embeddings_tensor = tf.get_default_graph().get_tensor_by_name('vggish/embeddings:0')

    # Create the model
    model = tf.keras.models.Model(inputs=input_tensor, outputs=embeddings_tensor)

    return model

model.summary(verbose=True)