#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 18 22:51:11 2023

@author: xinglidongsheng
"""
import tensorflow.compat.v1 as tf
import vggish_input
import vggish_params
import vggish_postprocess
import vggish_slim
import tf_slim as slim
import vggish_params as params


def get_activations(input_tensor):
    # Load the model and PCA parameters
    vggish_slim.define_vggish_slim(training=False)
    sess = tf.Session()
    vggish_slim.load_vggish_slim_checkpoint(sess, FLAGS.checkpoint)
    features_tensor = sess.graph.get_tensor_by_name(vggish_params.INPUT_TENSOR_NAME)
    embedding_tensor = sess.graph.get_tensor_by_name(vggish_params.OUTPUT_TENSOR_NAME)
    pproc = vggish_postprocess.Postprocessor(FLAGS.pca_params)

    # Get the activations of each layer
    activations = []
    with sess.as_default():
        for i, op in enumerate(sess.graph.get_operations()):
            if op.type == "Relu":
                layer_name = op.name.split("/")[-2]
                layer_activation = sess.run(op.outputs[0], feed_dict={features_tensor: input_tensor})
                activations.append((layer_name, layer_activation))
    return activations


flags = tf.app.flags
flags.DEFINE_string(
        'checkpoint', '/Users/xinglidongsheng/ml/models-master/research/audioset/vggish/vggish_model.ckpt',
        'Path to the VGGish checkpoint file.')
flags.DEFINE_string(
        'pca_params', '/Users/xinglidongsheng/ml/models-master/research/audioset/vggish/vggish_pca_params.npz',
        'Path to the VGGish PCA parameters file.')
FLAGS = flags.FLAGS
def chooselayer(index):

    # Load an example audio clip
    example_wav_file = '/Users/xinglidongsheng/ml/kelletal2018-master/demo_stim/example_1.wav'
    audio_input = vggish_input.wavfile_to_examples(example_wav_file)

    # Get the activations of each layer
    activations = get_activations(audio_input)

    # Print the activations to the console
    for layer_name, layer_activation in activations:
        print(f'{layer_name}: {layer_activation.shape}')

    # Save the activations to a file
    #np.savez('activations.npz', *activations)
    feature = activations[index][1]
    return feature
    