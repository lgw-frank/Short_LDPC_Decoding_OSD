# -*- coding: utf-8 -*-
"""
Created on Fri Nov 12 13:11:35 2021

@author: Administrator
"""
import tensorflow as tf
def get_tfrecords_example(feature, label):
    tfrecords_features = {}
    feat_shape = feature.shape
    tfrecords_features['feature'] = tf.train.Feature(float_list=tf.train.FloatList(value=feature))
    tfrecords_features['label'] = tf.train.Feature(int64_list=tf.train.Int64List(value=label))
    tfrecords_features['shape'] = tf.train.Feature(int64_list=tf.train.Int64List(value=list(feat_shape)))
    return tf.train.Example(features = tf.train.Features(feature = tfrecords_features))
#writing all data to tfrecord file
def make_tfrecord(data, out_filename):
    feats,labels = data
    tfrecord_wrt = tf.io.TFRecordWriter(out_filename)
    ndatas = len(labels)
    for inx in range(ndatas):
        # if not (inx+1)%100:
        #     print(inx+1,"th record is stored with total ", ndatas," records to be stored.")
        exmp = get_tfrecords_example(feats[inx], labels[inx])
        exmp_serial = exmp.SerializeToString()
        tfrecord_wrt.write(exmp_serial)
    tfrecord_wrt.close()