#!/usr/bin/python
import tensorflow as tf

from config import Config
from model import CaptionGenerator
from dataset import prepare_train_data, prepare_eval_data, prepare_test_data

FLAGS = tf.app.flags.FLAGS

tf.flags.DEFINE_string('phase', 'train',
                       'The phase can be train, eval or test')

tf.flags.DEFINE_boolean('load', False,
                        'Turn on to load a pretrained model from either \
                        the latest checkpoint or a specified file')

tf.flags.DEFINE_string('model_file', None,
                       'If sepcified, load a pretrained model from this file')

tf.flags.DEFINE_boolean('load_cnn', False,
                        'Turn on to load a pretrained CNN model')

tf.flags.DEFINE_string('cnn_model_file', './vgg16_no_fc.npy',
                       'The file containing a pretrained CNN model')

tf.flags.DEFINE_boolean('train_cnn', False,
                        'Turn on to train both CNN and RNN. \
                         Otherwise, only RNN is trained')

tf.flags.DEFINE_integer('beam_size', 3,
                        'The size of beam search for caption generation')

def main(argv):
    config = Config()
    config.phase = FLAGS.phase
    config.train_cnn = FLAGS.train_cnn
    config.beam_size = FLAGS.beam_size

    with tf.Session() as sess:
        if FLAGS.phase == 'train':
            # training phase
            data = prepare_train_data(config)
            model = CaptionGenerator(config)
            sess.run(tf.global_variables_initializer())
            if FLAGS.load:
                model.load(sess, FLAGS.model_file)
            if FLAGS.load_cnn:
                model.load_cnn(sess, FLAGS.cnn_model_file)
            tf.get_default_graph().finalize()
            model.train(sess, data)

        elif FLAGS.phase == 'eval':
            # evaluation phase
            coco, data, vocabulary = prepare_eval_data(config)
            model = CaptionGenerator(config)
            model.load(sess, FLAGS.model_file)
            tf.get_default_graph().finalize()
            model.eval(sess, coco, data, vocabulary)

        elif FLAGS.phase == 'test':
            # testing phase
            data, vocabulary = prepare_test_data(config)
            model = CaptionGenerator(config)
            model.load(sess, FLAGS.model_file)
            tf.get_default_graph().finalize()
            model.test(sess, data, vocabulary)

        else:
            # save to tfserving
            from tensorflow.saved_model import builder as saved_model_builder
            from tensorflow.saved_model.signature_def_utils import predict_signature_def
            from tensorflow.saved_model import tag_constants

            data, vocabulary = prepare_test_data(config)

            model = CaptionGenerator(config)
            model.load(sess, FLAGS.model_file)

            writer = tf.summary.FileWriter('logs', sess.graph)
            writer.close()

            builder = saved_model_builder.SavedModelBuilder('./tfserving/models/v1.pb')

            # Create prediction signature to be used by TensorFlow Serving Predict API
            input_tensor = tf.get_default_graph().get_tensor_by_name('Placeholder:0')
            output_tensor = tf.get_default_graph().get_tensor_by_name('Placeholder_4:0')
            tensor_info_x = tf.saved_model.utils.build_tensor_info(input_tensor)
            tensor_info_y = tf.saved_model.utils.build_tensor_info(output_tensor)
            signature = (tf.saved_model.signature_def_utils.build_signature_def(
                                    inputs={'images': tensor_info_x},
                                    outputs={'scores': tensor_info_y},
                                    method_name= tf.saved_model.signature_constants.PREDICT_METHOD_NAME))
            
            # Save the meta graph and the variables
            builder.add_meta_graph_and_variables(sess=sess, tags=[tag_constants.SERVING],
                                                    signature_def_map={"predict": signature})

            builder.save()

if __name__ == '__main__':
    tf.app.run()
