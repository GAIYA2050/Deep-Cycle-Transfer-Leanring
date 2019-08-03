import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle
from keras.utils import to_categorical
from sklearn import svm, datasets

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from reader_image import get_train_batch,get_test_batch1,get_roc_batch
import tensorflow as tf
from model import CycleGAN
#from reader import Reader
from reader_image import get_train_batch,get_test_batch
from datetime import datetime
import os
import logging
from plot_til.tool import  dense_to_one_hot,jisuan,roc
import utils

os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3'
FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer('batch_size', 1, 'batch size, default: 1')
tf.flags.DEFINE_integer('image_size', 256, 'image size, default: 256')
tf.flags.DEFINE_bool('use_lsgan', True,
                     'use lsgan (mean squared error) or cross entropy loss, default: True')
tf.flags.DEFINE_string('norm', 'instance',
                       '[instance, batch] use instance norm or batch norm, default: instance')
tf.flags.DEFINE_integer('lambda1', 10,
                        'weight for forward cycle loss (X->Y->X), default: 10')
tf.flags.DEFINE_integer('lambda2', 10,
                        'weight for backward cycle loss (Y->X->Y), default: 10')
tf.flags.DEFINE_float('learning_rate', 2e-4,
                      'initial learning rate for Adam, default: 0.0002')
tf.flags.DEFINE_float('beta1', 0.5,
                      'momentum term of Adam, default: 0.5')
tf.flags.DEFINE_float('pool_size', 50,
                      'size of image buffer that stores previously generated images, default: 50')
tf.flags.DEFINE_integer('ngf', 64,
                        'number of gen filters in first conv layer, default: 64')

tf.flags.DEFINE_string('X', './data/image/banana_train',
                       'X tfrecords file for training, default: data/tfrecords/banana.tfrecords')
tf.flags.DEFINE_string('Y', './data/image/toxo_toxo_train',
                       'Y tfrecords file for training, default: data/tfrecords/toxo_toxo.tfrecords')
tf.flags.DEFINE_string('load_model','20190611-1650',
                       'folder of saved model that you wish to continue training (e.g. 20170602-1936), default: None')
tf.flags.DEFINE_integer('classes', 4,
                        'number of classes, default: 4')
def train():
    if FLAGS.load_model is not None:
        checkpoints_dir = "checkpoints/20190611-1650"
    else:
        current_time = datetime.now().strftime("%Y%m%d-%H%M")
        checkpoints_dir = "checkpoints/{}".format(current_time)
        try:
            os.makedirs(checkpoints_dir)
        except os.error:
            pass

    graph = tf.Graph()
    with graph.as_default():
        cycle_gan = CycleGAN(
            #X_train_file=FLAGS.X,
            #Y_train_file=FLAGS.Y,
            batch_size=FLAGS.batch_size,
            image_size=FLAGS.image_size,
            use_lsgan=FLAGS.use_lsgan,
            norm=FLAGS.norm,
            lambda1=FLAGS.lambda1,
            lambda2=FLAGS.lambda2,
            learning_rate=FLAGS.learning_rate,
            beta1=FLAGS.beta1,
            ngf=FLAGS.ngf
        )
        G_loss, D_Y_loss, F_loss, D_X_loss, teacher_loss, student_loss, learning_loss, \
        x_correct, y_correct, fake_x_correct, softmax3, fake_x_pre, f_fakeX, fake_x, fake_y_ = cycle_gan.model()
        # G_loss, D_Y_loss, F_loss, D_X_loss, teacher_loss, student_loss, learning_loss,
        # x_correct,y_correct,fake_y_correct,softmax,fake_y_pre,f_fakeX, fake_x, fake_y= cycle_gan.model()
        #optimizers = cycle_gan.optimize(student_loss)
        summary_op = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(checkpoints_dir, graph)
        saver = tf.train.Saver()

    with tf.Session(graph=graph) as sess:
        if FLAGS.load_model is not None:
            checkpoint = tf.train.get_checkpoint_state(checkpoints_dir)
            #checkpoints_dir = r"C:\Users\Administrator\Desktop\pure_VGG\checkpoints\2018\model.ckpt-31800"
            meta_graph_path = "checkpoints/20190611-1650/model.ckpt-94000.meta"
            restore = tf.train.import_meta_graph(meta_graph_path)
            restore.restore(sess, "checkpoints/20190611-1650/model.ckpt-94000")
            step = 0
        else:
            sess.run(tf.global_variables_initializer())
            step = 0

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        f_yroc = np.array([[ 1.91536183e-05, 9.99966621e-01, 1.17612826e-05, 2.36588585e-06]])
        sum_label = np.array([[0,1,0,0]])
        sum_y_pre3 = np.array([[0, 1, 0, 0]])
        num_classes = FLAGS.classes

        try:
            while not coord.should_stop():


                result_dir = './result'
                plot_dir = os.path.join(result_dir, 'tsne_pca')
                roc_dir = os.path.join(result_dir, 'auc_roc')
                utils.prepare_dir(plot_dir)
                utils.prepare_dir(roc_dir)

                # x_image, x_label, _ = get_test_batch1('X', 100, FLAGS.image_size, FLAGS.image_size,
                #                                                   "./dataset/")
                # y_image, y_label, _ = get_test_batch1('Y', 10, FLAGS.image_size, FLAGS.image_size,
                #                                       "./dataset/")
                y_image, y_label = get_roc_batch(FLAGS.image_size, FLAGS.image_size, "./dataset/Y")
                fake_x_pre4 = []
                sotfmaxour = []
                fake_x_correct_cout = 0
                length3 = len(y_label)
                print('length3',length3)
                features_d = []
                for i in range(length3):

                    # ximgs = []
                    # xlbs = []
                    # ximgs.append(x_image[i])
                    # xlbs.append(x_label[i])

                    yimgs = []
                    ylbs = []
                    yimgs.append(y_image[i])
                    ylbs.append(y_label[i])


                    softmax_fakex,fakex_pre,ffakeX,fake_x_correct_eval = (
                        sess.run(
                            [softmax3,fake_x_pre,f_fakeX,fake_x_correct],
                            feed_dict={cycle_gan.y: yimgs,
                                       cycle_gan.y_label: ylbs,
                                       # cycle_gan.x: ximgs,
                                       # cycle_gan.x_label: xlbs


                                       }
                        )
                    )
                    step += 1
                    features_d.append(ffakeX[0])
                    if fake_x_correct_eval:
                        fake_x_correct_cout = fake_x_correct_cout + 1

                    print('fake_x_correct_eval', fake_x_correct_eval)
                    print('-----------Step %d:-------------' % step)
                    # print('ylbs', ylbs)
                    # print('softmax_fakex',softmax_fakex[0])
                    sotfmaxour.append(softmax_fakex[0])
                    fakex_pre_zhi = fakex_pre[0]
                    fake_x_pre4.append(fakex_pre_zhi)

                # print('y-label',y_label)
                # print('fake_x_pre4', fake_x_pre4)
                # print('sotfmaxour',sotfmaxour)

                print('fake_x_accuracy: {}'.format(fake_x_correct_cout / length3))
                one_hot = dense_to_one_hot(np.array(fake_x_pre4),num_classes = num_classes)
                sum_label = dense_to_one_hot(np.array(y_label),num_classes = num_classes)
                # print('one_hot', one_hot)
                #  accuarcy  f1-score ....
                jisuan(y_label,fake_x_pre4)
                print('sum_label.shape',sum_label.shape)
                print(np.array(sotfmaxour).shape)

                roc(sum_label,sotfmaxour,num_classes,roc_dir)



                # TSNE

                print('len features_d', len(features_d))
                tsne = TSNE(n_components=2, learning_rate=4).fit_transform(features_d)
                pca = PCA().fit_transform(features_d)
                plt.figure(figsize=(12, 6))
                plt.subplot(121)
                plt.scatter(tsne[:, 0], tsne[:, 1], c=y_label)
                plt.subplot(122)
                plt.scatter(pca[:, 0], pca[:, 1], c=y_label)
                plt.colorbar()  # 使用这一句就可以分辨出，颜色对应的类了！神奇啊。

                plt.savefig(os.path.join(plot_dir, 'plot.pdf'))
                exit()




        except KeyboardInterrupt:
            logging.info('Interrupted')
            coord.request_stop()
        except Exception as e:
            coord.request_stop(e)
        finally:
            # save_path = saver.save(sess, checkpoints_dir + "/model.ckpt", global_step=step)
            # print("Model saved in file: %s" % save_path)
            # When done, ask the threads to stop.
            coord.request_stop()
            coord.join(threads)

def main(unused_argv):
    train()

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    tf.app.run()



