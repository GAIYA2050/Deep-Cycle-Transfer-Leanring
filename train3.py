import tensorflow as tf
from model import CycleGAN
from reader_image import get_train_batch,get_test_batch
from datetime import datetime
import os
import logging
from utils import ImagePool

os.environ["CUDA_VISIBLE_DEVICES"] = '3'
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
tf.flags.DEFINE_string('load_model', None,
                       'folder of saved model that you wish to continue training (e.g. 20170602-1936), default: None')

model='train'  # train or test
def train():
    max_accuracy = 0.98
    learning_loss_set = 4.0
    if FLAGS.load_model is not None:
        checkpoints_dir = "checkpoints/" + FLAGS.load_model.lstrip("checkpoints/")
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
        #G_loss, D_Y_loss, F_loss, D_X_loss, teacher_loss, student_loss, learning_loss, x_correct, y_correct, fake_y_correct, fake_y_pre
        G_loss, D_Y_loss, F_loss, D_X_loss, teacher_loss, student_loss, learning_loss ,x_correct,y_correct,fake_y_correct= cycle_gan.model()
        optimizers = cycle_gan.optimize(G_loss, D_Y_loss, F_loss, D_X_loss, teacher_loss, student_loss, learning_loss)
        summary_op = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(checkpoints_dir, graph)
        saver = tf.train.Saver()

    with tf.Session(graph=graph) as sess:
        if FLAGS.load_model is not None:
            checkpoint = tf.train.get_checkpoint_state(checkpoints_dir)
            meta_graph_path = "checkpoints/20190224-1130/model.ckpt-7792.meta"
            print('meta_graph_path', meta_graph_path)
            restore = tf.train.import_meta_graph(meta_graph_path)
            restore.restore(sess, "checkpoints/20190224-1130/model.ckpt-7792")

            step = 7792
            #meta_graph_path = checkpoint.model_checkpoint_path + ".meta"
            #restore = tf.train.import_meta_graph(meta_graph_path)
            #restore.restore(sess, tf.train.latest_checkpoint(checkpoints_dir))
            #step = int(meta_graph_path.split("-")[2].split(".")[0])
        else:
            sess.run(tf.global_variables_initializer())
            step = 0

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        last_1=0.0
        last_2=0.0
        best_1=0.0
        best_2=0.0

        try:
            while not coord.should_stop():
                #x_image, x_label = get_batch_images(FLAGS.batch_size, FLAGS.image_size, FLAGS.image_size, FLAGS.X)
                #y_image, y_label = get_batch_images(FLAGS.batch_size, FLAGS.image_size, FLAGS.image_size, FLAGS.Y)
                x_image, x_label = get_train_batch("X",FLAGS.batch_size,FLAGS.image_size,FLAGS.image_size,"./dataset/")
                #print('x_label',x_label)
                y_image, y_label = get_train_batch("Y",FLAGS.batch_size,FLAGS.image_size,FLAGS.image_size,"./dataset/")
                #print('y_label', y_label)
                # get previously generated images
               # fake_y_val, fake_x_val = sess.run([fake_y, fake_x],feed_dict={cycle_gan.x: x_image, cycle_gan.y: y_image})

                # train
                _, G_loss_val, D_Y_loss_val, F_loss_val, D_X_loss_val,teacher_loss_eval, student_loss_eval, learning_loss_eval, summary = (
                    sess.run(
                        [optimizers, G_loss, D_Y_loss, F_loss, D_X_loss, teacher_loss, student_loss, learning_loss, summary_op],
                        feed_dict={cycle_gan.x: x_image, cycle_gan.y: y_image,
                                   cycle_gan.x_label:x_label,cycle_gan.y_label:y_label}
                    )
                )

                train_writer.add_summary(summary, step)
                train_writer.flush()


                if  step % 100 ==0:

                    print('-----------Step %d:-------------' % step)
                    print('  G_loss   : {}'.format(G_loss_val))
                    print('  D_Y_loss : {}'.format(D_Y_loss_val))
                    print('  F_loss   : {}'.format(F_loss_val))
                    print('  D_X_loss : {}'.format(D_X_loss_val))
                    print('teacher_loss: {}'.format(teacher_loss_eval))
                    print('student_loss: {}'.format(student_loss_eval))
                    print('learning_loss: {}'.format(learning_loss_eval))

                if step% 100 == 0 and step>=10:
                    print('Now is in testing! Please wait result...')
                    test_images_y, test_labels_y = get_test_batch("Y", FLAGS.image_size, FLAGS.image_size, "./dataset/")
                    fake_y_correct_cout = 0
                    for i in range((len(test_images_y))):
                        y_imgs = []
                        y_lbs = []
                        y_imgs.append(test_images_y[i])
                        y_lbs.append(test_labels_y[i])
                        y_correct_eval, fake_y_correct_eval = (
                            sess.run(
                                [y_correct, fake_y_correct],
                                feed_dict={ cycle_gan.y: y_imgs, cycle_gan.y_label: y_lbs}
                            )
                        )
                        if fake_y_correct_eval:
                            fake_y_correct_cout=fake_y_correct_cout+1

                    print('fake_y_accuracy: {}'.format(fake_y_correct_cout /len(test_labels_y)))





                    # print('Now is in testing! Please wait result...')
                    # #save_path = saver.save(sess, checkpoints_dir + "/model.ckpt", global_step=step)
                    # #print("Model saved in file: %s" % save_path)
                    # test_images_y,test_labels_y= get_test_batch("Y",FLAGS.image_size,FLAGS.image_size,"./dataset/")
                    # test_images_x,test_labels_x= get_test_batch("X",FLAGS.image_size,FLAGS.image_size,"./dataset/")
                    # y_correct_cout=0
                    # fake_y_correct_cout=0
                    # for i in range(min(len(test_images_y),len(test_images_x))):
                    #     y_imgs=[]
                    #     y_lbs=[]
                    #     y_imgs.append(test_images_y[i])
                    #     y_lbs.append(test_labels_y[i])
                    #     x_imgs=[]
                    #     x_lbs=[]
                    #     x_imgs.append(test_images_x[i])
                    #     x_lbs.append(test_labels_x[i])
                    #     y_correct_eval,fake_y_correct_eval = (
                    #         sess.run(
                    #             [y_correct,fake_y_correct],
                    #             feed_dict={cycle_gan.x: x_imgs, cycle_gan.y: y_imgs,
                    #                        cycle_gan.x_label: x_lbs,cycle_gan.y_label: y_lbs}
                    #         )
                    #     )
                    #     #print('y_correct_eval', y_correct_eval)
                    #     #print('y_correct_cout',y_correct_cout)
                    #     #print('fake_y_correct_eval', fake_y_correct_eval)
                    #     #print('fake_y_correct_cout',fake_y_correct_cout)
                    #     #if y_correct_eval[0][0]:
                    #     if y_correct_eval:
                    #         y_correct_cout=y_correct_cout+1
                    #     #if fake_y_correct_eval[0][0]:
                    #     if fake_y_correct_eval:
                    #         fake_y_correct_cout=fake_y_correct_cout+1
                    #
                    #
                    # print('y_accuracy: {}'.format(y_correct_cout/(min(len(test_labels_y),len(test_labels_x)))))
                    # print('fake_y_accuracy: {}'.format(fake_y_correct_cout/(min(len(test_labels_y),len(test_labels_x)))))
                    # y_accuracy_1 = format(y_correct_cout / (min(len(test_labels_y), len(test_labels_x))))
                    # fake_y_accuracy_1 = format(fake_y_correct_cout / (min(len(test_labels_y), len(test_labels_x))))
                    #
                    # #print('test_images_len:',len(test_images_y))
                    # #print('test_labels_len:', len(test_labels_y))



                step += 1

        except KeyboardInterrupt:
            logging.info('Interrupted')
            coord.request_stop()
        except Exception as e:
            coord.request_stop(e)
        finally:
            save_path = saver.save(sess, checkpoints_dir + "/model.ckpt", global_step=step)
            print("Model saved in file: %s" % save_path)
            # When done, ask the threads to stop.
            coord.request_stop()
            coord.join(threads)

def main(unused_argv):
    train()

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    tf.app.run()
