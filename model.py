import tensorflow as tf
import ops
import utils
import numpy as np
from discriminator import Discriminator
from classifier import Classifier
from generator import Generator
from resnet_classifier import resnetClassifier

REAL_LABEL=0.9
class CycleGAN:
    def __init__(self,
                 batch_size=1,
                 image_size=256,
                 use_lsgan=True,
                 norm='instance',
                 lambda1=10,
                 lambda2=10,
                 learning_rate=2e-4,
                 beta1=0.5,
                 ngf=64
                 ):
        """
        Args:
          batch_size: integer, batch size
          image_size: integer, image size
          lambda1: integer, weight for forward cycle loss (X->Y->X)
          lambda2: integer, weight for backward cycle loss (Y->X->Y)
          use_lsgan: boolean
          norm: 'instance' or 'batch'
          learning_rate: float, initial learning rate for Adam
          beta1: float, momentum term of Adam
          ngf: number of gen filters in first conv layer
        """
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.use_lsgan = use_lsgan
        use_sigmoid = not use_lsgan
        self.batch_size = batch_size
        self.image_size = image_size
        self.learning_rate = learning_rate
        self.beta1 = beta1

        self.is_training = tf.placeholder_with_default(True, shape=[], name='is_training')

        self.G = Generator('G', self.is_training, ngf=ngf, norm=norm, image_size=image_size)
        self.D_Y = Discriminator('D_Y', self.is_training, norm=norm, use_sigmoid=use_sigmoid)

        self.F = Generator('F', self.is_training, norm=norm, image_size=image_size)
        self.D_X = Discriminator('D_X', self.is_training, norm=norm, use_sigmoid=use_sigmoid)

        #self.C = Classifier("C", self.is_training, norm=norm, use_sigmoid=use_sigmoid)
        #self.Cx=Classifier("Cx", self.is_training, norm=norm, use_sigmoid=use_sigmoid)
        self.Cx = resnetClassifier("Cx", 1, self.is_training)
        self.Cy=Classifier("Cy", self.is_training, norm=norm, use_sigmoid=use_sigmoid)
        self.x = tf.placeholder(tf.float32, shape=[batch_size, image_size, image_size, 3])
        self.y = tf.placeholder(tf.float32, shape=[batch_size, image_size, image_size, 3])
        self.x_label = tf.placeholder(tf.int64, [None])  # denotes class label for DMKM
        self.y_label = tf.placeholder(tf.int64, [None])
        # self.fake_x = tf.placeholder(tf.float32, shape=[batch_size, image_size, image_size, 3])
        # self.fake_y = tf.placeholder(tf.float32, shape=[batch_size, image_size, image_size, 3])

        #self.Uy = tf.placeholder(tf.float32, [batch_size, 2])
        #self.Ux2y = tf.placeholder(tf.float32, [batch_size, 1])
        #self.U_fakeY = tf.placeholder(tf.float32, [batch_size, 2])
        #self.Ux2y = tf.placeholder(tf.float32, [batch_size, 1])


        #self.fakeX_label = tf.placeholder(tf.float32, [None])
        #self.fakeY_label = tf.placeholder(tf.float32, [None])


        #self.ClusterX = tf.placeholder(tf.float32, [1, 100])
        #self.ClusterY = tf.placeholder(tf.float32, [2, 100])
        #self.Cluster_fakeX = tf.placeholder(tf.float32, [1, 100])
        #self.Cluster_fakeY = tf.placeholder(tf.float32, [2, 100])

    def model(self):

        x = self.x
        y = self.y

        cycle_loss = self.cycle_consistency_loss(self.G, self.F, x, y)

        # X -> Y
        fake_y = self.G(x)
        G_gan_loss = self.generator_loss(self.D_Y, fake_y,  use_lsgan=self.use_lsgan)
        G_loss = G_gan_loss + cycle_loss
        D_Y_loss = self.discriminator_loss(self.D_Y, y, fake_y, use_lsgan=self.use_lsgan)

        # Y -> X
        fake_x = self.F(y)
        #Disperse_loss = self.disperse_loss(fake_x, self.Cx, self.Uy2x)
        F_gan_loss = self.generator_loss(self.D_X, fake_x,  use_lsgan=self.use_lsgan)
        F_loss = F_gan_loss + cycle_loss# - Disperse_loss
        D_X_loss = self.discriminator_loss(self.D_X, x, fake_x, use_lsgan=self.use_lsgan)

        #features
        f_x, softmax1 = self.Cx(x, '1')
        f_y,softmax2=self.Cy(y,'1')

        f_fakeX,softmax3=self.Cx(fake_x,'2')

        f_fakeY,softmax4=self.Cy(fake_y,'2')
        #print('f_x:',f_x)
        fake_x_pre=tf.argmax(f_fakeX,1)
        x_pre=tf.argmax(f_x,1)
        y_pre = tf.argmax(f_y, 1)
        fake_y_pre=tf.argmax(f_fakeY,1)
        print('pre:',x_pre)
        fake_y_correct=tf.equal(fake_y_pre, self.x_label)
        fake_x_correct = tf.equal(fake_x_pre, self.y_label)
        y_correct=tf.equal(y_pre, self.y_label)
        x_correct=tf.equal(x_pre,self.x_label)
        print('correct:', y_correct)


        # teacher-net
        teacher_loss_x = self.teacher_loss(f_x, label=self.x_label)
        teacher_loss_fakeX = self.teacher_loss(f_fakeX, label=self.y_label)
        teacher_loss=teacher_loss_x+teacher_loss_fakeX

        # student-net
        student_loss_y = self.student_loss(f_y, label=self.y_label)
        student_loss_fakeY = self.student_loss(f_fakeY, label=self.x_label)
        student_loss=student_loss_y+student_loss_fakeY

        teach_loss=teacher_loss+student_loss

        #Learning-step
        ts_loss=self.learning_loss(f_x,f_fakeY)
        st_loss=self.learning_loss(f_y,f_fakeX)
        learning_loss=ts_loss+st_loss+teach_loss



        # return G_loss, D_Y_loss, F_loss, D_X_loss, teacher_loss, student_loss, learning_loss, x_correct,y_correct,fake_y_correct
        return G_loss, D_Y_loss, F_loss, D_X_loss, teacher_loss, student_loss, learning_loss, \
               x_correct,y_correct,fake_x_correct,softmax3,fake_x_pre,f_fakeX, fake_x, fake_y


        #return softmax3,fake_y_pre
        #return f_fakeX
    def optimize(self, G_loss, D_Y_loss, F_loss, D_X_loss, teacher_loss, student_loss,learning_loss):
        def make_optimizer(loss, variables=tf.global_variables, name='Adam',starter_learning_rate=self.learning_rate):
            """ Adam optimizer with learning rate 0.0002 for the first 100k steps (~100 epochs)
                and a linearly decaying rate that goes to zero over the next 100k steps
            """
            global_step = tf.Variable(0, trainable=False)
            #starter_learning_rate = self.learning_rate
            end_learning_rate = 0.0
            start_decay_step = 100000
            decay_steps = 100000
            beta1 = self.beta1
            learning_rate = (
                tf.where(
                    tf.greater_equal(global_step, start_decay_step),
                    tf.train.polynomial_decay(starter_learning_rate, global_step - start_decay_step,
                                              decay_steps, end_learning_rate,
                                              power=1.0),
                    starter_learning_rate
                )
            )

            tf.summary.scalar('learning_rate/{}'.format(name), learning_rate)

            learning_step = (
                tf.train.AdamOptimizer(learning_rate, beta1=beta1, name=name)
                    .minimize(loss, global_step=global_step, var_list=variables)
            )
            return learning_step

        G_optimizer = make_optimizer(G_loss, self.G.variables, name='Adam_G')
        D_Y_optimizer = make_optimizer(D_Y_loss, self.D_Y.variables, name='Adam_D_Y')
        F_optimizer = make_optimizer(F_loss, self.F.variables, name='Adam_F')
        D_X_optimizer = make_optimizer(D_X_loss, self.D_X.variables, name='Adam_D_X')
        #learning_optimizer = make_optimizer(learning_loss, name='Adam_learn', starter_learning_rate=self.learning_rate)
        teacher_optimizer = make_optimizer(teacher_loss, self.Cx.variables, name='Adam_teacher_loss', starter_learning_rate=2e-6)
        student_optimizer=make_optimizer(student_loss,self.Cy.variables,name='Adam_student_loss',starter_learning_rate=2e-6)
        learningCx_optimizer = make_optimizer(learning_loss, self.Cx.variables, name='Adam_Cx',starter_learning_rate=2e-6)
        learningCy_optimizer = make_optimizer(learning_loss, self.Cy.variables, name='Adam_Cy',starter_learning_rate=2e-6)

        with tf.control_dependencies([G_optimizer, D_Y_optimizer, F_optimizer, D_X_optimizer, teacher_optimizer,student_optimizer,
                                    learningCx_optimizer,learningCy_optimizer]):
        #with tf.control_dependencies(
         #           [G_optimizer, D_Y_optimizer, F_optimizer, D_X_optimizer, learning_optimizer]):
            return tf.no_op(name='optimizers')

    def discriminator_loss(self, D, y, fake_y, use_lsgan=True):
        """ Note: default: D(y).shape == (batch_size,5,5,1),
                           fake_buffer_size=50, batch_size=1
        Args:
          G: generator object
          D: discriminator object
          y: 4D tensor (batch_size, image_size, image_size, 3)
        Returns:
          loss: scalar
        """
        if use_lsgan:
            # use mean squared error
            #error_real = tf.reduce_mean(tf.squared_difference(D(y), label))
            error_real = tf.reduce_mean(tf.squared_difference(D(y), REAL_LABEL))
            error_fake = tf.reduce_mean(tf.square(D(fake_y)))
        else:
            # use cross entropy
            error_real = -tf.reduce_mean(ops.safe_log(D(y)))
            error_fake = -tf.reduce_mean(ops.safe_log(1 - D(fake_y)))
        loss = (error_real + error_fake) / 2
        return loss

    def generator_loss(self, D, fake_y, use_lsgan=True):
        """
        fool discriminator into believing that G(x) is real
        """
        if use_lsgan:
            # use mean squared error
            loss = tf.reduce_mean(tf.squared_difference(D(fake_y), REAL_LABEL))
        else:
            # heuristic, non-saturating loss
            loss = -tf.reduce_mean(ops.safe_log(D(fake_y))) / 2
        return loss

    def cycle_consistency_loss(self, G, F, x, y):
        """ cycle consistency loss (L1 norm)
        """
        forward_loss = tf.reduce_mean(tf.abs(F(G(x)) - x))
        backward_loss = tf.reduce_mean(tf.abs(G(F(y)) - y))
        loss = self.lambda1 * forward_loss + self.lambda2 * backward_loss
        return loss

    def teacher_loss(self, result,label):
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=result, labels=label)
        loss = tf.reduce_mean(cross_entropy, name="loss")
        return loss
    def student_loss(self,result,label):
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=result, labels=label)
        loss = tf.reduce_mean(cross_entropy)
        return loss

    def learning_loss(self, teacher_out, student_out):
        loss = tf.reduce_mean(tf.squared_difference(teacher_out, student_out))
        return loss