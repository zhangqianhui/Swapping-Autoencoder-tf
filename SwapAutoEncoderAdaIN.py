import tensorflow as tf
import numpy as np
from Dataset import save_images
from tfLib.ops import *
from tfLib.loss import *
from tfLib.advloss import *
import os
import functools

class SAE(object):

    # build model
    def __init__(self, dataset, opt):

        self.dataset = dataset
        self.opt = opt
        # placeholder
        self.x = tf.placeholder(tf.float32,
                                [self.opt.batch_size, self.opt.img_size, self.opt.img_size, self.opt.input_nc])
        #pos
        self.pfake = tf.placeholder(tf.float32, [self.opt.batch_size // 2 * self.opt.crop_n, 4])
        self.preal = tf.placeholder(tf.float32, [self.opt.batch_size // 2 * pow(self.opt.crop_n, 2), 4])
        self.preal2 = tf.placeholder(tf.float32, [self.opt.batch_size // 2 * self.opt.crop_n, 4])

        self.lr_decay = tf.placeholder(tf.float32, None, name='lr_decay')
        self.noise_strength = tf.placeholder(tf.float32, [], name='noise')

    def build_model(self):

        self.x_list = tf.split(self.x, num_or_size_splits=2, axis=0)
        self.x1 = self.x_list[0]
        self.y = self.x_list[1]
        self.sx, self.tx = self.Encoder(self.x1)
        self.sy, self.ty = self.Encoder(self.y)
        self._x = self.G(self.sx, self.tx)
        self._xy = self.G(self.sx, self.ty)

        self.g_logits = self.D(tf.concat([self._x, self._xy], axis=0))
        self.d_logits = self.D(self.x)

        #recon loss
        self.recon_loss = L1(self._x, self.x1)

        d_loss_fun, g_loss_fun = get_adversarial_loss(self.opt.loss_type)

        self.d_gan_loss = d_loss_fun(self.d_logits, self.g_logits)
        self.g_gan_loss = g_loss_fun(self.g_logits)
        self.gp_x_loss, self.logits_x = self.gradient_penalty_just_real(self.x)

        # swapping loss
        self.y_local_list = self.croplocal(self.y, self.preal, num_or_size_splits=pow(self.opt.crop_n, 2))
        self._xy_local_list = self.croplocal(self._xy, self.pfake, num_or_size_splits=self.opt.crop_n)
        self.y2_local_list = self.croplocal(self.y, self.preal2, num_or_size_splits=self.opt.crop_n)

        self.co_fake_logits = self.Co_D(self._xy_local_list, self.y_local_list)
        self.co_real_logits = self.Co_D(self.y2_local_list, self.y_local_list)
        self.co_gan_loss = d_loss_fun(self.co_real_logits, self.co_fake_logits)
        self.g_co_gan_loss = g_loss_fun(self.co_fake_logits)
        self.gp_co_loss = self.gradient_penalty_just_real(self.y2_local_list, self.y_local_list, is_d=False)

        self.co_gan_loss = self.co_gan_loss
        self.g_co_gan_loss = self.g_co_gan_loss

        self.D_loss =self.d_gan_loss + self.opt.lam_gp_d * self.gp_x_loss
        self.G_loss = self.g_gan_loss + self.recon_loss + self.g_co_gan_loss
        self.Co_loss = self.co_gan_loss + self.opt.lam_gp_co * self.gp_co_loss

    def croplocal(self, x, p, num_or_size_splits=8):

        preal_list = tf.split(p, num_or_size_splits=num_or_size_splits, axis=0)
        x_local_list = []
        for i in range(len(preal_list)):
            y_local = self.crop_resize(x, tf.cast(preal_list[i], dtype=tf.float32))
            x_local_list.append(y_local)
        x_local_list = tf.stack(x_local_list, 1)
        _, _, h, w, c = x_local_list.get_shape().as_list()
        x_local_list = tf.reshape(x_local_list, shape=[-1, h, w, c])

        return x_local_list

    def gradient_penalty_just_real(self, x, y=None, is_d=True):

        if is_d:
            discri_logits = self.D(x)
            gradients = tf.gradients(tf.reduce_sum(discri_logits), [x])[0]
            slopes = tf.reduce_sum(tf.square(gradients), [1, 2, 3])
            return 0.5 * tf.reduce_mean(slopes), tf.reduce_sum(discri_logits)
        else:
            discri_logits = self.Co_D(x, y)
            gradients = tf.gradients(tf.reduce_sum(discri_logits), [x])[0]
            slopes = tf.reduce_sum(tf.square(gradients), [1, 2, 3])
            return 0.5 * tf.reduce_mean(slopes)

    def build_test_model(self):

        self.x_list = tf.split(self.x, num_or_size_splits=2, axis=0)
        self.x1 = self.x_list[0]
        self.y = self.x_list[1]
        self.sx, self.tx = self.Encoder(self.x1)
        self.sy, self.ty = self.Encoder(self.y)
        self._x = self.G(self.sx, self.tx)
        self._xy = self.G(self.sx, self.ty)

    def crop_resize(self, input, boxes):
        shape = [int(item) for item in input.shape.as_list()]
        return tf.image.crop_and_resize(input, boxes=boxes, box_ind=list(range(0, int(shape[0]))),
                                        crop_size=[int(shape[1] / 4), int(shape[2] / 4)])

    def train(self):

        self.t_vars = tf.trainable_variables()

        self.d_vars = getTrainVariable(vars=self.t_vars, scope='Discriminator')
        self.g_vars = getTrainVariable(vars=self.t_vars, scope='Generator')
        self.en_vars = getTrainVariable(vars=self.t_vars, scope='Encoder')
        self.co_vars = getTrainVariable(vars=self.t_vars, scope='Co-occurrence')

        assert len(self.t_vars) == len(self.d_vars + self.g_vars + self.en_vars + self.co_vars)

        self.saver = tf.train.Saver()

        opti_D = tf.train.AdamOptimizer(self.opt.lr_d, beta1=self.opt.beta1, beta2=self.opt.beta2). \
            minimize(loss=self.D_loss, var_list=self.d_vars)
        opti_G = tf.train.AdamOptimizer(self.opt.lr_g, beta1=self.opt.beta1, beta2=self.opt.beta2). \
            minimize(loss=self.G_loss, var_list=self.g_vars + self.en_vars)
        opti_Co = tf.train.AdamOptimizer(self.opt.lr_co, beta1=self.opt.beta1, beta2=self.opt.beta2).minimize(
            loss=self.Co_loss, var_list=self.co_vars)

        init = tf.global_variables_initializer()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:

            sess.run(init)
            ckpt = tf.train.get_checkpoint_state(self.opt.checkpoints_dir)
            if ckpt and ckpt.model_checkpoint_path:
                start_step = int(ckpt.model_checkpoint_path.split('model_', 2)[1].split('.', 2)[0])
                self.saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                start_step = 0

            step = start_step
            print("Start read dataset")

            tr_img, te_img = self.dataset.input()
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            _te_img = sess.run(te_img)
            print("Start entering the looping")
            while step <= self.opt.niter:

                if step < self.opt.niter:
                    lr_decay = (self.opt.niter - step) / self.opt.niter
                else:
                    lr_decay = 0

                _tr_img = sess.run(tr_img)
                preal = self.get_pos(self.opt.batch_size // 2 * self.opt.crop_n * self.opt.crop_n)
                pfake = self.get_pos(self.opt.batch_size // 2 * self.opt.crop_n)
                preal2 = self.get_pos(self.opt.batch_size // 2 * self.opt.crop_n)
                # noise_strength = self.opt.initial_noise_factor * \
                #             max(0.0, 1.0 - (step / self.opt.niter) / self.opt.noise_ramp_length) ** 2

                f_d = {self.x: _tr_img,
                       self.pfake: pfake,
                       self.preal: preal,
                       self.preal2: preal2,
                       self.lr_decay: lr_decay,
                       self.noise_strength: 0.0}

                # optimize G
                sess.run(opti_Co, feed_dict=f_d)
                sess.run(opti_D, feed_dict=f_d)
                sess.run(opti_G, feed_dict=f_d)

                if step % 500 == 0:

                    o_loss = sess.run([self.D_loss, self.G_loss, self.Co_loss, self.d_gan_loss,
                            self.g_gan_loss, self.co_gan_loss, self.gp_x_loss, self.gp_co_loss, self.recon_loss], feed_dict=f_d)
                    print("step %d d_loss=%.4f, g_loss=%.4f, co_loss=%.4f, d_gan_loss=%.4f, "
                          "g_gan_loss=%.4f, co_gan_loss=%.4f, gp_x_loss=%.4f, gp_co_loss=%.4f, recon_loss=%.4f, lr_decay=%.4f" % (step,
                          o_loss[0], o_loss[1], o_loss[2], o_loss[3], o_loss[4], o_loss[5], o_loss[6], o_loss[7], o_loss[8], lr_decay))

                if np.mod(step, 500) == 0:

                    tr_o = sess.run([self.x1, self.y, self._x, self._xy, self._xy_local_list, self.y2_local_list], feed_dict=f_d)
                    _tr_o = self.Transpose(np.array([tr_o[0], tr_o[1], tr_o[2], tr_o[3]]))

                    f_d = {self.x: _te_img, self.lr_decay: lr_decay, self.noise_strength: 0}
                    te_o = sess.run([self.x1, self.y, self._x, self._xy], feed_dict=f_d)
                    _te_o = self.Transpose(np.array([te_o[0], te_o[1], te_o[2], te_o[3]]))
                    _local_o = self.Transpose(np.array([tr_o[4], tr_o[5]]))

                    save_images(_tr_o, '{}/{:02d}_tr.jpg'.format(self.opt.sample_dir, step))
                    save_images(_te_o, '{}/{:02d}_te.jpg'.format(self.opt.sample_dir, step))
                    save_images(_local_o, '{}/{:02d}_te_local.jpg'.format(self.opt.sample_dir, step))

                if np.mod(step, self.opt.save_model_freq) == 0 and step != 0:
                    self.saver.save(sess, os.path.join(self.opt.checkpoints_dir, 'model_{:06d}.ckpt'.format(step)))
                step += 1

            save_path = self.saver.save(sess, os.path.join(self.opt.checkpoints_dir, 'model_{:06d}.ckpt'.format(step)))
            coord.request_stop()
            coord.join(threads)

            print("Model saved in file: %s" % save_path)

    def test(self):

        init = tf.global_variables_initializer()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:

            sess.run(init)
            self.saver = tf.train.Saver()
            ckpt = tf.train.get_checkpoint_state(self.opt.checkpoints_dir)
            print('Load checkpoint', ckpt)
            if ckpt and ckpt.model_checkpoint_path:
                self.saver.restore(sess, ckpt.model_checkpoint_path)
                print('Load Succeed!')
            else:
                print('Do not exists any checkpoint,Load Failed!')
                exit()

            _, test_batch_image = self.dataset.input()
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            batch_num = self.opt.test_num // self.opt.batch_size
            for i in range(batch_num):
                f_te_img = sess.run(test_batch_image)
                f_d = {self.x: f_te_img}
                output = sess.run([self.x1, self.y, self._x, self._xy], feed_dict=f_d)
                _te_o = self.Transpose(np.array([output[0], output[1], output[2], output[3]]))
                save_images(_te_o, '{}/{:02d}_o.jpg'.format(self.opt.test_sample_dir, i))

            coord.request_stop()
            coord.join(threads)

    def Transpose(self, list):
        refined_list = np.transpose(np.array(list), axes=[1, 2, 0, 3, 4])
        refined_list = np.reshape(refined_list, [refined_list.shape[0] * refined_list.shape[1],
                                                 refined_list.shape[2] * refined_list.shape[3], -1])
        return refined_list

    def D(self, x):

        n_layers_d = self.opt.n_layers_d
        ndf = self.opt.ndf
        conv2d_first = functools.partial(conv2d, k=1, s=1, output_dim=ndf)
        conv2d_middle = functools.partial(conv2d, k=3, s=1, padding='VALID')
        ful_final1 = functools.partial(fully_connect)
        ful_final2 = functools.partial(fully_connect, output_dim=1)
        ResBlock_ = functools.partial(Resblock, relu_type='lrelu', padding='SAME', ds=True, use_IN=False)
        with tf.variable_scope("Discriminator", reuse=tf.AUTO_REUSE):
            x = conv2d_first(x, output_dim=ndf, scope='conv_first')
            for i in range(n_layers_d):
                c_dim = np.minimum(self.opt.ndf * np.power(2, i + 1), 256)
                x = ResBlock_(x, o_dim=c_dim, scope='r_en{}'.format(i))
            x = lrelu(conv2d_middle(lrelu(x), output_dim=c_dim, scope='conv_middle'))
            x = tf.reshape(x, [-1, np.prod([d.value for d in x.shape[1:]])])
            x = lrelu(ful_final1(x, output_dim=c_dim, scope='ful_final1'))
            x = ful_final2(x, scope='ful_final2')

            return x

    def Co_D(self, x_local, y_local_list):

        ful_s = functools.partial(fully_connect, output_dim=2048)
        ful_m = functools.partial(fully_connect, output_dim=1024, scope='ful_m')
        ful_f = functools.partial(fully_connect, output_dim=1, scope='ful_f')
        with tf.variable_scope("Co-occurrence", reuse=tf.AUTO_REUSE):
            x_local_fp = self.Patch_Encoder(x_local)
            y_local_fp = self.Patch_Encoder(y_local_list)
            _, h, w, c = y_local_fp.get_shape().as_list()
            y_local_fp = tf.reshape(y_local_fp, shape=[-1, self.opt.crop_n, h, w, c])
            y_local_fp = tf.reduce_mean(y_local_fp, axis=1)
            fp = tf.reshape(tf.concat([x_local_fp, y_local_fp], axis=-1),
                            [-1, x_local_fp.shape[-1] + y_local_fp.shape[-1]])

            fp = lrelu(ful_s(fp, scope='ful_s1'))
            fp = lrelu(ful_s(fp, scope='ful_s2'))
            fp = lrelu(ful_m(fp))
            logits = ful_f(fp)

            return logits

    def Patch_Encoder(self, x):

        n_layers_co_d = self.opt.n_layers_co_d
        ncodf = self.opt.ncodf
        conv2d_first = functools.partial(conv2d, k=3, s=1, output_dim=ncodf)
        conv2d_middle = functools.partial(conv2d, k=3, s=1, padding='VALID')
        ResBlockDs = functools.partial(Resblock, relu_type='lrelu', padding='SAME', ds=True, use_IN=False)
        ResBlock = functools.partial(Resblock, relu_type='lrelu', padding='SAME', ds=False, use_IN=False)
        with tf.variable_scope("Co-occurrence", reuse=tf.AUTO_REUSE):
            x = conv2d_first(x, output_dim=ncodf, scope='conv_first')
            for i in range(n_layers_co_d):
                c_dim = [64, 128, 256, 384]
                x = ResBlockDs(x, o_dim=c_dim[i], scope='rds{}'.format(i))
            x = lrelu(ResBlock(x, o_dim=c_dim[-1] * 2, scope='r_1'))
            x = lrelu(conv2d_middle(x, output_dim=c_dim[-1], scope='conv_middle'))

            return x

    def Encoder(self, x_init):

        nef = self.opt.nef
        n_layers_e = self.opt.n_layers_e
        conv2d_first = functools.partial(conv2d, k=1, s=1, output_dim=nef)
        conv2d_final = functools.partial(conv2d, k=1, s=1)
        ful = functools.partial(fully_connect, output_dim=512)
        with tf.variable_scope("Encoder", reuse=tf.AUTO_REUSE):

            x = x_init
            x = conv2d_first(x, scope='conv')
            for i in range(n_layers_e):
                c_dim = np.minimum(self.opt.nef * np.power(2, i + 1), 256)
                x = Resblock(x, o_dim=c_dim, use_IN=False, scope='r_en{}'.format(i))

            #stru
            s = lrelu(conv2d(lrelu(x), output_dim=256, k=1, s=1, scope='conv_s1'))
            s = conv2d_final(s, output_dim=8, scope='conv_s2')

            #texture
            t = lrelu(conv2d(x, output_dim=256, k=1, padding='VALID', scope='conv_t1'))
            t = lrelu(conv2d(t, output_dim=512, k=1, padding='VALID', scope='conv_t2'))
            t = avgpool2d(t, k=t.shape[-2])
            t = ful(tf.squeeze(t, axis=[1,2]), scope='ful_t3')
            return s, t

    def G(self, structure, texture):

        n_layers_g = self.opt.n_layers_g
        conv2d_final = functools.partial(conv2d, k=1, s=1, padding='VALID', output_dim=self.opt.output_nc)
        RAA = functools.partial(Resblock_AdaIn_Affline_layers, style_code=texture)
        with tf.variable_scope("Generator", reuse=tf.AUTO_REUSE):

            s = structure
            for i in range(2):
                c_dim = 128 * (i+1)
                s = RAA(s, o_dim=c_dim, us=False, scope='AdaInAffline_{}'.format(i))

            for i in range(n_layers_g):
                c_dim = [512, 512, 256, 128]
                s = RAA(s, o_dim=c_dim[i], scope='AdaInAfflineD_{}'.format(i))

            s = tf.nn.tanh(conv2d_final(lrelu(s), scope='f'))
            return s

    def get_pos(self, batch_size):

        batch_pos = []
        for i in range(batch_size):
            pos = []
            rate = np.random.uniform(4, 8, size=1)

            wh = self.opt.img_size // rate
            x = np.random.randint(wh // 2, self.opt.img_size - wh //2)
            y = np.random.randint(wh // 2, self.opt.img_size - wh //2)

            center = (x, y)
            scale = center[1] - wh // 2
            down_scale = center[1] + wh // 2
            l1_1 = int(scale)
            u1_1 = int(down_scale)

            scale = center[0] - wh // 2
            down_scale = center[0] + wh // 2
            l1_2 = int(scale)
            u1_2 = int(down_scale)

            pos.append(float(l1_1) / self.opt.img_size)
            pos.append(float(l1_2) / self.opt.img_size)
            pos.append(float(u1_1) / self.opt.img_size)
            pos.append(float(u1_2) / self.opt.img_size)
            batch_pos.append(pos)

        return np.array(batch_pos)

