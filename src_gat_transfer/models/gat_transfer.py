"""
GAT_Transfer
"""

import numpy as np
import tensorflow as tf

from src_gat_transfer.models import layers
from src_gat_transfer.base.base_model import BaseModel
from src_gat_transfer.base.checkmate import BestCheckpointSaver
from src_gat_transfer.models.metrics import *

tf.random.set_random_seed(1234)

class GAT_inference_model(BaseModel):
    def __init__(self, config, inputs, bias_mat, attn_drop, ffd_drop, activation=tf.nn.elu, residual=False):
        super(GAT_inference_model, self).__init__(config)
        self.config = config
        self.inputs = inputs
        self.bias_mat = bias_mat
        self.attn_drop = attn_drop
        self.ffd_drop = ffd_drop
        self.activation = activation
        self.residual = residual
    def feature_extractor(self,x):
        with tf.variable_scope("generator",reuse=tf.AUTO_REUSE) as scope:
            attns = []
            coefs = []
            for _ in range(self.config.num_heads[0]):
                head, coef = layers.attn_head(x, bias_mat=self.bias_mat,
                                          out_sz=self.config.hid_units[0], activation=self.activation,
                                          in_drop=self.ffd_drop, coef_drop=self.attn_drop, residual=False)
            # head: [1, num_nodes, hid_unit], coef: [1, num_nodes, num_nodes]
                attns.append(head)
                coefs.append(coef)

            h_1 = tf.concat(attns, axis=-1)  # h_1: [1, num_nodes, hid_unit*num_heads[0]]

        # if more transformer layers added, add residual to them
            for i in range(1, len(self.config.hid_units)):
            # print("att layer {}".format(i))
                h_old = h_1
                attns = []
                for _ in range(self.config.num_heads[i]):
                    head, _ = layers.attn_head(h_1, bias_mat=self.bias_mat,
                                           out_sz=self.config.hid_units[i], activation=self.activation,
                                           name_scope='g_attn',
                                           in_drop=self.ffd_drop, coef_drop=self.attn_drop, residual=self.residual)
                    attns.append(head)
                h_1 = tf.concat(attns, axis=-1)
            return h_1
    def feature_extractor_old(self,x):
        attns = []
        coefs = []
        for _ in range(self.config.num_heads[0]):
            head, coef = layers.attn_head(x, bias_mat=self.bias_mat,
                                          out_sz=self.config.hid_units[0], activation=self.activation,
                                          in_drop=self.ffd_drop, coef_drop=self.attn_drop, residual=False)
            # head: [1, num_nodes, hid_unit], coef: [1, num_nodes, num_nodes]
            attns.append(head)
            coefs.append(coef)

        h_1 = tf.concat(attns, axis=-1)  # h_1: [1, num_nodes, hid_unit*num_heads[0]]

        # if more transformer layers added, add residual to them
        for i in range(1, len(self.config.hid_units)):
            # print("att layer {}".format(i))
            h_old = h_1
            attns = []
            for _ in range(self.config.num_heads[i]):
                head, _ = layers.attn_head(h_1, bias_mat=self.bias_mat,
                                           out_sz=self.config.hid_units[i], activation=self.activation,
                                           name_scope='g_attn',
                                           in_drop=self.ffd_drop, coef_drop=self.attn_drop, residual=self.residual)
                attns.append(head)
            h_1 = tf.concat(attns, axis=-1)
        return h_1

    def classifier(self, h_1, reuse=False):
        """
        the input is from the first layer of GAT (multihead attention as encoder)
        """
        with tf.variable_scope("discriminator",reuse=tf.AUTO_REUSE) as scope:
            #if reuse:
                #scope.reuse_variables()
            # the output head number is 1
            # is it necessary to apply multihead attention again as the encoder? or we can use a simple ffd
            out = []
            for i in range(self.config.num_heads[-1]):
                head, _ = layers.attn_head(h_1, bias_mat=self.bias_mat,
                                           out_sz=self.config.num_classes, activation=lambda x: x,
                                           in_drop=self.ffd_drop, coef_drop=self.attn_drop, residual=False)
                out.append(head)
                #print("qqqq",i,h_1.shape)
            logits = tf.add_n(out) / self.config.num_heads[-1]
            return logits

    def forward(self, x):
        features = self.feature_extractor(x)
        logits = self.classifier(features)
        return logits


class GAT(BaseModel):
    def __init__(self, config):
        super(GAT, self).__init__(config)
        self.config = config
        self.batch_size = config.batch_size
        
        with tf.name_scope('input'):
            # individual node features
            self.features = tf.placeholder(dtype=tf.float32,
                                            shape=(self.config.batch_size,
                                                self.config.source_num_nodes + self.config.target_num_nodes,
                                                self.config.feature_size),
                                                name='features')  # input feature
            # adjacency matrix
            self.bias = tf.placeholder(dtype=tf.float32,
                                         shape=(self.config.batch_size,
                                                self.config.source_num_nodes + self.config.target_num_nodes,
                                                self.config.source_num_nodes + self.config.target_num_nodes),
                                                name='adj_bias')  # bias vector from sex adj

            # labels
            self.labels = tf.placeholder(dtype=tf.int32,
                                        shape=(self.config.batch_size,
                                               self.config.source_num_nodes + self.config.target_num_nodes,
                                               self.config.num_classes),
                                             name='labels')  # labels

            # masks to distinguish train and test
            self.masks = tf.placeholder(dtype=tf.int32,
                                        shape=(self.config.batch_size,
                                               self.config.source_num_nodes + self.config.target_num_nodes),
                                         name='masks_source')  # consider the info or not

            # dropout rate
            self.attn_drop = tf.placeholder(dtype=tf.float32, shape=(), name='attn_drop')  # attention dropout
            self.ffd_drop = tf.placeholder(dtype=tf.float32, shape=(), name='ffd_drop')  # feed forward dropout
            self.is_training = tf.placeholder(dtype=tf.bool, shape=())

            self.initializer = tf.random_normal_initializer(stddev=0.1)

        self.build_model()
        self.init_saver()

    def build_model(self):
        if self.config.nonlinearity == 'relu':
            activation = tf.nn.relu

        # declare the generater and discriminator derived from GAT
        clf_model = GAT_inference_model(self.config,self.features,self.bias,self.attn_drop,self.ffd_drop)
        self.feature_extractor = clf_model.feature_extractor
        self.discriminator = clf_model.classifier

        # generator: generate representations for both the source and target using GAT (share parameters between s&t)
        # see if reuse needed!
        # [1, num_nodes, hid_unit * num_heads[0]]
        generator_repres = self.feature_extractor(self.features)
        # split to [1, source_num_nodes, hid], [1, target_num_nodes, hid]
        repre_source, repre_target = tf.split(generator_repres,
                                              [self.config.source_num_nodes, self.config.target_num_nodes], 1)
        '''
        repre_source = tf.squeeze(repre_source,axis=0)
        repre_target = tf.squeeze(repre_target,axis=0)

        con_d = repre_source.shape[0]
        if con_d == 377:
            repre_source = tf.pad(repre_source,[[189,189],[189,189]],'CONSTANT')            repre_target = tf.pad(repre_target,[[189,188],[189,188]],'CONSTANT')        else:
            repre_source = tf.pad(repre_source,[[189,188],[189,188]],'CONSTANT')            repre_target = tf.pad(repre_target,[[189,189],[189,189]],'CONSTANT')

        repre_source = tf.expand_dims(repre_source,axis=0)
        repre_target = tf.expand_dims(repre_target,axis=0)
        '''
        # Gradient Penalty (based on https://github.com/changwoolee/WGAN-GP-tensorflow/blob/master/model.py#116)
        self.epsilon = tf.random_uniform(
            shape=[self.batch_size, 1, 1],
            minval=0.,
            maxval=1.)
        repre_target_ = tf.reduce_mean(repre_target, 1,keepdims=True)
        repre_source_ = tf.reduce_mean(repre_source, 1,keepdims=True)
        X_hat = repre_source + self.epsilon * (repre_target_ - repre_source_)
        X_hat = generator_repres + self.epsilon * (repre_target_ - repre_source_)
        #print("www",repre_source.shape,repre_target.shape,generator_repres.shape)        
        D_X_hat = self.discriminator(X_hat, reuse=True)
        grad_D_X_hat = tf.gradients(D_X_hat, [X_hat])[0]
        red_idx = list(range(1, X_hat.shape.ndims))
        slopes = tf.sqrt(tf.reduce_sum(tf.square(grad_D_X_hat), reduction_indices=red_idx))
        gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)

        # make sure that the same classifier is used when classifying source and target
        d_logits = self.discriminator(generator_repres, reuse=True)
        d_logits_source, d_logits_target = tf.split(d_logits,
                                              [self.config.source_num_nodes, self.config.target_num_nodes], 1)
        #d_logits_target = self.discriminator(repre_target, reuse=False)
        #d_logits_source = self.discriminator(repre_source, reuse=True)

        with tf.name_scope("loss"):
            # reshape anyway to make them comparable and fit for the masked** loss and evaluation
            d_logits_target_reshape = tf.reshape(d_logits, [-1, self.config.num_classes])
            labels_reshape = tf.reshape(self.labels, [-1, self.config.num_classes])
            masks_reshape = tf.reshape(self.masks, [-1])

            ## WGAN Loss
            # generator loss to see how well on the target set
            self.g_loss = masked_softmax_cross_entropy(d_logits_target_reshape, labels_reshape, masks_reshape)
            # discriminator loss to see how well source and target are distinguished
            self.d_loss = tf.reduce_mean(d_logits_target) - tf.reduce_mean(d_logits_source)
            self.d_loss = self.d_loss + 10.0 * gradient_penalty   # 10 can be adjusted

            # other outputs
            self.probs = tf.sigmoid(d_logits_target_reshape)
            self.preds = tf.cast(tf.argmax(d_logits_target_reshape, 1), tf.int32)
            self.accuracy = masked_accuracy(d_logits_target_reshape, labels_reshape, masks_reshape)
            self.f1 = masked_micro_f1(d_logits_target_reshape, labels_reshape, masks_reshape)

            # weights to evaluate the contribution of each neighbor to draw figure
            # self.att_weights_sex = att_weights_sex
            # self.att_weights_venue = att_weights_venue
            # self.W_sex = W_sex
            # self.W_venue = W_venue

        # def training(self, loss, lr, l2_coef):
        with tf.name_scope("train_op"):
            # weight decay
            vars = tf.trainable_variables()
            self.generator_vars = [v for v in vars if 'generator' in v.name]
            self.discriminator_vars = [v for v in vars if 'discriminator' in v.name]
            print("hahah",vars,self.generator_vars,self.discriminator_vars)
            # L2 loss
            lossL2_g = tf.add_n([tf.nn.l2_loss(v) for v in self.generator_vars if v.name not
                               in ['bias', 'gamma', 'b', 'g', 'beta']]) * self.config.l2_coef
            lossL2_d = tf.add_n([tf.nn.l2_loss(v) for v in self.discriminator_vars if v.name not
                               in ['bias', 'gamma', 'b', 'g', 'beta']]) * self.config.l2_coef

            self.g_loss += lossL2_g
            self.d_loss += lossL2_d

            # # optimizer: minimize g loss, minimize d loss
            self.g_optimizer = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate, name='g_opt')\
                .minimize(self.g_loss, var_list=self.generator_vars)
            self.d_optimizer = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate, name='d_opt')\
                .minimize(self.d_loss, var_list=self.discriminator_vars)


    class GAT_inference_model(BaseModel):
        def __init__(self, config, inputs, bias_mat, attn_drop, ffd_drop, activation=tf.nn.elu, residual=False):
            super(GAT_inference_model, self).__init__(config)
            self.config = config
            self.inputs = inputs
            self.bias_mat = bias_mat
            self.attn_drop = attn_drop
            self.ffd_drop = ffd_drop
            self.activation = activation
            self.residual = residual

        def feature_extractor(self):
            attns = []
            coefs = []
            for _ in range(self.config.num_heads[0]):
                head, coef = layers.attn_head(self.inputs, bias_mat=self.bias_mat,
                                              out_sz=self.config.hid_units[0], activation=self.activation,
                                              name_scope='g_attn',
                                              in_drop=self.ffd_drop, coef_drop=self.attn_drop, residual=False)
                # head: [1, num_nodes, hid_unit], coef: [1, num_nodes, num_nodes]
                attns.append(head)
                coefs.append(coef)

            h_1 = tf.concat(attns, axis=-1)  # h_1: [1, num_nodes, hid_unit*num_heads[0]]

            # if more transformer layers added, add residual to them
            for i in range(1, len(self.config.hid_units)):
                # print("att layer {}".format(i))
                h_old = h_1
                attns = []
                for _ in range(self.config.num_heads[i]):
                    head, _ = layers.attn_head(h_1, bias_mat=self.bias_mat,
                                               out_sz=self.config.hid_units[i], activation=self.activation,
                                               name_scope='g_attn',
                                               in_drop=self.ffd_drop, coef_drop=self.attn_drop, residual=self.residual)
                    attns.append(head)
                h_1 = tf.concat(attns, axis=-1)
            return h_1

        def classifier(self, h_1, reuse=False):
            """
            the input is from the first layer of GAT (multihead attention as encoder)
            """
            with tf.variable_scope("discriminator") as scope:
                if reuse:
                    scope.reuse_variables()
                # the output head number is 1
                # is it necessary to apply multihead attention again as the encoder? or we can use a simple ffd
                out = []
                for i in range(self.config.num_heads[-1]):
                    head, _ = layers.attn_head(h_1, bias_mat=self.bias_mat,
                                               out_sz=self.config.num_classes, activation=lambda x: x,
                                               name_scope='d_attn',
                                               in_drop=self.ffd_drop, coef_drop=self.attn_drop, residual=False)
                    out.append(head)

                logits = tf.add_n(out) / self.config.num_heads[-1]
                return logits

        def forward(self, x):
            features = self.feature_extractor(x)
            logits = self.classifier(features)
            return logits

    # def discriminator(self, inputs, ffd_drop, reuse):
    #     with tf.variable_scope("discriminator") as scope:
    #         if reuse:
    #             scope.reuse_variables()
    #         hidden = tf.layers.dense(inputs, self.config.hid_units[0], name='d_hidden')
    #         classify_feats = tf.nn.dropout(hidden, 1 - ffd_drop)
    #         logits = tf.layers.dense(classify_feats, self.config.num_classes, name='d_logits')
    #         return logits
    #
    # def generator(self, inputs, bias_mat, activation=tf.nn.elu, residual=False):
    #     """
    #     multihead attention to receive inputs and output encoded representation for each node
    #     :param inputs:
    #     :param nb_classes:
    #     :param attn_drop:
    #     :param ffd_drop:
    #     :param bias_mat:
    #     :param hid_units:
    #     :param n_heads:
    #     :param activation:
    #     :param residual:
    #     :return:
    #     """
    #     # the hidden vector for each node is the concatenation of multiple attention head
    #     attns = []
    #     coefs = []
    #     for _ in range(self.config.num_heads[0]):
    #         head, coef = layers.attn_head(inputs, bias_mat=bias_mat,
    #                                       out_sz=self.config.hid_units[0], activation=activation,
    #                                       in_drop=self.ffd_drop, coef_drop=self.attn_drop, residual=False)
    #         # head: [1, num_nodes, hid_unit], coef: [1, num_nodes, num_nodes]
    #         attns.append(head)
    #         coefs.append(coef)
    #
    #     h_1 = tf.concat(attns, axis=-1) # h_1: [1, num_nodes, hid_unit*num_heads[0]]
    #
    #     # if more transformer layers added, add residual to them
    #     for i in range(1, len(self.config.hid_units)):
    #         # print("att layer {}".format(i))
    #         h_old = h_1
    #         attns = []
    #         for _ in range(self.config.num_heads[i]):
    #             head, _ = layers.attn_head(h_1, bias_mat=bias_mat,
    #                                           out_sz=self.config.hid_units[i], activation=activation,
    #                                           in_drop=self.ffd_drop, coef_drop=self.attn_drop, residual=residual)
    #             attns.append(head)
    #         h_1 = tf.concat(attns, axis=-1)
    #
    #     # the output head number is 1
    #     out = []
    #     for i in range(self.config.num_heads[-1]):
    #         head, _ = layers.attn_head(h_1, bias_mat=bias_mat,
    #                                     out_sz=self.config.num_classes, activation=lambda x: x,
    #                                     in_drop=self.ffd_drop, coef_drop=self.attn_drop, residual=False)
    #         out.append(head)
    #
    #     logits = tf.add_n(out) / self.config.num_heads[-1]
    #     return logits

    def init_saver(self):
        # here you initialize the tensorflow saver that will be used in saving the checkpoints.
        # save the 5 best ckpts

        best_ckpt_saver = BestCheckpointSaver(
            save_dir=self.config.best_model_dir,
            num_to_keep=1,
            maximize=True
        )
        # self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)
        self.saver = best_ckpt_saver

    @staticmethod
    def preshape(logits, labels, nb_classes):
        new_sh_lab = [-1]
        new_sh_log = [-1, nb_classes]
        log_resh = tf.reshape(logits, new_sh_log)
        lab_resh = tf.reshape(labels, new_sh_lab)
        return log_resh, lab_resh

    @staticmethod
    def confmat(logits, labels):
        preds = tf.argmax(logits, axis=1)
        return tf.confusion_matrix(labels, preds)


