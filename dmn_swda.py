import numpy as np
import tensorflow as tf
from tensorflow.python.ops import rnn_cell
import os


from attn_gru import AttentionGRUCell
from nn import weight, bias, dropout
from data_utils import minibatches, pad_sequences
from swda_data import   load_train_data
from tensorflow.python.util import nest
from tensorflow.python.ops import math_ops
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,4,6'

#os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2'

batch_size = 100
max_sen_len = 40
nwords = 19801
embed_size = 200
hidden_size = 200
nlabels = 36
clip = 5
keep_prob = 0.8
memory_step = 2
proj_size = 100
learning_rate = 0.1
weight_decay = 0.0001
learning_rate_decay = 0.9
num_epochs = 50
train_examples = 400000
dev_examples = 100000
logdir = "train_gau_self_gated"
num_layers = 2
_BIAS_VARIABLE_NAME = "bias"
_WEIGHTS_VARIABLE_NAME = "kernel"

def _linear(args,
            output_size,
            bias,
            bias_initializer=None,
            kernel_initializer=None):
    """Linear map: sum_i(args[i] * W[i]), where W[i] is a variable.

    Args:
        args: a 2D Tensor or a list of 2D, batch x n, Tensors.
        output_size: int, second dimension of W[i].
        bias: boolean, whether to add a bias term or not.
        bias_initializer: starting value to initialize the bias
            (default is all zeros).
        kernel_initializer: starting value to initialize the weight.

    Returns:
        A 2D Tensor with shape [batch x output_size] equal to
        sum_i(args[i] * W[i]), where W[i]s are newly created matrices.

    Raises:
        ValueError: if some of the arguments has unspecified or wrong shape.
    """
    if args is None or (nest.is_sequence(args) and not args):
        raise ValueError("`args` must be specified")
    if not nest.is_sequence(args):
        args = [args]

    # Calculate the total size of arguments on dimension 1.
    total_arg_size = 0
    shapes = [a.get_shape() for a in args]
    for shape in shapes:
        if shape.ndims != 2:
            raise ValueError("linear is expecting 2D arguments: %s" % shapes)
        if shape[1].value is None:
            raise ValueError("linear expects shape[1] to be provided for shape %s, "
                       "but saw %s" % (shape, shape[1]))
        else:
            total_arg_size += shape[1].value

    dtype = [a.dtype for a in args][0]

    # Now the computation.
    scope = tf.get_variable_scope()
    with tf.variable_scope(scope) as outer_scope:
        weights = tf.get_variable(
            _WEIGHTS_VARIABLE_NAME, [total_arg_size, output_size],
            dtype=dtype,
            initializer=kernel_initializer)
        if len(args) == 1:
            res = math_ops.matmul(args[0], weights)
        else:
            res = math_ops.matmul(array_ops.concat(args, 1), weights)
        if not bias:
            return res
        with tf.variable_scope(outer_scope) as inner_scope:
            inner_scope.set_partitioner(None)
            if bias_initializer is None:
                bias_initializer = init_ops.constant_initializer(0.0, dtype=dtype)
            biases = tf.get_variable(
                _BIAS_VARIABLE_NAME, [output_size],
                dtype=dtype,
                initializer=bias_initializer)
        return nn_ops.bias_add(res, biases)


class DMNModel():
    def __init__(self):
        self.global_step = tf.Variable(0, trainable=False)
        
        with tf.variable_scope("placeholder"):
            self.input = tf.placeholder(tf.int32, shape=[batch_size, max_sen_len], name='inputs')  #  [N, W]
            self.labels = tf.placeholder(tf.int32, shape = [batch_size, nlabels], name = "labels")  
            self.position_encoding = tf.placeholder(dtype = tf.float32, shape = [max_sen_len], name = "position_encoding")
            self.source_mask = tf.placeholder(dtype = tf.int32, shape = [batch_size, max_sen_len], name = "source_mask")
            self.is_training = tf.placeholder(tf.bool)
           
        with tf.variable_scope('word_embedding'):
            embedding = tf.get_variable(name = "embedding", 
                                        shape = [nwords, embed_size], 
                                        dtype = tf.float32, 
                                        initializer=tf.random_uniform_initializer(minval = -0.1, maxval = 0.1))
                                        #initializer=tf.contrib.layers.xavier_initializer())
            
            self.inputs = tf.nn.embedding_lookup(embedding, self.input) # [N, W, D]
            noise = tf.random_normal(shape=tf.shape(self.inputs), mean=0.0, stddev=0.05, dtype=tf.float32)

            self.encoder_inputs = tf.cond(self.is_training, lambda : self.inputs + noise, lambda : self.inputs)
        
        with tf.variable_scope('input_fusion'):
            yh
            self.setup_encoder() 
            facts = self.encoder_output # [N, t, D]
            print(facts, "********************************")

        with tf.variable_scope("question"):
            question_vec = tf.get_variable(name = "question_vec", 
                                            shape = [batch_size , 2 * hidden_size], 
                                            dtype = tf.float32,
                                            initializer=tf.random_uniform_initializer(minval = -0.1, maxval = 0.1))
                                            #initializer=tf.contrib.layers.xavier_initializer()) #[N, 2H]

        # Episodic Memory
        with tf.variable_scope('episode'):
            prev_memory = question_vec #[N, 2H]
            for i in range(memory_step):
                print("===> generating episode", i)
                episode = self.generate_episode(prev_memory, question_vec, facts, i) 

                with tf.variable_scope("hop_%d" % i):
                    prev_memory = tf.layers.dense(inputs = tf.concat([prev_memory, episode, question_vec], 1),
                                                 units =  2 * hidden_size, 
                                                 activation = tf.nn.relu)
            
            output = prev_memory #[N, 2H]

        with tf.variable_scope("project"):
            W = tf.get_variable(name = "W", 
                                shape = [2 * hidden_size, proj_size], 
                                dtype = tf.float32, 
                                initializer=tf.random_uniform_initializer(minval = -0.1, maxval = 0.1))
            b = tf.get_variable(name = "b", 
                                shape = [proj_size], 
                                dtype = tf.float32, 
                                initializer=tf.constant_initializer(0))
            
            proj = tf.nn.xw_plus_b(output, W, b)
            proj = tf.nn.relu(proj)
            l2_loss = tf.nn.l2_loss(W)
        

        with tf.variable_scope('logits'):    
            output = dropout(proj, keep_prob, self.is_training)
            logits = tf.layers.dense(tf.concat([output, question_vec], 1), nlabels, activation=None) #[N, nlabels]
            
        with tf.variable_scope('loss'):
            # Cross-Entropy loss
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = self.labels)
            loss = tf.reduce_mean(cross_entropy)
            self.total_loss = loss + weight_decay * l2_loss 

        with tf.variable_scope('accuracy'):
            # Accuracy
            predicts = tf.cast(tf.argmax(logits, 1), tf.int32)
            corrects = tf.equal(predicts, tf.cast(tf.argmax(self.labels, 1), tf.int32))
            self.num_corrects = tf.reduce_sum(tf.cast(corrects, tf.float32))
            self.accuracy = tf.reduce_mean(tf.cast(corrects, tf.float32))

        # Training
        with tf.variable_scope("train_op"):
            lr = tf.train.exponential_decay(learning_rate, self.global_step, train_examples / batch_size, learning_rate_decay)
            optimizer = tf.train.AdagradOptimizer(lr)

            grads, vs = zip(*optimizer.compute_gradients(self.total_loss))
            grads, gnorm = tf.clip_by_global_norm(grads, clip)
            self.opt_op = optimizer.apply_gradients(zip(grads, vs), global_step = self.global_step)
    
    

    def get_attention(self, q_vec, prev_memory, fact_vec, reuse):
        '''
        q_vec.shape -> [N, 2H]
        pre_memory.shape -> [N, 2H]
        fact_vec.shape -> [N, 2H]
        '''
        with tf.variable_scope("attention", reuse = reuse):
            features = [fact_vec * q_vec,
                        fact_vec * prev_memory,
                        tf.abs(fact_vec - q_vec), 
                        tf.abs(fact_vec - prev_memory)] # 4 * [N, 2H]
            feature_vec = tf.concat(features, axis = 1) # [N, 4 * 2H]
            attention = tf.contrib.layers.fully_connected(inputs = feature_vec, 
                                                        num_outputs =  hidden_size,  
                                                        activation_fn = tf.nn.tanh, 
                                                        reuse=reuse, 
                                                        scope="fc1") # [N, H]

            attention = tf.contrib.layers.fully_connected(inputs = attention, 
                                                        num_outputs =  1, 
                                                        activation_fn = None, 
                                                        reuse = reuse, 
                                                        scope = 'fc2')

        return attention # [N, 1]

    def generate_episode(self, memory, q_vec, fact_vecs, hop_index):
        '''
        memory.shape -> [N, 2H]
        q_vec.shape -> [N,2H]
        fact_vecs.shape -> [N, W, 2H]
        '''
        print("####################################", fact_vecs)
        attentions = [tf.squeeze(self.get_attention(q_vec, memory, fv, bool(hop_index) or bool(i)), axis = 1)
                        for i, fv in enumerate(tf.unstack(fact_vecs, axis = 1))] # W * [N]
        attentions = tf.transpose(tf.stack(attentions)) # [N, W]

        
        attentions = tf.nn.softmax(attentions)
        attentions = tf.expand_dims(attentions, axis = -1) # [N, W, 1]

        reuse = True if hop_index > 0 else False

        gru_inputs = tf.concat([fact_vecs, attentions], 2) # [N, W, 2H+1]

        with tf.variable_scope("attention_gru", reuse = reuse):
            fw = AttentionGRUCell(2*hidden_size)
            bw = AttentionGRUCell(2*hidden_size)
            outputs, final_state = tf.nn.bidirectional_dynamic_rnn(fw, bw, gru_inputs, dtype = tf.float32)
            
            episode = final_state[0] + final_state[1]
        return episode # [N, 2H]
    
    def setup_encoder(self):
        self.encoder_cell = rnn_cell.GRUCell(hidden_size)
        with tf.variable_scope("pryamid_encoder"):
            inp = self.encoder_inputs
            
            mask = self.source_mask
            out = None
            for i in range(num_layers):
                with tf.variable_scope("encoder_cell_%d" % i) as scope:
                    srclen = tf.reduce_sum(mask, reduction_indices = 1)
                    fw_cell = tf.nn.rnn_cell.GRUCell(hidden_size)
                    bw_cell = tf.nn.rnn_cell.GRUCell(hidden_size)
                    outputs, _ = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, inp, srclen, dtype = tf.float32) #[N, T, D]
                    out = outputs[0] + outputs[1]
                    dropin, mask = self.downscale(out, mask)
                    inp = dropout(dropin, keep_prob, self.is_training)
            out = tf.concat(outputs, axis = -1)
            # self.encoder_output 实际是facts
            self.encoder_output = out 


    def downscale(self, inp, mask):
        with tf.variable_scope("downscale"):
            inshape = inp.get_shape().as_list()
            batchSize, T, dim = inshape[0], inshape[1], inshape[2]
            inp2d = tf.reshape(inp, [-1, 2*hidden_size]) # [N,T,D] -> [NT/2, 2D]
            out2d = _linear(inp2d, hidden_size, False) # [NT/2, D]
            
            out3d = tf.reshape(out2d, [batchSize, tf.to_int32(T/2), dim]) # [NT/2, D] -> [N, T/2, D]
            out3d.set_shape([None, None, hidden_size]) 
            out = tf.nn.tanh(out3d)

            mask = tf.reshape(mask, [-1, 2]) # [N, T] -> [NT/2, 2]
            mask = tf.cast(mask, tf.bool)
            mask = tf.reduce_any(mask , reduction_indices = 1) # [NT/2 , ]
            mask = tf.to_int32(mask)
            mask = tf.reshape(mask, tf.stack([batchSize, -1])) # [N, T/2]
        return out3d, mask
    

def main():

    data, tags = load_train_data()

    data = np.array(data)
    print(data[10])
    tags = np.array(tags)
    num_examples = len(data)
    shuffle_index = np.random.permutation(num_examples)
    data = data[shuffle_index]
    tags = tags[shuffle_index]
    
    

    train_X, dev_X = data[:train_examples], data[train_examples:train_examples + dev_examples]
    train_Y, dev_Y = tags[:train_examples ], tags[train_examples:train_examples + dev_examples]

    def get_one_hot(labels):
        res = np.zeros([batch_size, nlabels], dtype = np.int32)
        res[np.arange(len(labels)), labels] = 1
        return res
    
    def get_mask(length):
        mask = np.zeros([batch_size, max_sen_len])
        for i, c in enumerate(length):
            mask[i, :c] = 1
        return mask



    position_encoding = 1.0 / np.arange(1, max_sen_len + 1)

    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.5 # maximun alloc gpu50% of MEM
    config.gpu_options.allow_growth = True #allocate dynamically

    with tf.Session(config = config) as sess:
        model = DMNModel()
        sess.run(tf.global_variables_initializer())

        saver = tf.train.Saver()
        writer = tf.summary.FileWriter(logdir, graph=sess.graph)

        for epoch in range(num_epochs):
            # train_step
            for t_sentences, t_labels in minibatches(train_X, train_Y,  batch_size):
                word_ids, sentence_length = pad_sequences(t_sentences, pad_tok = 19800, nlevels = 1)
                t_labels = get_one_hot(t_labels)
                mask = get_mask(sentence_length)
                #position_encoding = positional_encoding(char_ids)

                step, train_loss, train_accuracy, _ = sess.run([model.global_step, model.total_loss, model.accuracy, model.opt_op], 
                    feed_dict = {model.input: word_ids, 
                                model.source_mask: mask,  model.labels: t_labels,
                                model.is_training: True, model.position_encoding:position_encoding})
                print("step = {}, train_loss = {}, train_accuracy = {}".format(step, train_loss, train_accuracy))
                
                train_accuracy_summary = tf.Summary()
                train_accuracy_summary.value.add(tag = 'train_accuracy', simple_value = train_accuracy)
                writer.add_summary(train_accuracy_summary, step)

                train_loss_summary = tf.Summary()
                train_loss_summary.value.add(tag = 'train_loss', simple_value = train_loss)
                writer.add_summary(train_loss_summary, step)

            # validation step   
            loss_dev = []
            acc_dev = []
           
            for d_sentences, d_labels in minibatches( dev_X, dev_Y,  batch_size):
                word_ids, sentence_length = pad_sequences(d_sentences, pad_tok = 0, nlevels = 1)
                d_labels = get_one_hot(d_labels) 
                mask = get_mask(sentence_length)
                #position_encoding = positional_encoding(word_ids)

                dev_loss, dev_accuracy = sess.run([model.total_loss, model.accuracy], 
                    feed_dict = {model.input: word_ids, 
                                model.source_mask: mask,  model.labels: d_labels,
                                model.is_training: False, model.position_encoding:position_encoding })


                
                
                loss_dev.append(dev_loss)
                acc_dev.append(dev_accuracy)


            valid_accuracy = np.mean(np.array(acc_dev))
            
            valid_loss = np.mean(np.array(loss_dev))
        
            dev_accuracy_summary = tf.Summary()
            dev_accuracy_summary.value.add(tag = 'dev_accuracy', simple_value = valid_accuracy)
            writer.add_summary(dev_accuracy_summary, step)

            dev_loss_summary = tf.Summary()
            dev_loss_summary.value.add(tag = 'dev_loss', simple_value = valid_loss)
            writer.add_summary(dev_loss_summary, step)

            print("step = {}, dev_loss = {}, dev_accuray = {}".format(step, valid_loss, valid_accuracy))


if __name__ == "__main__":
    main()
