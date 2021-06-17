import numpy as np
from scipy.special import expit, softmax
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import math
from sklearn.preprocessing import OneHotEncoder
from tensorflow_privacy.privacy.analysis.compute_dp_sgd_privacy_lib import compute_dp_sgd_privacy

np.random.seed(37)

# Adapted from https://datascience.oneoffcoder.com/restricted-boltzmann-machine.html
        
def get_weight_matrix(n_visible, n_hidden, mean=0.0, stdev=0.1):
    """
    Initializes a weight matrix specifying the links between the visible and hidden layers.
    Note that the weight will not be of dimension n_visible x n_hidden, but instead,
    n_visible + 1 x n_hidden + 1. The extra dimension is to account for the bias.

    :param n_visible: Number of visible nodes.
    :param n_hidden: Number of hidden nodes.
    :param mean: Mean of sampling distribution. Default value is 0.0.
    :param stdev: Standard deviation of sampling distribution. Default value is 0.1.
    :return: Weight matrix.
    """
    W = np.random.normal(loc=mean, scale=stdev, size=n_visible * n_hidden).reshape(n_visible, n_hidden)
    b_v = np.random.normal(loc=mean, scale=stdev, size=n_visible)
    b_h = np.random.normal(loc=mean, scale=stdev, size=n_hidden)
    return W, b_v, b_h

def positive_contrastive_divergence(X, W, b_v, b_h, norm_clip):
    """
    Executes positive contrastive divergence (+CD) phase.

    :param X: Input data to visible layer.
    :param W: Weights between connections of visible and hidden layers.
    :param b_v: biases for the visible layer
    :param b_h: biases for the hidden layer
    :return: A tuple of 1) positive hidden states and 2) positive associations.
    """
    ph_activations = X.dot(W) + b_h
    ph_probs = expit(ph_activations)
    ph_states = ph_probs > np.random.rand(ph_probs.shape[1])
    p_associations_W = X.T.dot(ph_probs)
    p_associations_b_v = X.sum(axis=0)
    p_associations_b_h = ph_probs.sum(axis=0)
    return ph_states, p_associations_W, p_associations_b_v, p_associations_b_h

def negative_contrastive_divergence(ph_states, W, b_v, b_h, cat, norm_clip):
    """
    Executes negative contrastive divergence (-CD) phase.

    :param ph_states: Positive hidden states.
    :param W: Weights between connections of visible and hidden layers.
    :param b_v: biases for the visible layer
    :param b_h: biases for the hidden layer
    :param cat: a list indicating the categorical size for each variable.
        [(C1_l, C1_r), (C2_l, C2_r), ...] such that the visible layer is grouped by
        [C1_l, C1_r), [C2_l, C2_r), ...
    :return: A tuple of 1) negative visible probabilities and 2) negative associations
    """
    nv_activations = ph_states.dot(W.T) + b_v
    nv_states = sample_visible(ph_states, W, b_v, b_h, cat)
    nh_activations = nv_states.dot(W) + b_h
    nh_probs = expit(nh_activations)
    n_associations_W = nv_states.T.dot(nh_probs)
    n_associations_b_v = nv_states.sum(axis=0)
    n_associations_b_h = nh_probs.sum(axis=0)
    return nv_states, n_associations_W, n_associations_b_v, n_associations_b_h

def sample_hidden(v_mat, W, b_v, b_h):
    h_activations = v_mat.dot(W) + b_h
    h_probs = expit(h_activations)
    h_states = h_probs > np.random.rand(h_probs.shape[1])
    return h_states

def sample_visible(h_mat, W, b_v, b_h, cat):
    v_activations = h_mat.dot(W.T) + b_v
    v_states = np.zeros((h_mat.shape[0], b_v.shape[0]))
    for cl, cr in cat:
        for i in range(h_mat.shape[0]):
            v_porbs_cat = softmax(v_activations[i, cl:cr])
            j = np.random.choice(np.arange(cl,cr), p=v_porbs_cat, size=1)
            v_states[i, j] = 1
    return v_states
    

def RBM_sample(n_samples, W, b_v, b_h, cat):
    """
    Performs daydreaming or dreaming to generate samples.

    :param n_samples: The number of sampling to perform.
    :param n_visible: The number of visible nodes.
    :param n_hidden: The number of hidden nodes.
    :param W: Weights between connections of visible and hidden layers.
    :return: A single sample.
    """
    n_visible, n_hidden = W.shape
    samples = []
    h_states = np.random.randint(2, size=n_hidden).reshape(-1, n_hidden)

    for i in range(n_samples):
        v_states = sample_visible(h_states, W, b_v, b_h, cat)
        samples.append(v_states[0])
        h_states = sample_hidden(v_states, W, b_v, b_h)

    samples = np.vstack(samples)
    return samples


def update_weights_dp(n, 
                   p_associations_W, p_associations_b_v, p_associations_b_h, 
                   n_associations_W, n_associations_b_v, n_associations_b_h, 
                   W, b_v, b_h, lr=0.1, norm_clip=1.0, noise_multiplier=1.0):
    """
    Updates the weights after positive and negative associations have been learned.

    :param n: The total number of samples.
    :param p_associations: Positive associations.
    :param n_associations: Negative associations.
    :param W: Weights between connections of visible and hidden layers.
    :param lr: Learning rate. Default value is 0.1.
    :return: The adjusted weights.
    """
    grad_W = p_associations_W - n_associations_W
    grad_b_v = p_associations_b_v - n_associations_b_v
    grad_b_h = p_associations_b_h - n_associations_b_h
    
    norm = np.linalg.norm(grad_W, 'fro')**2 + \
            np.linalg.norm(grad_b_v, 2)**2 + \
            np.linalg.norm(grad_b_h, 2)**2
    norm = np.sqrt(norm)
    print(norm)
    normalizer = max(1, norm/norm_clip)
    normalized_grad_W = grad_W / normalizer
    normalized_grad_b_v = grad_b_v / normalizer
    normalized_grad_b_h = grad_b_h / normalizer
    
    noisy_grad_W = normalized_grad_W + np.random.normal(
        scale=noise_multiplier * norm_clip, size=normalized_grad_W.shape)
    noisy_grad_b_v = normalized_grad_b_v + np.random.normal(
        scale=noise_multiplier * norm_clip, size=normalized_grad_b_v.shape)
    noisy_grad_b_h = normalized_grad_b_h + np.random.normal(
        scale=noise_multiplier * norm_clip, size=normalized_grad_b_h.shape)
    
    
    W_new = W + lr * (noisy_grad_W / float(n))
    b_v_new = b_v + lr * (noisy_grad_b_v / float(n))
    b_h_new = b_h + lr * (noisy_grad_b_h / float(n))
    return W_new, b_v_new, b_h_new

def RBM_train_dp(X, cat, target_epsilon, target_delta, norm_clip=1.0, noise_multiplier=1.0,
                 lr=0.1, n_hidden = None, max_iters=10, batch_size=64, shuffle=True):
    """
    Trains the RBM model.

    :param X: The data matrix. It is one-hot encoded.
    :param cat: a list indicating the categorical size for each variable.
        [(C1_l, C1_r), (C2_l, C2_r), ...] such that the visible layer is grouped by
        [C1_l, C1_r), [C2_l, C2_r), ...
    :param n_hidden: the size of the hidden layer.
    :param max_iters: The maximum number of iterations. Default value is 100.
    :param batch_size: The batch size.
    :param shuffle: A boolean indicating if the data should be shuffled before each training epoch.
    :return: Returns a tuple of 1) the learned weight matrix and 2) the losses (errors) during training iterations.
    """
    b_size = batch_size
    if b_size < 1:
        b_size = 1
    if b_size > X.shape[0]:
        b_size = X.shape[0]
        
    n_visible = X.shape[1]
    if n_hidden is None:
        n_hidden = 0
        for cl, cr in cat:
            n_hidden += math.ceil(math.log2(cr-cl))
    W, b_v, b_h = get_weight_matrix(n_visible, n_hidden, mean=0.0, stdev=0.1)

    n = X.shape[0]
    loss_trace = []
    for epoch in tqdm(range(max_iters)):
        anticipated_epsilon, rdp_order = compute_dp_sgd_privacy(n, batch_size, noise_multiplier, (epoch+1), target_delta)
        print(f'anticipated epsilon for epoch {epoch} is {anticipated_epsilon} with rdp order {rdp_order}.')
        if anticipated_epsilon > target_epsilon:
            break
        S = np.copy(X)
        if shuffle is True:
            np.random.shuffle(S)
        start = 0
        #pbar = tqdm(total=X.shape[0])
        while True:
            stop = start + b_size
            X_batch = S[start:stop, :]
            ph_states, p_associations_W, p_associations_b_v, p_associations_b_h = \
                positive_contrastive_divergence(X_batch, W, b_v, b_h)
            nv_states, n_associations_W, n_associations_b_v, n_associations_b_h = \
                negative_contrastive_divergence(ph_states, W, b_v, b_h, cat)
            W, b_v, b_h = update_weights_dp(X.shape[0], \
                                         p_associations_W, p_associations_b_v, p_associations_b_h, \
                                         n_associations_W, n_associations_b_v, n_associations_b_h, \
                                         W, b_v, b_h, lr, norm_clip, noise_multiplier)
            #print(X_batch)
            #print(ph_states)
            #print(nv_states)
            #print('')

            error = np.sum((X_batch - nv_states) ** 2) / float(b_size)
            t = (epoch, error)
            loss_trace.append(t)

            start = start + b_size
            #pbar.update(b_size)
            if start >= X.shape[0]:
                break
        #pbar.close()
        print(f'Epoch {epoch}, epsilon {anticipated_epsilon}, error {error}')

    loss_df = pd.DataFrame(data=loss_trace, columns=['epoch', 'loss'])
    return W, b_v, b_h, loss_df

def plot_loss(loss, note=''):
    """
    Plots the loss over training iterations.

    :param loss: A dataframe of loss. Should have 2 columns: epoch and loss.
    :param note: A note to add to the title.
    :return: None
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    plt.plot(loss['epoch'], loss['loss'])
    ax.set_title('Loss over time, {}'.format(note))
    ax.set_xlabel(r'epoch')
    ax.set_ylabel(r'loss')
    
    
def RBMCat(real_data, metadata):
    table_name = metadata.get_tables()[0]
    df = real_data[table_name].copy()
    cat_cols = [col for col, dtype in df.dtypes.items() if dtype == 'object']
    df_cat = df[cat_cols]
    X_cat = df_cat.values
    enc = OneHotEncoder(sparse=False)
    X_cat_onehot = enc.fit_transform(X_cat)
    cat = np.r_[0, np.cumsum([len(arr) for arr in enc.categories_])]
    cat = list(zip(cat[:-1], cat[1:]))
    
    target_epsilon=1.0
    target_delta=1e-7
    W, b_v, b_h, loss_df = RBM_train_dp(X_cat_onehot, cat, 
                                     target_epsilon, target_delta, 
                                     norm_clip=1.0, noise_multiplier=1.5, 
                                     lr=0.1, max_iters=20, batch_size=64)
    n_samples = len(df)
    sample = RBM_sample(n_samples, W, b_v, b_h, cat)
    X_cat_inverse = enc.inverse_transform(sample)
    df[cat_cols] = X_cat_inverse
    print(X_cat_inverse)
    return {table_name: df}