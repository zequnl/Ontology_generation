import torch
#torch.cuda.set_device(1)
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.autograd import Variable
import numpy as np
import math
from model.common_layer import EncoderLayer, DecoderLayer, MultiHeadAttention, Conv, PositionwiseFeedForward, LayerNorm , _gen_bias_mask ,_gen_timing_signal, share_embedding, LabelSmoothing, NoamOpt, _get_attn_subsequent_mask,  get_input_from_batch, get_output_from_batch, get_input_by_index
from utils import config
import random
#from numpy import random
import os
import pprint
from tqdm import tqdm
pp = pprint.PrettyPrinter(indent=1)
import os
import time

random.seed(123)
torch.manual_seed(123)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(123)

class SoftDotAttention(nn.Module):
    """Soft Dot Attention.
    Ref: http://www.aclweb.org/anthology/D15-1166
    Adapted from PyTorch OPEN NMT.
    """

    def __init__(self, dim):
        """Initialize layer."""
        super(SoftDotAttention, self).__init__()
        self.linear_in = nn.Linear(dim, dim, bias=False)
        self.sm = nn.Softmax()
        self.linear_out = nn.Linear(dim * 2, dim, bias=False)
        self.tanh = nn.Tanh()
        self.mask = None

    def forward(self, input, context):
        """Propogate input through the network.
        input: batch x dim
        context: batch x sourceL x dim
        """
        target = self.linear_in(input).unsqueeze(2)  # batch x dim x 1

        # Get attention
        attn = torch.bmm(context, target).squeeze(2)  # batch x sourceL
        attn = self.sm(attn)
        attn3 = attn.view(attn.size(0), 1, attn.size(1))  # batch x 1 x sourceL

        weighted_context = torch.bmm(attn3, context).squeeze(1)  # batch x dim
        h_tilde = torch.cat((weighted_context, input), 1)

        h_tilde = self.tanh(self.linear_out(h_tilde))

        return h_tilde, attn


class LSTMAttentionDot(nn.Module):
    r"""A long short-term memory (LSTM) cell with attention."""

    def __init__(self, input_size, hidden_size, batch_first=True):
        """Initialize params."""
        super(LSTMAttentionDot, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = 1
        self.batch_first = batch_first

        self.input_weights = nn.Linear(input_size, 4 * hidden_size)
        self.hidden_weights = nn.Linear(hidden_size, 4 * hidden_size)

        self.attention_layer = SoftDotAttention(hidden_size)

    def forward(self, input, hidden, ctx, ctx_mask=None):
        """Propogate input through the network."""
        def recurrence(input, hidden):
            """Recurrence helper."""
            hx, cx = hidden  # n_b x hidden_dim
            gates = self.input_weights(input) + \
                self.hidden_weights(hx)
            ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

            ingate = F.sigmoid(ingate)
            forgetgate = F.sigmoid(forgetgate)
            cellgate = F.tanh(cellgate)
            outgate = F.sigmoid(outgate)

            cy = (forgetgate * cx) + (ingate * cellgate)
            hy = outgate * F.tanh(cy)  # n_b x hidden_dim
            h_tilde, alpha = self.attention_layer(hy, ctx.transpose(0, 1))

            return h_tilde, cy

        if self.batch_first:
            input = input.transpose(0, 1)

        output = []
        steps = range(input.size(0))
        for i in steps:
            hidden = recurrence(input[i], hidden)
            if isinstance(hidden, tuple):
                output.append(hidden[0])
            else:
                output.append(hidden)
            #output.append(isinstance(hidden, tuple) and hidden[0] or hidden)
            #output.append(isinstance(hidden, tuple) and hidden[0] or hidden)

        output = torch.cat(output, 0).view(input.size(0), *output[0].size())

        if self.batch_first:
            output = output.transpose(0, 1)

        return output, hidden

class Generator(nn.Module):
    "Define standard linear + softmax generation step."
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x, attn_dist=None, enc_batch_extend_vocab=None, extra_zeros=None, temp=1, beam_search=False):
        logit = self.proj(x)
        return F.log_softmax(logit,dim=-1)

class R_MLP(nn.Module):
    def __init__(self, input_size, latent_dim):
        super(R_MLP, self).__init__()
        self.input_size = input_size
        self.latent_dim = latent_dim
        self.hidden_layer = nn.Linear(input_size, latent_dim)
        self.hidden_mu = nn.Linear(latent_dim, latent_dim)
        self.hidden_var = nn.Linear(latent_dim, latent_dim)
        self.mu = nn.Linear(latent_dim, latent_dim)
        self.var = nn.Linear(latent_dim, latent_dim)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.relu(self.hidden_layer(x))
        x_hmu = self.relu(self.hidden_mu(x))
        x_hvar = self.relu(self.hidden_var(x))
        x_mu = self.tanh(self.mu(x_hmu))
        x_var = self.tanh(self.var(x_hvar))
        return x_mu + torch.exp(x_var / 2.) * torch.randn(x_mu.size()).cuda(), x_mu, x_var
    
class P_MLP(nn.Module):
    def __init__(self, input_size, latent_dim):
        super(P_MLP, self).__init__()
        self.input_size = input_size
        self.latent_dim = latent_dim
        self.hidden_layer = nn.Linear(input_size, int(0.62 * latent_dim))
        self.hidden_mu = nn.Linear(int(0.62 * latent_dim), int(0.77 * latent_dim))
        self.hidden_var = nn.Linear(int(0.62 * latent_dim), int(0.77 * latent_dim))
        self.mu = nn.Linear(int(0.77 * latent_dim), latent_dim)
        self.var = nn.Linear(int(0.77 * latent_dim), latent_dim)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.relu(self.hidden_layer(x))
        x_hmu = self.relu(self.hidden_mu(x))
        x_hvar = self.relu(self.hidden_var(x))
        x_mu = self.tanh(self.mu(x_hmu))
        x_var = self.tanh(self.var(x_hvar))
        return x_mu + torch.exp(x_var / 2.) * torch.randn(x_mu.size()).cuda(), x_mu, x_var

class GraphSAGE(nn.Module):
    def __init__(self, feature_dim, embed_dim1, embed_dim2, adj_lists, num_sample=10, gcn=True, cuda=True, feature_transform=False):
        super(GraphSAGE, self).__init__()
        self.adj_lists = adj_lists
        self.weight2 = nn.Parameter(
                torch.FloatTensor(embed_dim1, feature_dim).cuda())
        self.weight1 = nn.Parameter(
                torch.FloatTensor(embed_dim2, embed_dim1).cuda())
        self.num_sample = num_sample
        self.gcn = gcn
        self.cuda = cuda
        init.xavier_uniform_(self.weight1)
        init.xavier_uniform_(self.weight2)
    def sample_neighbors(self, nodes, num_sample):
        to_neighs = [self.adj_lists[int(node)] for node in nodes]
        _set = set
        if not num_sample is None:
            _sample = random.sample
            samp_neighs = [_set(_sample(to_neigh, 
                            num_sample
                            )) if len(to_neigh) >= num_sample else to_neigh for to_neigh in to_neighs]
        else:
            samp_neighs = to_neighs
        if self.gcn:
            samp_neighs = [samp_neigh | set([nodes[i]]) for i, samp_neigh in enumerate(samp_neighs)]
        unique_nodes_list = list(set.union(*samp_neighs))
        unique_nodes = {n:i for i,n in enumerate(unique_nodes_list)}
        return samp_neighs, unique_nodes_list, unique_nodes
    
    def get_mask(self, samp_neighs, unique_nodes):
        mask = Variable(torch.zeros(len(samp_neighs), len(unique_nodes)))
        column_indices = [unique_nodes[n] for samp_neigh in samp_neighs for n in samp_neigh]   
        row_indices = [i for i in range(len(samp_neighs)) for j in range(len(samp_neighs[i]))]
        mask[row_indices, column_indices] = 1
        if self.cuda:
            mask = mask.cuda()
        num_neigh = mask.sum(1, keepdim=True)
        mask = mask.div(num_neigh)
        return mask
    def forward(self, nodes, features, samp_neighs, unique_nodes_list, unique_nodes, samp_neighs2, unique_nodes_list2, unique_nodes2):
        #samp_neighs, unique_nodes_list, unique_nodes = self.sample_neighbors(nodes, self.num_sample)
        #samp_neighs2, unique_nodes_list2, unique_nodes2 = self.sample_neighbors(unique_nodes_list, self.num_sample)  
        embed_matrix2 = features
        mask2 = self.get_mask(samp_neighs2, unique_nodes2)
        #print(unique_nodes_list2)
        #print("masks", mask2.size())
        #print("features", features.size())
        to_feats2 = mask2.mm(embed_matrix2)
        #print("features2", to_feats2)
        combined2 = to_feats2
        combined2 = F.relu(self.weight2.mm(combined2.t()))
        #print("feature2_trans", combined2)
        mask1 = self.get_mask(samp_neighs, unique_nodes)
        embed_matrix1 = combined2
        to_feats1 = mask1.mm(embed_matrix1.t())
        combined1 = to_feats1
        combined1 = F.relu(self.weight1.mm(combined1.t()))
        #print("return", combined1.size())
        return combined1.t()
        
class VAE(nn.Module):
    def __init__(self, vocab, adj_lists, num_sample, model_file_path=None, is_eval=False, load_optim=False):
        super(VAE, self).__init__()
        self.vocab = vocab
        self.vocab_size = vocab.n_words

        self.embedding = share_embedding(self.vocab,config.preptrained)
        self.encoder = nn.LSTM(config.emb_dim, config.hidden_dim, config.hop, bidirectional=False, batch_first=True,
                               dropout=0.2)
        self.encoder_r = nn.LSTM(config.emb_dim, config.hidden_dim, config.hop, bidirectional=False, batch_first=True,
                               dropout=0.2)
        self.num_sample = num_sample
        self.graphsage = GraphSAGE(config.hidden_dim, config.hidden_dim, config.hidden_dim, adj_lists, num_sample=self.num_sample, gcn=True, cuda=config.USE_CUDA)
        self.represent = R_MLP(2 * config.hidden_dim, 68)
        self.prior = P_MLP(config.hidden_dim, 68)
        self.mlp_b = nn.Linear(config.hidden_dim + 68, self.vocab_size)
        self.encoder2decoder = nn.Linear(
            config.hidden_dim + 68,
            config.hidden_dim)
        self.decoder = LSTMAttentionDot(config.emb_dim, config.hidden_dim, batch_first=True)   
        self.generator = Generator(config.hidden_dim,self.vocab_size)

        if config.weight_sharing:
            # Share the weight matrix between target word embedding & the final logit dense layer
            self.generator.proj.weight = self.embedding.weight

        self.criterion = nn.NLLLoss(ignore_index=config.PAD_idx)
        if (config.label_smoothing):
            self.criterion = LabelSmoothing(size=self.vocab_size, padding_idx=config.PAD_idx, smoothing=0.1)
            self.criterion_ppl = nn.NLLLoss(ignore_index=config.PAD_idx)
        if is_eval:
            self.encoder = self.encoder.eval()
            self.encoder_r = self.encoder_r.eval()
            self.graphsage = self.graphsage.eval()
            self.represent = self.represent.eval()
            self.prior = self.prior.eval()
            self.mlp_b = self.mlp_b.eval()
            self.encoder2decoder = self.encoder2decoder.eval()
            self.decoder = self.decoder.eval()
            self.generator = self.generator.eval()
            self.embedding = self.embedding.eval()

    
        self.optimizer = torch.optim.Adam(self.parameters(), lr=config.lr)
        if(config.noam):
            self.optimizer = NoamOpt(config.hidden_dim, 1, 4000, torch.optim.Adam(self.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
        if config.use_sgd:
            self.optimizer = torch.optim.SGD(self.parameters(), lr=config.lr)
        if model_file_path is not None:
            print("loading weights")
            state = torch.load(model_file_path, map_location= lambda storage, location: storage)
            print("LOSS",state['current_loss'])
            self.encoder.load_state_dict(state['encoder_state_dict'])
            self.encoder_r.load_state_dict(state['encoder_r_state_dict'])
            self.graphsage.load_state_dict(state['graphsage_state_dict'])
            self.represent.load_state_dict(state['represent_state_dict'])
            self.prior.load_state_dict(state['prior_state_dict'])
            self.mlp_b.load_state_dict(state['mlp_b_state_dict'])
            self.encoder2decoder.load_state_dict(state['encoder2decoder_state_dict'])
            self.decoder.load_state_dict(state['decoder_state_dict'])
            self.generator.load_state_dict(state['generator_dict'])
            self.embedding.load_state_dict(state['embedding_dict'])
            if (load_optim):
                self.optimizer.load_state_dict(state['optimizer'])

        if (config.USE_CUDA):
            self.encoder = self.encoder.cuda()
            self.encoder_r = self.encoder_r.cuda()
            self.represent = self.represent.cuda()
            self.prior = self.prior.cuda()
            self.mlp_b = self.mlp_b.cuda()
            self.encoder2decoder = self.encoder2decoder.cuda()
            self.decoder = self.decoder.cuda()
            self.generator = self.generator.cuda()
            self.criterion = self.criterion.cuda()
            self.embedding = self.embedding.cuda()
        self.model_dir = config.save_path
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        self.best_path = ""
    
    def save_model(self, running_avg_ppl, iter, f1_g,f1_b,ent_g,ent_b, log=False, d="save/paml_model_sim"):
        state = {
            'iter': iter,
            'encoder_state_dict': self.encoder.state_dict(),
            'encoder_r_state_dict': self.encoder_r.state_dict(),
            'graphsage_state_dict': self.graphsage.state_dict(),
            'represent_state_dict': self.represent.state_dict(),
            'prior_state_dict': self.prior.state_dict(),
            'mlp_b_state_dict': self.mlp_b.state_dict(),
            'encoder2decoder_state_dict': self.encoder2decoder.state_dict(),
            'decoder_state_dict': self.decoder.state_dict(),
            'generator_dict': self.generator.state_dict(),
            'embedding_dict': self.embedding.state_dict(),
            #'optimizer': self.optimizer.state_dict(),
            'current_loss': running_avg_ppl
        }
        if log:
            model_save_path = os.path.join(d, 'model_{}_{:.4f}_{:.4f}_{:.4f}_{:.4f}_{:.4f}'.format(iter,running_avg_ppl,f1_g,f1_b,ent_g,ent_b) )
        else:
            model_save_path = os.path.join(self.model_dir, 'model_{}_{:.4f}_{:.4f}_{:.4f}_{:.4f}_{:.4f}'.format(iter,running_avg_ppl,f1_g,f1_b,ent_g,ent_b) )
        self.best_path = model_save_path
        torch.save(state, model_save_path)
    
    def get_state(self, batch):
        """Get cell states and hidden states."""
        batch_size = batch.size(0) \
            if self.encoder.batch_first else batch.size(1)
        h0_encoder = Variable(torch.zeros(
            self.encoder.num_layers,
            batch_size,
            config.hidden_dim
        ), requires_grad=False)
        c0_encoder = Variable(torch.zeros(
            self.encoder.num_layers,
            batch_size,
            config.hidden_dim
        ), requires_grad=False)

        return h0_encoder.cuda(), c0_encoder.cuda()
    
    def get_graph_feature(self, batch):
        enc_batch, _, enc_lens, enc_batch_extend_vocab, extra_zeros, _, _, node_indexs = get_input_from_batch(batch)
        self.h0_encoder, self.c0_encoder = self.get_state(enc_batch)
        src_h, (src_h_t, src_c_t) = self.encoder(
            self.embedding(enc_batch), (self.h0_encoder, self.c0_encoder))
        h_t = src_h_t[-1]
        c_t = src_c_t[-1]
        self.features = h_t
        self.cts = c_t
    
    def train_one_batch(self, batch, data_loader_all, train=True):
        ## pad and other stuff
        enc_batch, _, enc_lens, enc_batch_extend_vocab, extra_zeros, _, _, node_indexs = get_input_from_batch(batch)
        dec_batch, _, _, _, _, node_indexs = get_output_from_batch(batch)
        
        if(config.noam):
            self.optimizer.optimizer.zero_grad()
        else:
            self.optimizer.zero_grad()

        ## Encode
        samp_neighs, unique_nodes_list, unique_nodes = self.graphsage.sample_neighbors(node_indexs, self.num_sample)
        samp_neighs2, unique_nodes_list2, unique_nodes2 = self.graphsage.sample_neighbors(unique_nodes_list, self.num_sample)
        #print(unique_nodes_list2)
        #print(unique_nodes2)
        neigh_enc_batch = get_input_by_index(data_loader_all, unique_nodes_list2)
        #print(neigh_enc_batch)
        self.h0_encoder_n, self.c0_encoder_n = self.get_state(neigh_enc_batch)
        src_h_n, (src_h_t_n, src_c_t_n) = self.encoder(
            self.embedding(neigh_enc_batch), (self.h0_encoder_n, self.c0_encoder_n))
        features = src_h_t_n[-1]
        #print("features", features)
        h_t = self.graphsage(node_indexs, features,samp_neighs, unique_nodes_list, unique_nodes, samp_neighs2, unique_nodes_list2, unique_nodes2)
        #print("graphsage", h_t)
        self.h0_encoder, self.c0_encoder = self.get_state(enc_batch)
        src_h, (src_h_t, src_c_t) = self.encoder(
            self.embedding(enc_batch), (self.h0_encoder, self.c0_encoder))
        c_t = src_c_t[-1]
        #print(c_t)
        self.h0_encoder_r, self.c0_encoder_r = self.get_state(dec_batch)
        src_h_r, (src_h_t_r, src_c_t_r) = self.encoder_r(
            self.embedding(dec_batch), (self.h0_encoder_r, self.c0_encoder_r))
        h_t_r = src_h_t_r[-1]
        c_t_r = src_c_t_r[-1]
        
        #sample and reparameter
        z_sample, mu, var = self.represent(torch.cat((h_t_r, h_t), 1))
        p_z_sample, p_mu, p_var = self.prior(h_t)
        
        # Decode
        decoder_init_state = nn.Tanh()(self.encoder2decoder(torch.cat((z_sample, h_t), 1)))
        
        sos_token = torch.LongTensor([config.SOS_idx] * enc_batch.size(0)).unsqueeze(1)
        if config.USE_CUDA: sos_token = sos_token.cuda()
        dec_batch_shift = torch.cat((sos_token,dec_batch[:, :-1]),1)
        target_embedding = self.embedding(dec_batch_shift)
        ctx = src_h.transpose(0, 1)
        trg_h, (_, _) = self.decoder(
            target_embedding,
            (decoder_init_state, c_t),
            ctx    
        )
        pre_logit = trg_h
        logit = self.generator(pre_logit)
        
        ## loss: NNL if ptr else Cross entropy
        re_loss = self.criterion(logit.contiguous().view(-1, logit.size(-1)), dec_batch.contiguous().view(-1))
        kl_losses = 0.5 * torch.sum(torch.exp(var - p_var) + (mu - p_mu) ** 2 / torch.exp(p_var) - 1. - var + p_var, 1)
        kl_loss = torch.mean(kl_losses)
        latent_logit = self.mlp_b(torch.cat((z_sample, h_t), 1)).unsqueeze(1)
        latent_logit = F.log_softmax(latent_logit,dim=-1)
        latent_logits = latent_logit.repeat(1, logit.size(1), 1)
        bow_loss = self.criterion(latent_logits.contiguous().view(-1, latent_logits.size(-1)), dec_batch.contiguous().view(-1))
        loss = re_loss + 0.48 * kl_loss + bow_loss
        if(train):
            loss.backward()
            self.optimizer.step()
        if(config.label_smoothing): 
            s_loss = self.criterion_ppl(logit.contiguous().view(-1, logit.size(-1)), dec_batch.contiguous().view(-1))
        
        return s_loss.item(), math.exp(min(s_loss.item(), 100)), loss.item(), re_loss.item(), kl_loss.item(), bow_loss.item()