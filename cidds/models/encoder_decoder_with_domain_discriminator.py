import torch
from torch import nn

from fairseq import utils
from fairseq.models import BaseFairseqModel

EVAL_BLEU_ORDER = 4

class EncoderDecoderWithDomainDisciriminator(BaseFairseqModel):
    def __init__(self, args, nmt):
        super().__init__()
        self.args = args
        self.nmt = nmt
        #Init domain classifier
        src_inner_dim = self.args.encoder_embed_dim
        tgt_inner_dim = self.args.decoder_embed_dim
        self.src_domain_discriminator = FeedForwadLayer(src_inner_dim, 2,
                                                          utils.get_activation_fn(self.args.discriminator_activation_fn),
                                                        self.args.discriminator_dropout)
        self.tgt_domain_discriminator = FeedForwadLayer(tgt_inner_dim, 2,
                                                          utils.get_activation_fn(self.args.discriminator_activation_fn),
                                                        self.args.discriminator_dropout)

    def forward(self, src_tokens, src_lengths, prev_output_tokens):
        decoder_output, extra = self.nmt(src_tokens, src_lengths, prev_output_tokens,features_only=True)
        nmt_output = self.nmt.decoder.output_layer(decoder_output)

        src_output = self.nmt.encoder(src_tokens, src_lengths)
        src_hidden_states = torch.transpose(src_output.encoder_out, 0,1)
        src_hidden_states = torch.sum(src_hidden_states, dim =1).squeeze()
        disc_output = self.src_domain_discriminator(src_hidden_states)

        tgt_hidden_states = torch.sum(decoder_output, dim=1).squeeze()
        tgt_disc_output = self.tgt_domain_discriminator(tgt_hidden_states)
        return {"nmt_output": (nmt_output, extra), "disc_output": disc_output, "tgt_disc_output": tgt_disc_output}



class FeedForwadLayer(nn.Module):
    def __init__(self, input_size, output_size, activation_fn,  dropout_prob=None):
        super(FeedForwadLayer, self).__init__()
        self.layer = nn.Linear(input_size, output_size)
        self.activate_fn = activation_fn
        if dropout_prob is not None:
            self.dropout = nn.Dropout(p=dropout_prob)
        else:
            self.dropout = None

    def forward(self, states):
        hidden_states = self.layer(states)
        if self.dropout:
            hidden_states = self.dropout(hidden_states)
        hidden_states = self.activate_fn(hidden_states)
        return hidden_states
