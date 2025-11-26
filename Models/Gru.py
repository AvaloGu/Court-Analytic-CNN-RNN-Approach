import torch
import torch.nn as nn
import torch.nn.functional as F
from Loaders.vocab import POINTSEPARATOR

class GRUCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.Wz = nn.Linear(input_size, hidden_size)
        self.Uz = nn.Linear(hidden_size, hidden_size)
        self.Wr = nn.Linear(input_size, hidden_size)
        self.Ur = nn.Linear(hidden_size, hidden_size)
        self.Wh = nn.Linear(input_size, hidden_size)
        self.Uh = nn.Linear(hidden_size, hidden_size)

    def forward(self, x, h_prev):
        z = torch.sigmoid(self.Wz(x) + self.Uz(h_prev)) # (Hidden,)
        r = torch.sigmoid(self.Wr(x) + self.Ur(h_prev)) # (Hidden,)
        h_tilde = torch.tanh(self.Wh(x) + self.Uh(r * h_prev)) # (Hidden,)
        h = (1 - z) * h_prev + z * h_tilde # (Hidden,)
        return h
    

class NaiveGRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=5):
        super().__init__()
        
        self.hidden_size = hidden_size # Hidden
        self.num_layers = num_layers # Layers
        
        self.cells = nn.ModuleList([GRUCell(input_size if i == 0 else hidden_size, hidden_size) for i in range(num_layers)])
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, h = None):
        num_frames, channels = x.size() # (NumFrames, C)

        if h is None:
            h = [torch.zeros(self.hidden_size).to(x.device) for _ in range(self.num_layers)] # list of (Hidden,) tensors

        out = []
        for t in range(num_frames):
            x_t = x[t, :] # (C,)
            for i in range(self.num_layers):
                h[i] = self.cells[i](x_t, h[i]) # (Hidden,)
                x_t = h[i]
            out.append(h[-1].unsqueeze(0)) # (1, Hidden)
        
        out = torch.cat(out, dim=0) # (NumFrames, Hidden)
        out = self.fc(out) # (NumFrames, 2_classes)
        return out, h # (NumFrames, 2_classes), list of (Hidden,) tensors
    

class EfficientGRUModel(nn.Module):
    r"""
    a wrapper around nn.GRU
    """

    def __init__(self, input_size = 768, hidden_size = 512, output_size = 6, number_layers = 5):
        super().__init__()

        # TODO: nn.gru also has dropout which you might want to use
        self.gru = nn.GRU(input_size, hidden_size, num_layers=number_layers, batch_first=False, bidirectional=True)
        self.ln = nn.LayerNorm(hidden_size*2)
        self.fc = nn.Linear(2*hidden_size, output_size)

    def forward(self, x, h=None):
        # x: (NumFrames, C) -> add batch dim at dim=1
        x = x.unsqueeze(1) # (NumFrames, 1, C)

        if h is None:
            h0 = None
        else:
            # already (2*Layers, 1, Hidden)
            h0 = h.to(x.device)

        y, hn = self.gru(x, h0) # y:(NumFrames, 1, Hidden*2), hn:(2*Layers, 1, Hidden)
        y = y.squeeze(1) # (NumFrames, Hidden*2)
        out = self.fc(self.ln(y)) # (NumFrames, output_size)
        return out, hn # (NumFrames, output_size), (2*Layers, 1, Hidden)
    

class GRUEncoder(nn.Module):
    r"""
    GRU encoder for stage 2
    """

    def __init__(self, input_size = 768, hidden_size = 512, number_layers = 5):
        super().__init__()

        # bidirectional GRU
        self.num_layers = number_layers
        self.hidden_size = hidden_size
        
        self.gru = nn.GRU(input_size, hidden_size, num_layers=number_layers, batch_first=False, bidirectional=True, dropout=0.4)
        self.ln_hid_enc = nn.LayerNorm(hidden_size*2)
        self.ln_enc = nn.LayerNorm(hidden_size*2)
        self.fc_enc = nn.Linear(2*hidden_size, hidden_size)
        self.skip_connect = nn.Linear(input_size, hidden_size*2) # Extra skip connect

    def forward(self, x, h=None):
        # x: (NumFrames, C) -> add batch dim at dim=1
        x_unsqueeze = x.unsqueeze(1) # (NumFrames, 1, C)

        y, hn = self.gru(x_unsqueeze, h) # y:(NumFrames, 1, Hidden*2), hn:(2*Layers, 1, Hidden)

        hn = hn.view(self.num_layers, 2, 1, self.hidden_size) # (Layers, 2, 1, Hidden)
        h_forward = hn[:, 0] # (Layers, 1, hidden)
        h_backward = hn[:, 1] # (Layers, 1, hidden)
        hidden = torch.cat((h_forward, h_backward), dim=-1) # (Layers, 1, 2*hidden)
        hidden = self.fc_enc(self.ln_hid_enc(hidden)) # (Layers, 1, hidden)

        y = y.squeeze(1) # (NumFrames, Hidden*2)
        out = self.skip_connect(x) + self.ln_enc(y) # (NumFrames, hidden_size*2), Extra skip connect

        return out, hidden # (NumFrames, hidden_size*2), (Layers, 1 hidden)
        

class GRUDecoder(nn.Module):
    r"""
    GRU decoder for stage 2
    """

    def __init__(self, hidden_size = 512, vocab_size = 32, number_layers = 5):
        super().__init__()

        self.hidden_size = hidden_size

        self.embd = nn.Embedding(vocab_size, hidden_size)

        self.att_proj_q = nn.Linear(hidden_size, hidden_size)
        self.att_proj_k = nn.Linear(hidden_size*2, hidden_size)
        self.att_proj_v = nn.Linear(hidden_size*2, hidden_size)

        self.gru_dec = nn.GRU(2*hidden_size, hidden_size, num_layers=number_layers, batch_first=False, dropout=0.4)

        self.ln_hid_dec = nn.LayerNorm(hidden_size)
        self.ln_out_dec = nn.LayerNorm(3*hidden_size)
        self.fc_out1 = nn.Linear(3*hidden_size, hidden_size//2) # Extra skip connect
        self.gelu = nn.GELU(approximate='tanh') # Extra skip connect
        self.fc_out2 = nn.Linear(hidden_size//2, vocab_size) # Extra skip connect

    def forward(self, enc_output, shot, hidden):
        # enc_output: (NumFrames, hidden_size*2)
        # shot: (,)
        # hidden: (Layers, 1 hidden_size)

        shot = shot.view(1, 1) # (1, 1)
        tok_emb = self.embd(shot) # (1, 1, hidden_size)
        s = hidden[-1] # (1, hidden_size)

        query = self.att_proj_q(s) # (1, hidden_size)
        key = self.att_proj_k(enc_output) # (NumFrames, hidden_size)
        att = key @ query.transpose(1, 0) # (NumFrames, 1)
        alphas = F.softmax(att, dim=0) # (NumFrames, 1)

        value = self.att_proj_v(enc_output) # (NumFrames, hidden_size)
        c = value.transpose(1,0) @ alphas # (hidden_size, 1)
        c = c.view(1, 1, -1) # (1, 1, hidden_size)
        c = c + s # Extra skip connect

        gru_input = torch.cat((tok_emb, c), dim = 2) #  # (1, 1, 2*hidden_size)
        output, h = self.gru_dec(gru_input, hidden) # (1, 1, hidden_size), (Layers, 1, hidden_size)

        h = self.ln_hid_dec(h)

        pre_logits = torch.cat((output, gru_input), dim=2).squeeze(1) # (1, 3*hidden_size)
        pre_logits = self.fc_out1(self.ln_out_dec(pre_logits)) # Extra skip connect
        pre_logits = self.gelu(pre_logits) # Extra skip connect
        logits = self.fc_out2(pre_logits) # (1, vocab_size) # Extra skip connect

        return logits.squeeze(0), h # (vocab_size,), (Layers, 1, hidden_size)
        

class GRUStage2(nn.Module):
    r"""
    GRU for stage 2
    """

    def __init__(self, input_size = 768, hidden_size = 512, vocab_size = 32, number_layers = 2):
        super().__init__()

        self.encoder = GRUEncoder(input_size=input_size, hidden_size=hidden_size, number_layers=number_layers)
        self.decoder = GRUDecoder(hidden_size= hidden_size, vocab_size=vocab_size, number_layers=number_layers)

    def forward(self, x, y=None, h=None):
        # x: (NumFrames, C)
        # y: (Num of shots in this point,)

        # append point separator token
        point_token = torch.tensor([POINTSEPARATOR], dtype=torch.long, device=y.device)
        
        if y is not None:
            y = torch.cat((point_token, y), dim=0) # (Num of shots in this point+1,)

        enc_output, h = self.encoder(x) # (NumFrames, hidden_size*2), (Layers, 1 hidden)

        outputs = [] # a list of (vocab_size,) tensors
        if y is not None:
            for i in range(len(y)-1):
                # teacher forcing
                logit, h = self.decoder(enc_output, y[i], h) # (vocab_size,), (Layers, 1, hidden_size)
                outputs.append(logit)
        else:
            current_token = point_token
            while current_token.item() != POINTSEPARATOR or len(outputs) == 0: # .item() brings it back to cpu
                # no teacher forcing, stop when POINTSEPARATOR is generated
                logit, h = self.decoder(enc_output, current_token, h) # (vocab_size,), (Layers, 1, hidden_size)
                current_token = logit.argmax() # should be a GPU tensor since logit is GPU tensor
                outputs.append(logit)

        logits = torch.stack(outputs, dim=0) # (num of shots, vocab_size)
        return logits