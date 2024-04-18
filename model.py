# # coding: utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class KT_backbone(nn.Module):
    def __init__(self, skill_dim, answer_dim, hidden_dim, output_dim):
        super(KT_backbone, self).__init__()
        self.skill_dim=skill_dim
        self.answer_dim=answer_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.rnn = nn.LSTM(self.skill_dim+self.answer_dim, self.hidden_dim, batch_first=True)
        self.fc = nn.Linear(self.hidden_dim*2, self.output_dim) 
        #self.fc = nn.Linear(self.hidden_dim, self.output_dim)
        self.sig = nn.Sigmoid()
        
        self.skill_emb = nn.Embedding(self.output_dim+1, self.skill_dim)
        self.skill_emb.weight.data[-1]= 0
        
        self.answer_emb = nn.Embedding(2+1, self.answer_dim)
        self.answer_emb.weight.data[-1]= 0
        
        self.attention_dim = 80
        self.mlp = nn.Linear(self.hidden_dim, self.attention_dim)
        self.similarity = nn.Linear(self.attention_dim, 1, bias=False)
        
        self.filter = FilterLayer()
    def _get_next_pred(self, res, skill):
        
        one_hot = torch.eye(self.output_dim, device=res.device)
        one_hot = torch.cat((one_hot, torch.zeros(1, self.output_dim).to(device)), dim=0)
        next_skill = skill[:, 1:]
        one_hot_skill = F.embedding(next_skill, one_hot)
        
        pred = (res * one_hot_skill).sum(dim=-1)
        return pred
    
    def attention_module(self, lstm_output):
        
        att_w = self.mlp(lstm_output)
        att_w = torch.tanh(att_w)
        att_w = self.similarity(att_w)
        
        alphas=nn.Softmax(dim=1)(att_w)
        
        attn_ouput=alphas*lstm_output
        attn_output_cum=torch.cumsum(attn_ouput, dim=1)
        attn_output_cum_1=attn_output_cum-attn_ouput

        final_output=torch.cat((attn_output_cum_1, lstm_output),2)
        
        return final_output


    def forward(self, skill, answer, perturbation=None):
        
        skill_embedding=self.skill_emb(skill)
        answer_embedding=self.answer_emb(answer)

        skill_embedding_filter = self.filter(skill_embedding)
        skill_answer=torch.cat((skill_embedding_filter,answer_embedding), 2)
        answer_skill=torch.cat((answer_embedding,skill_embedding_filter), 2)
        
        # skill_answer=torch.cat((skill_embedding,answer_embedding), 2)
        # answer_skill=torch.cat((answer_embedding,skill_embedding), 2)
        
        answer=answer.unsqueeze(2).expand_as(skill_answer)
        
        skill_answer_embedding=torch.where(answer==1, skill_answer, answer_skill)
        
        skill_answer_embedding1=skill_answer_embedding
        
        if  perturbation is not None:
            skill_answer_embedding+=perturbation
        
        out,_ = self.rnn(skill_answer_embedding)
        out=self.attention_module(out)
        res = self.sig(self.fc(out))

        res = res[:, :-1, :]
        pred_res = self._get_next_pred(res, skill)
        
        return pred_res, skill_answer_embedding1,skill_embedding,skill_embedding_filter,res

class FilterLayer(nn.Module):
    def __init__(self):
        super(FilterLayer, self).__init__()
        # 做完self-attention 做一个前馈全连接 LayerNorm 输出
        self.complex_weight = nn.Parameter(torch.randn(1, 500//2 + 1,256, 2, dtype=torch.float32) * 0.02)
        self.out_dropout = nn.Dropout(0.5)
        self.LayerNorm = nn.LayerNorm(256)


    def forward(self, input_tensor):
        # [batch, seq_len, hidden]
        #sequence_emb_fft = torch.rfft(input_tensor, 2, onesided=False)  # [:, :, :, 0]
        #sequence_emb_fft = torch.fft(sequence_emb_fft.transpose(1, 2), 2)[:, :, :, 0].transpose(1, 2)
        batch, seq_len, hidden = input_tensor.shape
        x = torch.fft.rfft(input_tensor, dim=1, norm='ortho') #torch.Size([256, 26, 64])
        weight = torch.view_as_complex(self.complex_weight) #torch.Size([1, 26, 64])
        x = x * weight
        sequence_emb_fft = torch.fft.irfft(x, n=seq_len, dim=1, norm='ortho') #torch.Size([256, 50, 64])
        hidden_states = self.out_dropout(sequence_emb_fft) #torch.Size([256, 50, 64])
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states

# coding: utf-8
# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# class KT_backbone(nn.Module):
#     def __init__(self, skill_dim, answer_dim, hidden_dim, output_dim):
#         super(KT_backbone, self).__init__()
#         self.skill_dim=skill_dim
#         self.answer_dim=answer_dim
#         self.hidden_dim = hidden_dim
#         self.output_dim = output_dim
#         self.rnn = nn.LSTM(self.skill_dim+self.answer_dim, self.hidden_dim, batch_first=True)
#         self.fc = nn.Linear(self.hidden_dim*2, self.output_dim)
#         self.sig = nn.Sigmoid()
        
#         self.skill_emb = nn.Embedding(self.output_dim+1, self.skill_dim)
#         self.skill_emb.weight.data[-1]= 0
        
#         self.answer_emb = nn.Embedding(2+1, self.answer_dim)
#         self.answer_emb.weight.data[-1]= 0
        
#         self.attention_dim = 80
#         self.mlp = nn.Linear(self.hidden_dim, self.attention_dim)
#         self.similarity = nn.Linear(self.attention_dim, 1, bias=False)
        
#     def _get_next_pred(self, res, skill):
        
#         one_hot = torch.eye(self.output_dim, device=res.device)
#         one_hot = torch.cat((one_hot, torch.zeros(1, self.output_dim).to(device)), dim=0)
#         next_skill = skill[:, 1:]
#         one_hot_skill = F.embedding(next_skill, one_hot)
        
#         pred = (res * one_hot_skill).sum(dim=-1)
#         return pred
    
#     def attention_module(self, lstm_output):
        
#         att_w = self.mlp(lstm_output)
#         att_w = torch.tanh(att_w)
#         att_w = self.similarity(att_w)
        
#         alphas=nn.Softmax(dim=1)(att_w)
        
#         attn_ouput=alphas*lstm_output
#         attn_output_cum=torch.cumsum(attn_ouput, dim=1)
#         attn_output_cum_1=attn_output_cum-attn_ouput

#         final_output=torch.cat((attn_output_cum_1, lstm_output),2)
        
#         return final_output


#     def forward(self, skill, answer, perturbation=None):
        
#         skill_embedding=self.skill_emb(skill)
#         answer_embedding=self.answer_emb(answer)
        
#         skill_answer=torch.cat((skill_embedding,answer_embedding), 2)
#         answer_skill=torch.cat((answer_embedding,skill_embedding), 2)
        
#         answer=answer.unsqueeze(2).expand_as(skill_answer)
        
#         skill_answer_embedding=torch.where(answer==1, skill_answer, answer_skill)

#         skill_answer_embedding1=skill_answer_embedding
        
#         if  perturbation is not None:
#             skill_answer_embedding+=perturbation
        
#         out,_ = self.rnn(skill_answer_embedding)
#         out=self.attention_module(out)
#         res = self.sig(self.fc(out))

#         res = res[:, :-1, :]
#         pred_res = self._get_next_pred(res, skill)
        
#         return pred_res, skill_answer_embedding1

# coding: utf-8
# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# class KT_backbone(nn.Module):
#     def __init__(self, skill_dim, answer_dim, hidden_dim, output_dim):
#         super(KT_backbone, self).__init__()
#         self.skill_dim=skill_dim
#         self.answer_dim=answer_dim
#         self.hidden_dim = hidden_dim
#         self.output_dim = output_dim
#         self.rnn = nn.LSTM(self.skill_dim+self.answer_dim, self.hidden_dim, batch_first=True)
#         self.fc = nn.Linear(self.hidden_dim*2, self.output_dim)
#         self.sig = nn.Sigmoid()
        
#         self.skill_emb = nn.Embedding(self.output_dim+1, self.skill_dim)
#         self.skill_emb.weight.data[-1]= 0
        
#         self.answer_emb = nn.Embedding(2+1, self.answer_dim)
#         self.answer_emb.weight.data[-1]= 0
        
#         self.attention_dim = 80
#         self.mlp = nn.Linear(self.hidden_dim, self.attention_dim)
#         self.similarity = nn.Linear(self.attention_dim, 1, bias=False)
        
#         self.filter = FilterLayer()
#     def _get_next_pred(self, res, skill):
        
#         one_hot = torch.eye(self.output_dim, device=res.device)
#         one_hot = torch.cat((one_hot, torch.zeros(1, self.output_dim).to(device)), dim=0)
#         next_skill = skill[:, 1:]
#         one_hot_skill = F.embedding(next_skill, one_hot)
        
#         pred = (res * one_hot_skill).sum(dim=-1)
#         return pred
    
#     def attention_module(self, lstm_output):
        
#         att_w = self.mlp(lstm_output)
#         att_w = torch.tanh(att_w)
#         att_w = self.similarity(att_w)
        
#         alphas=nn.Softmax(dim=1)(att_w)
        
#         attn_ouput=alphas*lstm_output
#         attn_output_cum=torch.cumsum(attn_ouput, dim=1)
#         attn_output_cum_1=attn_output_cum-attn_ouput

#         final_output=torch.cat((attn_output_cum_1, lstm_output),2)
        
#         return final_output


#     def forward(self, skill, answer, perturbation=None):
        
#         skill_embedding=self.skill_emb(skill)
#         answer_embedding=self.answer_emb(answer)

#         # skill_embedding_filter = self.filter(skill_embedding)
#         skill_answer=torch.cat((skill_embedding,answer_embedding), 2)
#         answer_skill=torch.cat((answer_embedding,skill_embedding), 2)
        
#         answer=answer.unsqueeze(2).expand_as(skill_answer)
        
#         skill_answer_embedding=torch.where(answer==1, skill_answer, answer_skill)

#         skill_answer_embedding_filter = self.filter(skill_answer_embedding)
        
#         skill_answer_embedding1=skill_answer_embedding_filter
        
#         if  perturbation is not None:
#             skill_answer_embedding_filter+=perturbation
        
#         out,_ = self.rnn(skill_answer_embedding_filter)
#         out=self.attention_module(out)
#         res = self.sig(self.fc(out))

#         res = res[:, :-1, :]
#         pred_res = self._get_next_pred(res, skill)
        
#         return pred_res, skill_answer_embedding1

# class FilterLayer(nn.Module):
#     def __init__(self):
#         super(FilterLayer, self).__init__()
#         # 做完self-attention 做一个前馈全连接 LayerNorm 输出
#         self.complex_weight = nn.Parameter(torch.randn(1, 500//2 + 1,352, 2, dtype=torch.float32) * 0.02)
#         self.out_dropout = nn.Dropout(0.5)
#         self.LayerNorm = nn.LayerNorm(352)


#     def forward(self, input_tensor):
#         # [batch, seq_len, hidden]
#         #sequence_emb_fft = torch.rfft(input_tensor, 2, onesided=False)  # [:, :, :, 0]
#         #sequence_emb_fft = torch.fft(sequence_emb_fft.transpose(1, 2), 2)[:, :, :, 0].transpose(1, 2)
#         batch, seq_len, hidden = input_tensor.shape
#         x = torch.fft.rfft(input_tensor, dim=1, norm='ortho') #torch.Size([256, 26, 64])
#         weight = torch.view_as_complex(self.complex_weight) #torch.Size([1, 26, 64])
#         x = x * weight
#         sequence_emb_fft = torch.fft.irfft(x, n=seq_len, dim=1, norm='ortho') #torch.Size([256, 50, 64])
#         hidden_states = self.out_dropout(sequence_emb_fft) #torch.Size([256, 50, 64])
#         hidden_states = self.LayerNorm(hidden_states + input_tensor)

#         return hidden_states