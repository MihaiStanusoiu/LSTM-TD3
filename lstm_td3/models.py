from copy import deepcopy

import ncps
import torch
from ncps.torch import CfC, LTC
from torch import nn as nn

from lstm_td3.utils.tools import SequenceCellType


class MLPCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, rnn_cell_type: SequenceCellType,
                 mem_pre_lstm_hid_sizes=(128,),
                 mem_lstm_hid_sizes=(128,),
                 mem_after_lstm_hid_size=(128,),
                 cur_feature_hid_sizes=(128,),
                 post_comb_hid_sizes=(128,),
                 hist_with_past_act=False):
        super(MLPCritic, self).__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.rnn_cell_type = rnn_cell_type
        self.hist_with_past_act = hist_with_past_act
        #
        self.mem_pre_lstm_layers = nn.ModuleList()
        self.mem_lstm_layers = nn.ModuleList()
        self.mem_after_lstm_layers = nn.ModuleList()

        self.cur_feature_layers = nn.ModuleList()
        self.post_combined_layers = nn.ModuleList()
        # Memory

        hidden_layer_factory = None
        if rnn_cell_type == SequenceCellType.LSTM.value:
            hidden_layer_factory = lambda input_size, output_size: nn.LSTM(input_size, output_size, batch_first=True)
        else:
            hidden_layer_factory = lambda input_size, units: CfC(input_size, units, batch_first=True, backbone_layers=list(mem_pre_lstm_hid_sizes).__len__(), backbone_units=mem_pre_lstm_hid_sizes[0])
        if self.hist_with_past_act:
            mem_pre_lstm_layer_size = [obs_dim + act_dim] + list(mem_pre_lstm_hid_sizes)
        else:
            mem_pre_lstm_layer_size = [obs_dim] + list(mem_pre_lstm_hid_sizes)
        for h in range(len(mem_pre_lstm_layer_size) - 1):
            self.mem_pre_lstm_layers += [nn.Linear(mem_pre_lstm_layer_size[h],
                                                   mem_pre_lstm_layer_size[h + 1]),
                                         nn.ReLU()]
        self.mem_lstm_layer_sizes = [mem_pre_lstm_layer_size[-1]] + list(mem_lstm_hid_sizes)
        for h in range(len(self.mem_lstm_layer_sizes) - 1):
            self.mem_lstm_layers += [hidden_layer_factory(self.mem_lstm_layer_sizes[h], self.mem_lstm_layer_sizes[h + 1])]
        #   After-LSTM
        self.mem_after_lstm_layer_size = [self.mem_lstm_layer_sizes[-1]] + list(mem_after_lstm_hid_size)
        for h in range(len(self.mem_after_lstm_layer_size) - 1):
            self.mem_after_lstm_layers += [nn.Linear(self.mem_after_lstm_layer_size[h],
                                                     self.mem_after_lstm_layer_size[h + 1]),
                                           nn.ReLU()]

        # Current Feature Extraction
        cur_feature_layer_size = [obs_dim + act_dim] + list(cur_feature_hid_sizes)
        for h in range(len(cur_feature_layer_size) - 1):
            self.cur_feature_layers += [nn.Linear(cur_feature_layer_size[h], cur_feature_layer_size[h + 1]),
                                        nn.ReLU()]

        # Post-Combination
        post_combined_layer_size = [self.mem_after_lstm_layer_size[-1] + cur_feature_layer_size[-1]] + list(
            post_comb_hid_sizes) + [1]
        for h in range(len(post_combined_layer_size) - 2):
            self.post_combined_layers += [nn.Linear(post_combined_layer_size[h], post_combined_layer_size[h + 1]),
                                          nn.ReLU()]
        self.post_combined_layers += [nn.Linear(post_combined_layer_size[-2], post_combined_layer_size[-1]),
                                      nn.Identity()]

    # def build_rnn_layer(self, units, output_size, batch_first):
    #     if self.rnn_cell_type == SequenceCellType.LSTM:
    #         return nn.LSTM(units, output_size, batch_first=batch_first)
    #     elif self.rnn_cell_type == SequenceCellType.LTC:
    #         wiring = ncps.wirings.Random(units, output_size, 0.75)
    #         return LTC(units, wiring, batch_first=batch_first)
    #     elif self.rnn_cell_type == SequenceCellType.CFC:
    #         return CfC(units, output_size, batch_first=batch_first)
    #     else:
    #         raise ValueError("Invalid rnn_cell_type")

    def forward(self, obs, act, hist_obs, hist_act, hist_seg_len):
        #
        tmp_hist_seg_len = deepcopy(hist_seg_len)
        tmp_hist_seg_len[hist_seg_len == 0] = 1
        if self.hist_with_past_act:
            x = torch.cat([hist_obs, hist_act], dim=-1)
        else:
            x = hist_obs
        # Memory
        #    Pre-LSTM
        for layer in self.mem_pre_lstm_layers:
            x = layer(x)
        #    LSTM
        for layer in self.mem_lstm_layers:
            x, h = layer(x)
        #    After-LSTM
        for layer in self.mem_after_lstm_layers:
            x = layer(x)
        if self.rnn_cell_type == SequenceCellType.LSTM.value:
            hist_out = torch.gather(x, 1,
                                    (tmp_hist_seg_len - 1).view(-1, 1).repeat(1, self.mem_after_lstm_layer_size[
                                        -1]).unsqueeze(
                                        1).long()).squeeze(1)
        else:
            hist_out = h

        # Current Feature Extraction
        x = torch.cat([obs, act], dim=-1)
        for layer in self.cur_feature_layers:
            x = layer(x)

        # Post-Combination
        extracted_memory = hist_out
        x = torch.cat([extracted_memory, x], dim=-1)

        for layer in self.post_combined_layers:
            x = layer(x)
        # squeeze(x, -1) : critical to ensure q has right shape.
        return torch.squeeze(x, -1), extracted_memory


class NCPActor(nn.Module):
    def __init__(self, units, obs_dim, act_dim, act_limit, mem_pre_lstm_hid_sizes=(128,)):
        super(NCPActor, self).__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.act_limit = act_limit
        self.mem_pre_lstm_layers = nn.ModuleList()
        mem_pre_lstm_layer_size = [obs_dim] + list(mem_pre_lstm_hid_sizes)
        for h in range(len(mem_pre_lstm_layer_size) - 1):
            self.mem_pre_lstm_layers += [nn.Linear(mem_pre_lstm_layer_size[h],
                                                   mem_pre_lstm_layer_size[h + 1]),
                                         nn.ReLU()]
        input_size = obs_dim
        wiring = ncps.wirings.AutoNCP(units, act_dim)
        self.rnn = LTC(input_size, wiring, batch_first=True, return_sequences=False)

    def forward(self, obs, hist_obs, hist_act, hist_seg_len):
        h = None
        with torch.no_grad():
            x = hist_obs
            for layer in self.mem_pre_lstm_layers:
                x = layer(x)
            _, h = self.rnn(x, h)
        x = obs
        for layer in self.mem_pre_lstm_layers:
            x = layer(x)
        a, h = self.rnn(x, h)
        return self.act_limit * a, h

class MLPActor(nn.Module):
    def __init__(self, obs_dim, act_dim, act_limit, rnn_cell_type: SequenceCellType,
                 mem_pre_lstm_hid_sizes=(128,),
                 mem_lstm_hid_sizes=(128,),
                 mem_after_lstm_hid_size=(128,),
                 cur_feature_hid_sizes=(128,),
                 post_comb_hid_sizes=(128,),
                 hist_with_past_act=False):
        super(MLPActor, self).__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.act_limit = act_limit
        self.rnn_cell_type = rnn_cell_type
        self.hist_with_past_act = hist_with_past_act
        #
        self.mem_pre_lstm_layers = nn.ModuleList()
        self.mem_lstm_layers = nn.ModuleList()
        self.mem_after_lstm_layers = nn.ModuleList()

        self.cur_feature_layers = nn.ModuleList()
        self.post_combined_layers = nn.ModuleList()

        # Memory
        if rnn_cell_type in [SequenceCellType.LSTM.value, SequenceCellType.CFC.value]:
            if rnn_cell_type == SequenceCellType.LSTM.value:
                hidden_layer_factory = lambda input_size, output_size: nn.LSTM(input_size, output_size,
                                                                               batch_first=True)
            else:
                hidden_layer_factory = lambda input_size, units: CfC(input_size, units, batch_first=True,
                                                                     backbone_layers=list(
                                                                         mem_pre_lstm_hid_sizes).__len__(),
                                                                     backbone_units=mem_pre_lstm_hid_sizes[0])
            #    Pre-LSTM
            if self.hist_with_past_act:
                mem_pre_lstm_layer_size = [obs_dim + act_dim] + list(mem_pre_lstm_hid_sizes)
            else:
                mem_pre_lstm_layer_size = [obs_dim] + list(mem_pre_lstm_hid_sizes)
            for h in range(len(mem_pre_lstm_layer_size) - 1):
                self.mem_pre_lstm_layers += [nn.Linear(mem_pre_lstm_layer_size[h],
                                                       mem_pre_lstm_layer_size[h + 1]),
                                             nn.ReLU()]
            #    LSTM
            self.mem_lstm_layer_sizes = [mem_pre_lstm_layer_size[-1]] + list(mem_lstm_hid_sizes)
            for h in range(len(self.mem_lstm_layer_sizes) - 1):
                self.mem_lstm_layers += [hidden_layer_factory(self.mem_lstm_layer_sizes[h], self.mem_lstm_layer_sizes[h + 1])]
            #   After-LSTM
            self.mem_after_lstm_layer_size = [self.mem_lstm_layer_sizes[-1]] + list(mem_after_lstm_hid_size)
            for h in range(len(self.mem_after_lstm_layer_size) - 1):
                self.mem_after_lstm_layers += [nn.Linear(self.mem_after_lstm_layer_size[h],
                                                         self.mem_after_lstm_layer_size[h + 1]),
                                               nn.ReLU()]
        elif rnn_cell_type == SequenceCellType.LTC.value:
            #    Pre-LTC NPC
            mem_pre_lstm_layer_size = [obs_dim] + list(mem_pre_lstm_hid_sizes)
            for h in range(len(mem_pre_lstm_layer_size) - 1):
                self.mem_pre_lstm_layers += [nn.Linear(mem_pre_lstm_layer_size[h],
                                                       mem_pre_lstm_layer_size[h + 1]),
                                             nn.ReLU()]
            # input_size = mem_pre_lstm_layer_size[-1]
            input_size = obs_dim
            wiring = ncps.wirings.AutoNCP(mem_lstm_hid_sizes[0], act_dim)
            # self.rnn = CfC(input_size, wiring, batch_first=True, return_sequences=True)
            self.rnn = LTC(input_size, wiring, batch_first=True, return_sequences=False)
            # self.rnn = CfC(input_size, mem_lstm_hid_sizes[0], batch_first=True,
            #                backbone_layers=list(mem_pre_lstm_hid_sizes).__len__(),
            #                backbone_units=mem_pre_lstm_hid_sizes[0])
            self.mem_after_lstm_layer_size = [mem_lstm_hid_sizes[0]]
        else:
            input_size = obs_dim
            if hist_with_past_act:
                input_size += act_dim
            self.rnn = CfC(input_size, mem_lstm_hid_sizes[0], batch_first=True,
                           backbone_layers=list(mem_pre_lstm_hid_sizes).__len__(),
                           backbone_units=mem_pre_lstm_hid_sizes[0])
            self.mem_after_lstm_layer_size = [mem_lstm_hid_sizes[0]]

        # Current Feature Extraction
        cur_feature_layer_size = [obs_dim] + list(cur_feature_hid_sizes)
        for h in range(len(cur_feature_layer_size) - 1):
            self.cur_feature_layers += [nn.Linear(cur_feature_layer_size[h], cur_feature_layer_size[h + 1]),
                                        nn.ReLU()]

        # Post-Combination
        post_combined_layer_size = [self.mem_after_lstm_layer_size[-1] + cur_feature_layer_size[-1]] + list(
            post_comb_hid_sizes) + [act_dim]
        for h in range(len(post_combined_layer_size) - 2):
            self.post_combined_layers += [nn.Linear(post_combined_layer_size[h], post_combined_layer_size[h + 1]),
                                          nn.ReLU()]
        self.post_combined_layers += [nn.Linear(post_combined_layer_size[-2], post_combined_layer_size[-1]), nn.Tanh()]

    @property
    def forward(self):
        _forward_val = getattr(self, "_forward_val", None)
        if _forward_val is not None:
            return _forward_val
        if self.compile:
            forward = torch.compile(self._forward, mode="reduce-overhead")
        else:
            forward = self._forward
        self._forward_val = forward
        return self._forward_val

    def _forward(self, obs, hist_obs, hist_act, hist_seg_len):
        #
        tmp_hist_seg_len = deepcopy(hist_seg_len)
        tmp_hist_seg_len[hist_seg_len == 0] = 1
        if self.hist_with_past_act:
            x = torch.cat([hist_obs, hist_act], dim=-1)
        else:
            x = hist_obs
        # Memory
        if self.rnn_cell_type == SequenceCellType.LTC.value:
            # for layer in self.mem_pre_lstm_layers:
            #     x = layer(x)
            x = torch.cat((hist_obs, obs.unsqueeze(dim=1)), dim=1)
            x, h = self.rnn(x)
            return self.act_limit * x, h
        else:
            #    Pre-LSTM
            for layer in self.mem_pre_lstm_layers:
                x = layer(x)
            #    LSTM
            for layer in self.mem_lstm_layers:
                x, h = layer(x)
            #    After-LSTM
            for layer in self.mem_after_lstm_layers:
                x = layer(x)

            if self.rnn_cell_type == SequenceCellType.LSTM.value:
                hist_out = torch.gather(x, 1,
                                        (tmp_hist_seg_len - 1).view(-1, 1).repeat(1, self.mem_after_lstm_layer_size[-1]).unsqueeze(
                                            1).long()).squeeze(1)
            else:
                hist_out = h

        # Current Feature Extraction
        x = obs
        for layer in self.cur_feature_layers:
            x = layer(x)

        # Post-Combination
        extracted_memory = hist_out
        x = torch.cat([extracted_memory, x], dim=-1)

        for layer in self.post_combined_layers:
            x = layer(x)
        return self.act_limit * x, extracted_memory


class MLPActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, rnn_cell_type: SequenceCellType, act_limit=1,
                 critic_mem_pre_lstm_hid_sizes=(128,),
                 critic_mem_lstm_hid_sizes=(128,),
                 critic_mem_after_lstm_hid_size=(128,),
                 critic_cur_feature_hid_sizes=(128,),
                 critic_post_comb_hid_sizes=(128,),
                 critic_hist_with_past_act=False,
                 actor_mem_pre_lstm_hid_sizes=(128,),
                 actor_mem_lstm_hid_sizes=(128,),
                 actor_mem_after_lstm_hid_size=(128,),
                 actor_cur_feature_hid_sizes=(128,),
                 actor_post_comb_hid_sizes=(128,),
                 actor_hist_with_past_act=False):
        super(MLPActorCritic, self).__init__()
        self.q1 = MLPCritic(obs_dim, act_dim, rnn_cell_type,
                            mem_pre_lstm_hid_sizes=critic_mem_pre_lstm_hid_sizes,
                            mem_lstm_hid_sizes=critic_mem_lstm_hid_sizes,
                            mem_after_lstm_hid_size=critic_mem_after_lstm_hid_size,
                            cur_feature_hid_sizes=critic_cur_feature_hid_sizes,
                            post_comb_hid_sizes=critic_post_comb_hid_sizes,
                            hist_with_past_act=critic_hist_with_past_act)
        self.q2 = MLPCritic(obs_dim, act_dim, rnn_cell_type,
                            mem_pre_lstm_hid_sizes=critic_mem_pre_lstm_hid_sizes,
                            mem_lstm_hid_sizes=critic_mem_lstm_hid_sizes,
                            mem_after_lstm_hid_size=critic_mem_after_lstm_hid_size,
                            cur_feature_hid_sizes=critic_cur_feature_hid_sizes,
                            post_comb_hid_sizes=critic_post_comb_hid_sizes,
                            hist_with_past_act=critic_hist_with_past_act)
        self.pi = MLPActor(obs_dim, act_dim, act_limit, rnn_cell_type,
                           mem_pre_lstm_hid_sizes=actor_mem_pre_lstm_hid_sizes,
                           mem_lstm_hid_sizes=actor_mem_lstm_hid_sizes,
                           mem_after_lstm_hid_size=actor_mem_after_lstm_hid_size,
                           cur_feature_hid_sizes=actor_cur_feature_hid_sizes,
                           post_comb_hid_sizes=actor_post_comb_hid_sizes,
                           hist_with_past_act=actor_hist_with_past_act)
        if rnn_cell_type == SequenceCellType.LTC.value:
            self.pi = NCPActor(actor_mem_lstm_hid_sizes[0], obs_dim, act_dim, act_limit)

    def act(self, obs, hist_obs=None, hist_act=None, hist_seg_len=None, device=None):
        if (hist_obs is None) or (hist_act is None) or (hist_seg_len is None):
            hist_obs = torch.zeros(1, 1, self.obs_dim).to(device)
            hist_act = torch.zeros(1, 1, self.act_dim).to(device)
            hist_seg_len = torch.zeros(1).to(device)
        with torch.no_grad():
            torch.compiler.cudagraph_mark_step_begin()
            act, h, = self.pi(obs, hist_obs, hist_act, hist_seg_len)
            return act.cpu().numpy()
