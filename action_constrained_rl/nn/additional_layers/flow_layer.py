import torch
import torch.nn as nn
import torch.nn.functional as F

from flow.base_constraint import UpsideDownConstraint
from torch.distributions import MultivariateNormal
from action_constrained_rl.constraint.constraint import Constraint
from stable_baselines3.common.logger import Logger
from copy import deepcopy
from typing import Callable
from dataclasses import dataclass
from core.flow.real_nvp import RealNvp
from core.flow.train_flow import update_flow_batch
import sys
from stable_baselines3.common.utils import polyak_update
from experiments.train_flow_backward import all_problems, BaseProblem, get_constraint_with_action_bounds, ConstrainedDistribution

@dataclass
class CombinedStateGenerator(Callable):
    gen1: Callable
    gen2: Callable
    dim_sizes: tuple

    def __call__(self, batch_size, dim_size):
        assert dim_size == sum(self.dim_sizes)
        return torch.cat([self.gen1(batch_size, self.dim_sizes[0]), self.gen2(batch_size, self.dim_sizes[1])], dim=1)



class ConstraintUD(UpsideDownConstraint):
    def __init__(self, constraint: Constraint, state_indexes: list, sigma:float, state_generator) -> None:
        super().__init__(sigma)
        self.rl_const = constraint
        self.var_count = constraint.a_dim
        self.conditional_param_count = len(state_indexes)
        self.state_generator = state_generator
        self.state_indexes = state_indexes

    def get_log_prob(self, z, y=None):
        cv = self.rl_const.getCVBatchForSelectedStates(y, z)
        return self.noise_prior.log_prob(cv/self.sigma)

    def generate_data(self, count, seed):
        total_count = 0
        batch_size = 1000
        state_generator = self.get_state_data_generator()
        a_all = []
        s_all = []
        has_conditional_vars =self.conditional_param_count > 0
        for i in range(1000):
            actions = torch.rand((batch_size, self.var_count))*2 -1 # map to -1, 1 range
            if has_conditional_vars:
                states = state_generator(batch_size, self.conditional_param_count)
                validity = self.constraint(actions.numpy(), states.numpy())
                states = states[validity]
                s_all.append(states)
            else:
                validity = self.constraint(actions.numpy(), None)
            actions = actions[validity]
            a_all.append(actions)
            total_count+= len(actions)
            if total_count >= count:
                print(f"{total_count}/{count}", len(actions))
                break
        a_all = torch.cat(a_all, dim=0)[:count]
        if has_conditional_vars:
            s_all = torch.cat(s_all, dim=0)[:count]
        return a_all, (s_all if has_conditional_vars else None)

    def constraint(self, x, y):
        return self.rl_const.isConstraintBatchSatisfiedForSelectedStates(y, x)
    
    def get_state_data_generator(self):
        return self.state_generator



class FlowLayer(torch.nn.Module):
    flow_file_path = None
    flow_direction_f = True
    problem_str: str = None
    problem: BaseProblem
    flow_kl_beta: float
    update_ref_flow:bool
    logger: Logger

    @staticmethod
    def setup_flow_layer():
        if not FlowLayer.flow_file_path:
            raise Exception("Flow file path is not set.")
        FlowLayer.flow = RealNvp.load_module(FlowLayer.flow_file_path).to(FlowLayer.device_id)
        FlowLayer.flow_train = deepcopy(FlowLayer.flow)
        FlowLayer.flow_ref = deepcopy(FlowLayer.flow)
        FlowLayer.flow_train.disable_grad(False)
        FlowLayer.flow.disable_grad()
        FlowLayer.flow_ref.disable_grad()
        problem:BaseProblem  = all_problems[FlowLayer.problem_str]
        FlowLayer.problem = problem
        constraint = problem.constraint
        FlowLayer.action_dim = constraint.var_count
        print("S-scaling factor:", FlowLayer.flow.s_scaling_factor)
        sys.stdout.flush()


    def __init__(self, constraint):
        super(FlowLayer, self).__init__()

    

    @staticmethod
    def flow_forward(actions, states):
        if len(FlowLayer.state_indexes) > 0:
            states = torch.stack([states[:, i] for i in FlowLayer.state_indexes], dim=1)
            v = torch.concat([actions, states], dim=1)
        else:
            states = None
            v = actions

        if FlowLayer.flow_direction_f:
            return FlowLayer.flow.f(v.double())[0][:,:actions.shape[1]].float()
        else:
            return FlowLayer.flow.g(v.double())[0][:,:actions.shape[1]].float() # Clip to -0.95, 0.95

        
    @staticmethod
    def to_tensor(value):
        return torch.tensor(value, dtype=torch.double, device=FlowLayer.device_id)

    def forward(self, actions, states, centers=None):
        return FlowLayer.flow_forward(actions, states)

