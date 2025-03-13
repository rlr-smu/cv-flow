from core.constraints import QuadraticConstraint, BoxConstraint, ConditionedQuadraticConstraint, CombinedConstraint, OrthoplexConstraint, ConditionedLinearConstraint, PowerConstraint, BaseConstraint, OrthoplexConstraintLB, LinearConstraint, BinaryConstraint, OneHotConstraint, IntegerConstraint, Unconstrained, BinerizedConstraint, FloorConstraint
import torch as th
from torch import TensorType, Tensor
from typing import Union, Literal
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from core.flow.data_loaders import UniformDataLoader, GaussianDataLoader, CombinedDataLoader


def get_box_constraint(dim, bounds=(-1, 1)):
    return BoxConstraint(dim, th.tensor([bounds[0]]*dim).double(), th.tensor([bounds[1]]*dim).double())



class BaseProblem:
    constraint: BaseConstraint
    state_action_bound_constraint: Union[BaseConstraint, None] = None # Used to sample using HMC 
    action_plot_range: list = [-1, 1]
    action_scale: tuple = (-1, 1)
    state_dist: Literal["Uniform", "Gaussian"] = "Uniform"
    state_dist_u_bound: float = 1 # For uniform: i.e. [-1, 1]
    state_dist_g_mu: float = 0.
    state_dist_g_sigma: float = 1. # For gaussian
    is_descrete: bool = False

    @property
    def action_bound_constraint(self):
        return get_box_constraint(self.constraint.var_count) # var_count=action_count
    
    def plot(self, scaled_samples: th.Tensor, feasibility=None):
        samples = self.unscale_action(scaled_samples[:, :self.constraint.var_count]) # only pick action part
        samples = samples.cpu().numpy()
        
        figure, ax = plt.figure(figsize=(5, 5)), plt.gca()
        H, xedges, yedges = np.histogram2d(samples[:, 0], samples[:, 1], bins=(80, 80))
        plot = ax.pcolormesh(xedges, yedges, H, cmap=plt.cm.jet, )
        ax.get_figure().colorbar(plot)
        return figure
    
                
    
    def get_state_data_loader(self, batch_size, batch_count, device, seed):
        conditional_parm_count = getattr(self.constraint, 'conditional_param_count', 0)
        if self.state_dist == "Uniform":
            return UniformDataLoader(batch_size, batch_count, conditional_parm_count, device, seed, (-self.state_dist_u_bound, self.state_dist_u_bound))
        if self.state_dist == "Gaussian":
            return GaussianDataLoader(batch_size, batch_count, conditional_parm_count, device, seed, self.state_dist_g_mu, self.state_dist_g_sigma)
        else:
            raise ValueError("Invalid state-dist")
    
    def unscale_action(self, action_batch: th.TensorType):
        return ((action_batch+1)/2)*(self.action_scale[1]-self.action_scale[0]) + self.action_scale[0]
    
    def scale_action(self, action_batch: th.TensorType):
        return (action_batch - self.action_scale[0]) * 2 /(self.action_scale[1]-self.action_scale[0]) - 1

    # Flow model is scaled: i.e. [-1, 1] -> [-1, 1] because it's easier for the model to learn
    # Constraint is unscaled.: i.e. [-1, 1] -> [0, 1] because it's easier to define constraints in actual range
    
    def is_feasible(self, scaled_samples):
        samples = self.unscale_action(scaled_samples)
        return self.constraint.is_feasible(samples)
    
"""
Define all the constraints used in all experiments here."""
#Box

class Box2(BaseProblem):
    constraint = get_box_constraint(2)


class R_L2(BaseProblem):
    constraint = QuadraticConstraint(2, th.eye(2).unsqueeze(dim=0).double(), th.tensor([0.05]).double())
    action_plot_range: list = [-1.2, 1.2]



class R_L2LB(BaseProblem):
    ub_const = QuadraticConstraint(2, th.eye(2).unsqueeze(dim=0).double(), th.tensor([0.05]).double())
    lb_const = QuadraticConstraint(2, -th.eye(2).unsqueeze(dim=0).double(), -th.tensor([0.045]).double())
    constraint = CombinedConstraint(2, 0, [ub_const, lb_const])
    action_plot_range: list = [-0.27, 0.27]

class H_L2(BaseProblem):
    constraint = QuadraticConstraint(3, th.eye(3).unsqueeze(dim=0).double(), th.tensor([0.05]).double())
    action_plot_range: list = [-0.27, 0.27]

class H_L2N(BaseProblem):
    constraint = QuadraticConstraint(3, -th.eye(3).unsqueeze(dim=0).double(), -th.tensor([1.373-0.756]).double()) # Mean-std lowerbound
    action_plot_range: list = [-1.3, 1.3]

class H_L2LB(BaseProblem):
    ub_const = QuadraticConstraint(3, th.eye(3).unsqueeze(dim=0).double(), th.tensor([0.05]).double())
    lb_const = QuadraticConstraint(3, -th.eye(3).unsqueeze(dim=0).double(), -th.tensor([0.045]).double())
    constraint = CombinedConstraint(3, 0, [ub_const, lb_const])
    action_plot_range: list = [-0.27, 0.27]

class H_L2_2(BaseProblem):
    constraint = QuadraticConstraint(3, th.eye(3).unsqueeze(dim=0).double(), th.tensor([1.5]).double())
    action_plot_range: list = [-0.27, 0.27]

class H_L2LB_2(BaseProblem):
    ub_const = QuadraticConstraint(3, th.eye(3).unsqueeze(dim=0).double(), th.tensor([1.5]).double())
    lb_const = QuadraticConstraint(3, -th.eye(3).unsqueeze(dim=0).double(), -th.tensor([1.4]).double())
    constraint = CombinedConstraint(3, 0, [ub_const, lb_const])
    action_plot_range: list = [-1.3, 1.3]

class W_L2(BaseProblem):
    constraint = QuadraticConstraint(6, th.eye(6).unsqueeze(dim=0).double(), th.tensor([0.05]).double())
    action_plot_range: list = [-0.27, 0.27]

class W_L2LB(BaseProblem):
    ub_const = QuadraticConstraint(6, th.eye(6).unsqueeze(dim=0).double(), th.tensor([0.05]).double())
    lb_const = QuadraticConstraint(6, -th.eye(6).unsqueeze(dim=0).double(), -th.tensor([0.045]).double())
    constraint = CombinedConstraint(6, 0, [ub_const, lb_const])
    action_plot_range: list = [-0.27, 0.27]

class R_T(BaseProblem):
    state_dist_u_bound: float = th.pi # For uniform: i.e. [-pi, pi]
    # R+T
    @staticmethod
    def get_q_m_r_t(c: th.Tensor):
        batch_size = len(c)
        Q=th.zeros((batch_size,2,2),device = c.device, dtype=c.dtype)
        cosg = th.cos(c[:, 0])
        Q[:,0,0] = 2 + 2 *cosg
        Q[:,0,1]= 2 * (1+cosg)
        Q[:,1,1]=1
        Q = Q.unsqueeze(dim=1)
        m = th.full((batch_size, 1), 0.05, dtype=c.dtype, device=c.device)
        return (Q, m)

    state_action_bound_constraint = BoxConstraint(3, th.tensor([-1, -1, -th.pi]).double(), th.tensor([1, 1, th.pi]).double())
    constraint = ConditionedQuadraticConstraint(2, 1, get_q_m=get_q_m_r_t.__get__(object))


class R_TLB(R_T):
    mean = 0.433
    std = 1.215

    @staticmethod
    def get_q_m_r_t_ub(c: th.Tensor):
        Q, m_ = R_T.get_q_m_r_t(c)
        m = th.full((len(c), 1), R_TLB.mean+R_TLB.std*.1, dtype=c.dtype, device=c.device)
        return (Q, m)

    @staticmethod
    def get_q_m_r_t_lb(c: th.Tensor):
        Q, m_ = R_T.get_q_m_r_t(c)
        m = th.full((len(c), 1), R_TLB.mean-R_TLB.std*.1, dtype=c.dtype, device=c.device)
        return (-Q, -m)

    ub_const = ConditionedQuadraticConstraint(2, 1, get_q_m=get_q_m_r_t_ub.__get__(object))
    lb_const = ConditionedQuadraticConstraint(2, 1, get_q_m=get_q_m_r_t_lb.__get__(object))
    constraint = CombinedConstraint(2, 1, [ub_const, lb_const])


class HC_O(BaseProblem):
    # HC+O
    high_b = th.tensor([1]*6 + [30]*6).double()
    state_action_bound_constraint = BoxConstraint(12, low=-high_b, high=high_b)
    constraint = OrthoplexConstraint(6, 6, 20)
    state_dist = "Gaussian"
    state_dist_g_sigma = 15

class HC_MA(BaseProblem):
    """ HC + MA
    Format a1, a4, w1, w4, theta1..6 = 10 variables
    w_1 a_1 sin(t1 + t2 + t3) + w_4 a_4 sin(t4 + t5+ + t6) <= 5
    """

    high_b = th.tensor([1, 1] + [30, 30] + [th.pi]*6).double()
    state_action_bound_constraint = BoxConstraint(10, low=-high_b, high=high_b)

    @staticmethod
    def get_a_b_hc_ma(c: th.Tensor):
        batch_size = len(c)
        w1, w2 = c[:, 2], c[:, 3]
        t1_3, t4_6 = c[:, 4:7].sum(dim=1), c[:, 7:10].sum(dim=1) # Sum of theta sets
        A = th.stack([w1 * th.sin(t1_3), w2*th.sin(t4_6)], dim=1).unsqueeze(dim=1)
        b = th.full((batch_size, 1), 5, dtype=c.dtype, device=c.device)
        return (A, b)

    constraint = ConditionedLinearConstraint(2, 8, get_a_b=get_a_b_hc_ma.__get__(object))


def get_power_constraint(dim:int, ub, state_bound):
    high_b = th.tensor([1]*dim + [state_bound]*dim).double()
    bounds = BoxConstraint(2*dim, low=-high_b, high=high_b)
    base = PowerConstraint(dim, dim, ub)
    return base, bounds

def get_orthoplex_constraint(dim:int, ub, state_bound):
    high_b = th.tensor([1]*dim + [state_bound]*dim).double()
    bounds = BoxConstraint(2*dim, low=-high_b, high=high_b)
    base = OrthoplexConstraint(dim, dim, ub)
    return base, bounds

class H_M(BaseProblem):
    state_dist_u_bound = 10
    constraint, state_action_bound_constraint = get_power_constraint(3, 10, state_dist_u_bound)

class W_M(BaseProblem):
    state_dist_u_bound = 10
    constraint, state_action_bound_constraint = get_power_constraint(6, 10, state_dist_u_bound)

    
def get_q_m_sin2(c: th.Tensor):
    batch_size, dim = c.shape
    sin_square = c.sin().square()
    Q = th.zeros((batch_size, dim, dim), dtype=c.dtype, device=c.device)
    Q[:, range(dim), range(dim)] = sin_square
    Q = Q.unsqueeze(dim=1) # batch of square matrics
    m = th.full((batch_size, 1), 0.1, device=c.device, dtype=c.dtype)
    return Q, m

def get_o_s_constraint(dim:int, ub1, state_bound1, state_bound2, lb1=None):
    """a1,..an,w1,..wn,t1,...tn"""
    o_const = OrthoplexConstraint(dim, dim, ub1)
    high_b = th.tensor([1]*dim + [state_bound1]*dim + [state_bound2]*dim).double()
    bounds = BoxConstraint(3*dim, low=-high_b, high=high_b)
    sin2_constraint = ConditionedQuadraticConstraint(dim, dim, get_q_m_sin2)
    o_const_indexes = th.LongTensor(range(dim*2))
    sin2_indexes = th.LongTensor(list(range(dim))+list(range(dim*2, dim*3)))
    constraints = [o_const, sin2_constraint]
    c_indexes = [o_const_indexes, sin2_indexes]
    if lb1 is not None:
        constraints.append(OrthoplexConstraintLB(dim, dim, lb1))
        c_indexes.append(th.LongTensor(range(dim*2)))
    base = CombinedConstraint(dim, dim*2, constraints, c_indexes)
    return base, bounds

    
class O_S_Problem(BaseProblem):
    def get_state_data_loader(self, batch_size, batch_count, device, seed):
        dim = self.constraint.var_count
        u1 = UniformDataLoader(batch_size, batch_count, dim, device, seed, (-10, 10))
        u2 = UniformDataLoader(batch_size, batch_count, dim, device, seed, (-th.pi, th.pi))
        return CombinedDataLoader(batch_size, batch_count, dim*2, device, seed, [u1, u2])

class H_O_S(O_S_Problem):
    constraint, state_action_bound_constraint = get_o_s_constraint(3, 10, 30, th.pi)

class H_O_S_LB(O_S_Problem):
    constraint, state_action_bound_constraint = get_o_s_constraint(3, 10, 30, th.pi, lb1=5)

class W_O_S(O_S_Problem):
    constraint, state_action_bound_constraint = get_o_s_constraint(6, 10, 30, th.pi)


all_problems = {
    "Box": Box2(),
    "R+L2": R_L2(),
    "R+L2LB": R_L2LB(),
    "R+T": R_T(),
    "R+TLB": R_TLB(),
    "HC+O": HC_O(),
    "HC+MA": HC_MA(),
    "H+L2": H_L2(),
    "H+L2N": H_L2N(),
    "H+L2LB": H_L2LB(),
    "H+L2_2": H_L2_2(),
    "H+L2LB_2": H_L2LB_2(),
    "W+L2": W_L2(),
    "W+L2LB": W_L2LB(),
    "H+M": H_M(),
    "W+M": W_M(),
    "H+O+S": H_O_S(),
    "H+O+S+LB": H_O_S_LB(),
    "W+O+S": W_O_S(),
}

