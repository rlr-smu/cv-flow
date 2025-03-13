
import torch as th
from core.flow.base_distribution import BaseDistribution
from core.constraints import BaseConstraint
from torch.distributions import Exponential, Normal, MultivariateNormal

class ConstrainedDistribution(BaseDistribution):
    """
    An un-normalized mollified probability distribution, based on cv signal. 
    When there are multiple constraint signals, `aggregate_method` describe how to combine them.
    """
    cv_aggregators = {
        'sum': lambda x: th.sum(th.clip(x, min=0), dim=1),
        'max': lambda x: th.max(th.clip(x, min=0), dim=1)[0]
    }

    def __init__(self, constraint: BaseConstraint, mollifier_sigma, aggregate_method: str="sum", scale_function= None):
        super().__init__()
        self.mollifier_sigma = mollifier_sigma
        self.constraint = constraint
        self.noise_prior = Normal(0, 1)
        self.aggregate_method = aggregate_method
        self.scale_function = scale_function

    def _get_cv(self, values: th.Tensor) -> th.Tensor:
        if self.scale_function is not None:
            values = self.scale_function(values)
        cv_all  = self.constraint.get_cv(values)
        return self.cv_aggregators[self.aggregate_method](cv_all)

    def log_prob(self, values: th.Tensor) -> th.Tensor:
        cv = self._get_cv(values)
        return self.noise_prior.log_prob(cv/self.mollifier_sigma)


class NormalPrior(BaseDistribution):
    def __init__(self, constraint: BaseConstraint, device):
        self.constraint = constraint
        self.noise_prior = MultivariateNormal(th.zeros(constraint.var_count, device=device), th.eye(constraint.var_count, device=device))
    
    def log_prob(self, values: th.Tensor) -> th.Tensor:
        return self.noise_prior.log_prob(values[:, :self.constraint.var_count])
        


class ConstrainedExponentialDistribution(ConstrainedDistribution):
    """
    An un-normalized mollified probability distribution, based on cv signal. 
    Instead of using a Gaussian noise, this distribution uses an exponential distribution.
    """
    
    def __init__(self, constraint: BaseConstraint, mollifier_sigma, aggregate_method: str = "sum", scale_function=None):
        super().__init__(constraint, mollifier_sigma, aggregate_method, scale_function)
        self.noise_prior = Exponential(1)

