# CV-Flow

The source code for the paper "[Leveraging Constraint Violation Signals For Action-Constrained Reinforcement Learning](https://arxiv.org/abs/2502.10431)".

## Training the CV Flow
The code to train the flow model using constraint violation signal can be found in `experiments/train_cv_flow.py`


## Integration with SAC
The modified SAC implementation with $|\hat{a}|^2$ to support Gaussian base distribution is available in `action_constrained_rl/sac/flow_sac.py`.  
Use the Additional Layer policy (`action_constrained_rl/nn/additional_layer_sac_policy.py`) with FlowLayer(`action_constrained_rl/nn/additional_layers/flow_layer.py`) as the additional layer. 

> The original code is from [omron-sinicx/action-constrained-RL-benchmark](https://github.com/omron-sinicx/action-constrained-RL-benchmark). Please refer to the [train.py](https://github.com/omron-sinicx/action-constrained-RL-benchmark/blob/master/train.py) on how to use additional layer policy.