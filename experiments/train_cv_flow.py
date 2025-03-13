import argparse
from dataclasses import dataclass
from experiments.common.setup_experiment import setup_experiment, flush_logs, get_value_logger
from experiments.problems import all_problems, BaseProblem
from core.constraints import BaseConstraint, CombinedConstraint
from typing import Literal
import os
from core.flow.real_nvp import RealNvp
from core.flow.train_flow import update_flow_batch
from core.flow.constrained_distribution import ConstrainedExponentialDistribution
from core.flow.data_loaders import UniformDataLoader, GaussianDataLoader, CombinedDataLoader
import torch as th
import time


import numpy as np

@dataclass
class Options:
    problem: str
    data_file: str # For testing recall calculation
    action_base_distribution: Literal['uniform', 'gaussian']
    test_sample_count: int = 100_000
    epochs: int = 500
    eval_freq: int = 10
    device: str = 'cpu'
    lr: float = 1e-5
    batch_size: int = 256 
    batches_per_epch: int = 1000 
    hidden_size: int = 256
    transform_count: int = 6
    mollifier_sigma: float = 0.00001
    gradient_clip_value: float = 0.1
    take_log_again: bool = False
    s_scaling_factor: float = 1.0

def get_combined_data_loader(params: Options, prob: BaseProblem, seed):
    constraint = prob.constraint
    state_data_loader = prob.get_state_data_loader(params.batch_size, params.batches_per_epch, params.device, seed) 
    if params.action_base_distribution == 'gaussian':
        action_data_loader = GaussianDataLoader(params.batch_size, params.batches_per_epch, constraint.var_count, params.device, seed)
    elif params.action_base_distribution == 'uniform':
        action_data_loader = UniformDataLoader(params.batch_size, params.batches_per_epch, constraint.var_count, params.device, seed, [-1, 1])

    return CombinedDataLoader(params.batch_size, params.batches_per_epch, constraint.dim, params.device, seed, [action_data_loader, state_data_loader])

def get_constraint_with_action_bounds(constraint: BaseConstraint, action_bound_constraint: BaseConstraint):
    conditional_p_count = getattr(constraint, 'conditional_param_count', 0)
    return CombinedConstraint(constraint.var_count, conditional_p_count, [constraint, action_bound_constraint], [th.LongTensor(range(constraint.dim)), th.LongTensor(range(constraint.var_count))])

def get_test_data_x(test_data_loader, test_sample_count):
    test_data_lst, test_data_count = [], 0
    for batch in test_data_loader:
        test_data_lst.append(batch)
        test_data_count += len(batch)
        if test_data_count >= test_sample_count:
            break
    return th.concat(test_data_lst)[:test_sample_count]

def main():
    """
    Train flow forwad using generated samples from a file.
    """
    args = setup_experiment("train_flow_backward", Options)
    logger = get_value_logger(args.log_dir)
    params: Options = args.params
    print("batches", params.batches_per_epch)
    flush_logs()

    # Get the constraint
    problem:BaseProblem  = all_problems[params.problem]
    problem.constraint = problem.constraint.to(params.device)
    action_bound_constraint = problem.action_bound_constraint.to(params.device)
    constraint = problem.constraint
    conditional_p_count = getattr(constraint, 'conditional_param_count', 0)

    # Define the flow model
    flow = RealNvp(constraint.var_count, params.transform_count, conditional_param_count=conditional_p_count, hidden_size=params.hidden_size, s_scaling_factor=params.s_scaling_factor).to(params.device)
    optimizer = th.optim.Adam([p for p in flow.parameters() if p.requires_grad == True], lr=params.lr)

    # Load dataset for testing, recall
    data = th.from_numpy(np.load(params.data_file)).double().to(params.device)
    if params.test_sample_count > len(data):
        logger.warn("Not enough samples in the dataset")
        # raise ValueError("Not enough samples in the dataset")

    test_data_z = data[: params.test_sample_count]
    test_data_z = problem.scale_action(test_data_z)

    # Random data loader
    train_data_loader = get_combined_data_loader(params, problem, 0)
    test_data_loader = get_combined_data_loader(params, problem, 1)
    test_data_x = get_test_data_x(test_data_loader, params.test_sample_count)

    # Constraint based mollifier
    constraint_with_action_bounds = get_constraint_with_action_bounds(constraint, action_bound_constraint).to(params.device)
    prior = ConstrainedExponentialDistribution(constraint_with_action_bounds, params.mollifier_sigma, scale_function=problem.unscale_action)

    os.makedirs(args.log_dir + "/figures", exist_ok=True)
    print(f"Runing for: {params.problem}")


    start_time = time.time()
    for epoch in range(params.epochs):
        losses = []
        # Update flow for each batch
        for batch in train_data_loader:
            loss = update_flow_batch(flow, prior, batch, optimizer, gradient_clip_value=params.gradient_clip_value, take_log_again=params.take_log_again)
            losses.append(loss)

        # Evaluate
        if (epoch+1)%params.eval_freq == 0:
            with th.no_grad():
                logger.record("train/mean_loss", np.mean(losses))

                # Calculate accuracy infr direction (x -(f)-> z) 
                generated_samples = flow.f(test_data_x)[0]
                validity = problem.is_feasible(generated_samples)
                valid_count = validity.int().sum().item()
                accuracy = valid_count/len(validity)
                logger.record("train/accuracy", accuracy)

                # Calculate recall
                mapped_x = flow.g(test_data_z)[0]
                mapped_x_action = mapped_x[:, : constraint.var_count] # only take action variables
                if params.action_base_distribution == 'uniform':
                    validity_z = th.all(mapped_x_action >= -1.0, dim=1) & th.all(mapped_x_action <= 1., dim=1)
                    valid_z_count = validity_z.int().sum().item()
                    recall = valid_z_count/len(validity_z)
                    logger.record("train/recall", recall)
                elif params.action_base_distribution == 'gaussian':
                    norm = mapped_x_action.norm(dim=1)
                    for i in range(1, 11):
                        validity_z = norm < i
                        valid_z_count = validity_z.int().sum().item()
                        recall = valid_z_count/len(validity_z)
                        logger.record(f"train/recall_sigma_{i}", recall)
                    fig = problem.plot(mapped_x_action)
                    fig.savefig(f"{args.log_dir}/figures/{epoch+1:05d}_inverse.png")
                else:
                    raise ValueError("Invalid action base distribution")


                fig = problem.plot(generated_samples, validity)
                fig.savefig(f"{args.log_dir}/figures/{epoch+1:05d}.png")

            elapsed_time = time.time() - start_time
            logger.record("train/time_elapsed", elapsed_time)
            print(f"Epoch: {epoch+1}: Mean loss {np.mean(losses):.4f}, Acc: {accuracy*100: .2f}%, Recall: {recall*100: .2f}%")
            flow.save_module(f"{args.log_dir}/model.pt")
            logger.record("train/epoch", epoch+1)
            flush_logs()
            logger.dump(epoch)

if __name__ == "__main__":
    main()