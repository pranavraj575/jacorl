################################################################################
###| SAC_IMG.PY - Runs Soft Actor Critic RL training on your enviornment    |###
###|            - model framework from rlkit directory                      |###
###|            - modified to take in a camera image along with robot       |###
###|              joint data as the observation, runs image through CNN     |###
################################################################################

import gym
import jaco_gym
import random
import numpy as np 
import rospy
import rlkit.torch.pytorch_util as ptu
from rlkit.data_management.env_replay_buffer import SlowEnvReplayBuffer
from jaco_gym.envs.task_envs.stack_cups_gazebo_img import JacoStackCupsGazeboImg

from rlkit.launchers.launcher_util import setup_logger
from rlkit.samplers.data_collector import MdpPathCollector
from rlkit.torch.sac.policies import TanhCNNGaussianPolicy, MakeDeterministic
from rlkit.torch.sac.sac import SACTrainer
from rlkit.torch.networks import ConcatCNN
from rlkit.torch.torch_rl_algorithm import TorchBatchRLAlgorithm
import os

def experiment(variant):
    expl_env = JacoStackCupsGazeboImg() #NormalizedBoxEnv(HalfCheetahEnv())
    eval_env = JacoStackCupsGazeboImg() #NormalizedBoxEnv(HalfCheetahEnv())
    obs_dim = expl_env.observation_space.low.size
    action_dim = eval_env.action_space.low.size

    width,height,channels=expl_env.image_dim
    M = variant['layer_size']
    #concat since the input looks like (obs, action) with two thingies
    qf1 = ConcatCNN( 
            input_width=width,
            input_height=height,
            input_channels=channels,
            output_size=1,
            kernel_sizes=[3,3,4],
            n_channels=[32,64,128],
            strides=[2,2,2],
            paddings=[0,0,0],
            hidden_sizes=[M, M],
            added_fc_input_size = obs_dim + action_dim - width*height*channels, # since these elements will be the image
            pool_type='max2d',
            pool_sizes=[2,2,2],
            pool_strides=[2,2,2],
            pool_paddings=[0,0,0],
    )
    qf2 = ConcatCNN(
            input_width=width,
            input_height=height,
            input_channels=channels,
            output_size=1,
            kernel_sizes=[3,3,4],
            n_channels=[32,64,128],
            strides=[2,2,2],
            paddings=[0,0,0],
            hidden_sizes=[M, M],
            added_fc_input_size = obs_dim + action_dim - width*height*channels, # since these elements will be the image
            pool_type='max2d',
            pool_sizes=[2,2,2],
            pool_strides=[2,2,2],
            pool_paddings=[0,0,0],
    )
    target_qf1 = ConcatCNN(
            input_width=width,
            input_height=height,
            input_channels=channels,
            output_size=1,
            kernel_sizes=[3,3,4],
            n_channels=[32,64,128],
            strides=[2,2,2],
            paddings=[0,0,0],
            hidden_sizes=[M, M],
            added_fc_input_size = obs_dim + action_dim - width*height*channels, # since these elements will be the image
            pool_type='max2d',
            pool_sizes=[2,2,2],
            pool_strides=[2,2,2],
            pool_paddings=[0,0,0],
    )
    target_qf2 = ConcatCNN(
            input_width=width,
            input_height=height,
            input_channels=channels,
            output_size=1,
            kernel_sizes=[3,3,4],
            n_channels=[32,64,128],
            strides=[2,2,2],
            paddings=[0,0,0],
            hidden_sizes=[M, M],
            added_fc_input_size = obs_dim + action_dim - width*height*channels, # since these elements will be the image
            pool_type='max2d',
            pool_sizes=[2,2,2],
            pool_strides=[2,2,2],
            pool_paddings=[0,0,0],
    )
    policy = TanhCNNGaussianPolicy(
            
            # same cnn arguments except outputs actions, and does not take in actions
            # note that this does not necessarily need to be true, just no reason to change it for now
            input_width=width,
            input_height=height,
            input_channels=channels,
            output_size=action_dim, #different
            kernel_sizes=[3,3,4],
            n_channels=[32,64,128],
            strides=[2,2,2],
            paddings=[0,0,0],
            hidden_sizes=[M, M],
            added_fc_input_size = obs_dim  - width*height*channels, #different
            pool_type='max2d',
            pool_sizes=[2,2,2],
            pool_strides=[2,2,2],
            pool_paddings=[0,0,0],
    )
    eval_policy = MakeDeterministic(policy)
    eval_path_collector = MdpPathCollector(
        eval_env,
        eval_policy,
    )
    expl_path_collector = MdpPathCollector(
        expl_env,
        policy,
    )
    replay_buffer = SlowEnvReplayBuffer(
        variant['replay_buffer_size'],
        expl_env,
        directory=os.path.join(variant['save_dir'],'replay_buff')
    )
    trainer = SACTrainer(
        env=eval_env,
        policy=policy,
        qf1=qf1,
        qf2=qf2,
        target_qf1=target_qf1,
        target_qf2=target_qf2,
        **variant['trainer_kwargs']
    )
    algorithm = TorchBatchRLAlgorithm(
        trainer=trainer,
        exploration_env=expl_env,
        evaluation_env=eval_env,
        exploration_data_collector=expl_path_collector,
        evaluation_data_collector=eval_path_collector,
        replay_buffer=replay_buffer,
        **variant['algorithm_kwargs']
    )
    algorithm.set_output_dir(variant['save_dir'])
    algorithm.to(ptu.device)
    #algorithm._end_epoch(0)
    algorithm.train()

if __name__ == "__main__":
    # noinspection PyTypeChecker
    variant = dict(
        save_dir=os.path.join(os.getcwd(),'cloutput/CNNTest'),
        algorithm="SAC",
        version="normal",
        layer_size=256,
        replay_buffer_size=int(1E6), # CHANGED: shrink because too big
        algorithm_kwargs=dict(
            num_epochs=3000,
            num_eval_steps_per_epoch=2500, # num of steps that evaluation happens on
            num_trains_per_train_loop=500, # number of times a batch of replay is selected for training
            num_expl_steps_per_train_loop=500, # number of exploration steps taken per training loop
            min_num_steps_before_training=1000, # fill replay buffer by this many steps before training
            max_path_length=500, # episode ends after this many
            batch_size=256, # batch size for training
        ),
        trainer_kwargs=dict(
            discount=0.99,
            soft_target_tau=5e-3,
            target_update_period=1,
            policy_lr=3E-4,
            qf_lr=3E-4,
            reward_scale=1,
            use_automatic_entropy_tuning=True,
        ),
    )
    rospy.init_node("sac_client")
    env_id = 'JacoCupsGazeboImg-v0'
    env = gym.make(env_id)
    env.reset()
    experiment(variant)
    env.close()
    setup_logger('name-of-experiment', variant=variant)
    ptu.set_gpu_mode(True)  # optionally set the GPU (default=False)
