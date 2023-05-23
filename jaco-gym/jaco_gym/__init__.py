from gym.envs.registration import register

# basic environment, use for testing
register(
    id='BasicJacoEnv-v0',
    entry_point='jaco_gym.envs.robot_env:JacoEnv',
    max_episode_steps=50
)

# cup stacky simulation
register(
    id='JacoCupsGazebo-v0',
    entry_point='jaco_gym.envs.task_envs.stack_cups_gazebo:JacoStackCupsGazebo',
    max_episode_steps=50
)

# cup stacky simulation with images
register(
    id='JacoCupsGazeboImg-v0',
    entry_point='jaco_gym.envs.task_envs.stack_cups_gazebo_img:JacoStackCupsGazeboImg',
    max_episode_steps=50
)
