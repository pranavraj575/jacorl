from gym.envs.registration import register

# basic environment, use for testing
register(
    id='BasicJacoEnv-v0',
    entry_point='jaco_gym.envs.robot_env:JacoEnv',
    max_episode_steps=50
)

# basic sim_environment
register(
    id='BasicJacoGazebo-v0',
    entry_point='jaco_gym.envs.gazebo_env:JacoGazeboEnv',
    max_episode_steps=50
)

# cup single grab simulation
register(
    id='JacoCupGrabGazebo-v0',
    entry_point='jaco_gym.envs.task_envs.single_cup_grasp_gazebo:JacoGrabCupGazebo',
    max_episode_steps=50
)

# cup multiple grab simulation
register(
    id='JacoMultiCupGrabGazebo-v0',
    entry_point='jaco_gym.envs.task_envs.multi_random_cup_grasp:JacoMultiGrabCupGazebo',
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

# cup stacky simulation with images
register(
    id='JacoJengaGazebo-v0',
    entry_point='jaco_gym.envs.task_envs.jengazebo:JacoJengaZebo',
    max_episode_steps=50
)