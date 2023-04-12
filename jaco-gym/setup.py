from setuptools import setup

setup(name='jaco_gym',
      version='0.0.1',
      install_requires=['gym==0.15.7',
          'scipy>=1.2.0',
          'stable-baselines[mpi]',
          'rospkg',
          'box2d', 
          'box2d-kengz', 
          'pyyaml', 
          'optuna', 
          'pytablewriter',
          'tensorflow-gpu==1.14']
)
