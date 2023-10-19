from setuptools import setup

setup(name='jaco_gym',
      version='0.0.1',
      install_requires=['gym==0.15.7',
          'scipy>=1.2.0',
          #'stable-baselines[mpi]',
          'markdown',
          'rospkg',
          'numpy==1.19.5',
          'box2d', 
          'box2d-kengz', 
          'pyyaml', 
          'pytz==2020.1',
          'optuna', 
          'pytablewriter',
          'tensorflow-gpu==2.2.0']
)
