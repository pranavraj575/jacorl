from collections import OrderedDict, deque
import os
import numpy as np
import warnings
import time

from rlkit.data_management.replay_buffer import ReplayBuffer


class ComplexReplayBuffer(ReplayBuffer): # saves on disk instead, slower access but more memory

    def __init__(
        self,
        max_replay_buffer_size,
        observation_dim,
        action_dim,
        env_info_sizes,
        directory,
        reload_dir=True, # if we want to reload previous memory from directory
        replace = True,
    ):
        
        self._observation_dim = observation_dim
        self._action_dim = action_dim
        self._max_replay_buffer_size = max_replay_buffer_size
        
        self.save_dir=directory
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
            history=[] #newly initialized
        elif reload_dir: # reload the directory in this case
            history=os.listdir(self.save_dir)
            history.sort() # since they are ordered by time
            print("RELOADING BUFFER OF SIZE",len(history),"FROM:",self.save_dir)
        else:
            print("DELETING:",self.save_dir)
            self._clear_all_from_mem()
            history=[] # newly initialized
        self._file_history=deque(history,maxlen=self._max_replay_buffer_size)
        self._size=len(self._file_history)
        self._env_info_keys = list(env_info_sizes.keys())
        self._replace = replace
    
    def _clear_from_mem(self,filee):
        os.remove(os.path.join(self.save_dir,filee))
    
    def _clear_all_from_mem(self):
        old_files=os.listdir(self.save_dir)
        for element in old_files:
            if element.endswith('.npz'):
                self._clear_from_mem(element)
    
    def add_sample(self, observation, action, reward, next_observation,
                   terminal, env_info, **kwargs):
        name=str(time.time()).replace('.','_')+'.npz'
        np.savez(os.path.join(self.save_dir,name),
                  observation=observation,
                  action=action.reshape(-1), # makes these arrays, so dimensionality is consistent
                  reward=reward.reshape(-1),
                  next_observation=next_observation,
                  terminal=terminal.reshape(-1),
                  env_info=env_info)
        if self._size < self._max_replay_buffer_size:
            self._size += 1
        else:
            del_file=self._file_history.popleft()# oldest file
            self._clear_from_mem(del_file) # delete file
        self._file_history.append(name)# adds to end of file history
        
        
        #self._observations[self._top] = observation
        #self._actions[self._top] = action
        #self._rewards[self._top] = reward
        #self._terminals[self._top] = terminal
        #self._next_obs[self._top] = next_observation

        #for key in self._env_info_keys:
        #    self._env_infos[key][self._top] = env_info[key]
        #self._advance()

    def terminate_episode(self):
        pass

    def clear(self):
        #self._top = 0
        self._size = 0
        self._clear_all_from_mem()
        self._episode_starts = []
        self._cur_episode_start = 0

    #def _advance(self):
    #    self._top = (self._top + 1) % self._max_replay_buffer_size
    #    if self._size < self._max_replay_buffer_size:
    #        self._size += 1

    def random_batch(self, batch_size):
        indices = np.random.choice(self._size, size=batch_size, replace=self._replace or self._size < batch_size)
        if not self._replace and self._size < batch_size:
            warnings.warn('Replace was set to false, but is temporarily set to true because batch size is larger than current size of replay.')
        OBS=[]
        A=[]
        R=[]
        TERM=[]
        OBS_P=[]
        dict_stuff={k:[] for k in self._env_info_keys}
        for index in indices:
            fn=os.path.join(self.save_dir,self._file_history[index])
            f=np.load(fn,allow_pickle=True)
            obs,a,r,term,obs_p,e_info=f['observation'],f['action'],f['reward'],f['terminal'],f['next_observation'],f['env_info']
            # note we index env_info since it is a dict, so it is saved as an object array
            OBS.append(obs)
            A.append(a)
            R.append(r)
            TERM.append(term)
            OBS_P.append(obs_p)
            for key in self._env_info_keys:
                dict_stuff[key].append(e_info[0][key])
        batch=dict(
            observations=np.array(OBS),
            actions=np.array(A),
            rewards=np.array(R),
            terminals=np.array(TERM),
            next_observations=np.array(OBS_P),
        )
        #TODO: here take indices and access the files
        #batch = dict(
        #    observations=self._observations[indices],
        #    actions=self._actions[indices],
        #    rewards=self._rewards[indices],
        #    terminals=self._terminals[indices],
        #    next_observations=self._next_obs[indices],
        #)
        for key in self._env_info_keys:
            assert key not in batch.keys()
            batch[key] = np.array(dict_stuff[key])
        return batch

    def rebuild_env_info_dict(self, idx):
        fn=os.path.join(self.save_dir,self._file_history[idx])
        return np.load(fn,allow_pickle=True)['env_info'][0]
        #return {
        #    key: self._env_infos[key][idx]
        #    for key in self._env_info_keys
        #}

    def batch_env_info_dict(self, indices):
        env_infos={k:[] for k in self._env_info_keys}
        for index in indices:
            fn=os.path.join(self.save_dir,self._file_history[index])
            env_info=np.load(fn,allow_pickle=True)['env_info'][0]
            for k in self._env_info_keys:
                env_infos[k].append(env_info[k])
        return {
            key: np.array(env_infos[key])
            for key in self._env_info_keys
        }

    def num_steps_can_sample(self):
        return self._size

    def get_diagnostics(self):
        return OrderedDict([
            ('size', self._size)
        ])


class SimpleReplayBuffer(ReplayBuffer):

    def __init__(
        self,
        max_replay_buffer_size,
        observation_dim,
        action_dim,
        env_info_sizes,
        replace = True,
    ):
        self._observation_dim = observation_dim
        self._action_dim = action_dim
        self._max_replay_buffer_size = max_replay_buffer_size
        self._observations = np.zeros((max_replay_buffer_size, observation_dim))
        # It's a bit memory inefficient to save the observations twice,
        # but it makes the code *much* easier since you no longer have to
        # worry about termination conditions.
        self._next_obs = np.zeros((max_replay_buffer_size, observation_dim))
        self._actions = np.zeros((max_replay_buffer_size, action_dim))
        # Make everything a 2D np array to make it easier for other code to
        # reason about the shape of the data
        self._rewards = np.zeros((max_replay_buffer_size, 1))
        # self._terminals[i] = a terminal was received at time i
        self._terminals = np.zeros((max_replay_buffer_size, 1), dtype='uint8')
        # Define self._env_infos[key][i] to be the return value of env_info[key]
        # at time i
        self._env_infos = {}
        for key, size in env_info_sizes.items():
            self._env_infos[key] = np.zeros((max_replay_buffer_size, size))
        self._env_info_keys = list(env_info_sizes.keys())

        self._replace = replace

        self._top = 0
        self._size = 0

    def add_sample(self, observation, action, reward, next_observation,
                   terminal, env_info, **kwargs):
        self._observations[self._top] = observation
        self._actions[self._top] = action
        self._rewards[self._top] = reward
        self._terminals[self._top] = terminal
        self._next_obs[self._top] = next_observation

        for key in self._env_info_keys:
            self._env_infos[key][self._top] = env_info[key]
        self._advance()

    def terminate_episode(self):
        pass

    def clear(self):
        self._top = 0
        self._size = 0
        self._episode_starts = []
        self._cur_episode_start = 0

    def _advance(self):
        self._top = (self._top + 1) % self._max_replay_buffer_size
        if self._size < self._max_replay_buffer_size:
            self._size += 1

    def random_batch(self, batch_size):
        indices = np.random.choice(self._size, size=batch_size, replace=self._replace or self._size < batch_size)
        if not self._replace and self._size < batch_size:
            warnings.warn('Replace was set to false, but is temporarily set to true because batch size is larger than current size of replay.')
        batch = dict(
            observations=self._observations[indices],
            actions=self._actions[indices],
            rewards=self._rewards[indices],
            terminals=self._terminals[indices],
            next_observations=self._next_obs[indices],
        )
        for key in self._env_info_keys:
            assert key not in batch.keys()
            batch[key] = self._env_infos[key][indices]
        return batch

    def rebuild_env_info_dict(self, idx):
        return {
            key: self._env_infos[key][idx]
            for key in self._env_info_keys
        }

    def batch_env_info_dict(self, indices):
        return {
            key: self._env_infos[key][indices]
            for key in self._env_info_keys
        }

    def num_steps_can_sample(self):
        return self._size

    def get_diagnostics(self):
        return OrderedDict([
            ('size', self._size)
        ])
