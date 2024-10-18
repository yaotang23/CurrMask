import numpy as np
import scipy 

class TeacherExp3(object):
    """Teacher with Exponential-weight algorithm for Exploration and Exploitation.
    """

    def __init__(self, tasks, gamma=0.3,mode='single',num_sub_task=1,init_mode='1'):
        self._tasks = tasks
        self._n_tasks = len(self._tasks)
        self._gamma = gamma
        self.init_mode = init_mode
        self._log_weights = np.zeros(self._n_tasks)
        self._mode = mode 
        self._num_sub_task = num_sub_task


    @property
    def task_probabilities(self):
        gamma_tmp = 0.2
       # weights = np.exp(self._log_weights - np.sum(self._log_weights))
        weights = scipy.special.softmax(self._log_weights)
        probs = ((1 - gamma_tmp)*weights / np.sum(weights) +
            gamma_tmp/self._n_tasks)
        return probs
    
    @property
    def info(self):
        return self._mode,self._num_sub_task

    def get_task(self):
        """Samples a task, according to current Exp3 belief.
        """
        if self._mode =='single':
            task_i = np.random.choice(self._n_tasks, p=self.task_probabilities)
            return self._tasks[task_i]
        elif self._mode == 'single_random_ratio':
            task_i = np.random.choice(self._n_tasks, p=self.task_probabilities)
            mask_len,zero = self._tasks[task_i]
            mask_ratio = np.random.choice([0.15,0.35, 0.55, 0.75, 0.95])
            return mask_len,mask_ratio
        else: #mode='multi': multi-task in an arm
            task_i = np.random.choice(self._n_tasks, p=self.task_probabilities)
            sub_order,mask_ratio = self._tasks[task_i]
            mask_len = np.random.randint((sub_order-1)*self._num_sub_task+1,sub_order*self._num_sub_task+1)
            return mask_len,mask_ratio


    @staticmethod
    def normalize_reward(reward, hist_rewards):
        assert len(hist_rewards) >= 2

        low_percentile = np.percentile(hist_rewards, 20)
        high_percentile = np.percentile(hist_rewards, 80)

        if reward < low_percentile:
            return -1
        if reward > high_percentile:
            return 1

        return 2 * (reward - low_percentile) / (high_percentile - low_percentile) - 1

    def update(self, task, reward):
        """ Updates the weight of task given current reward observed
        """
        task_i = self._tasks.index(task)
        reward_corrected = reward/self.task_probabilities[task_i] 
        self._log_weights[task_i] += self._gamma*reward_corrected/self._n_tasks  