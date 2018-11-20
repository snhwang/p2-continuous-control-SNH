# Project 2: Continuous Control -- Report

This project was one of the requirements for completing the Deep Reinforcement Learning Nanodegree (DRLND) course at Udacity.com. In short, an agent represented by a double jointed arm is trained to follow a target within its hand. A reward of +0.1 is provided for each step that the agent's hand is at the target location. Thus, the goal of each agent is to maintain its position at the target location for as many time steps as possible. The environment was provided with 2 variations: one with a single agent and one with 20 agents. Please refer to README.md for more details about the environment and installation. As also stated in the README file, the goal was to obtain a score of at least 30 averaged over 100 episodes (and over all agents if more than one). 

## Implementation

The implementation was adapted from the course materials. Specifically, it was based on the code for the pendulum environment from the Open AI Gym provided at the course repository (https://github.com/udacity/deep-reinforcement-learning/tree/master/ddpg-pendulum).

The environment was created using Unity's ML Agents toolkit, but the learning algorithms were implemented with python 3.6 (specifically 3.6.6). The code was run using Jupyter notebook.

I utilized the deep deterministic policy gradient (DDPG) algorithm, which is a variant of the actor-critic learning method. An actor neural network predicts an action and the critic network models the goodness of the choice (i.e., Q value). This algorithm is discussed in more detail in Lillicrap et al. (Continuous Control with Deep Reinforcement Learning, https://arxiv.org/pdf/1509.02971.pdf).

A few of the features of the algorithm are discussed in further detail below.

#### Soft Updating

During training, the neural network weights are updated. This can become unstable or oscillatory. To reduce this, a method of "soft updating" is utilized. Rather than completely replacing the old weights with the new weights, the soft update method gradually blends in the local network to the target network. Basically it is a weighted average, strongly weighted to the original network so that changes occur gradually. The weight for the target network is `tau`, and the weight for the local network is thus `1-tau`:

`target_parameters(k+1) = tau * local_parameters(k) + (1-tau) * target_parameters(k)`,

where k is the timestep.

Smaller `tau` thus indicates the the weighting is shifted more towards the current target model. Since the weights are initialized randomly at the start of training, the weights early in training are likely not as useful. So, I modified the code to allow for starting with a high value of `tau` which decays towards a smaller value.

#### Action Noise

On of the problems of learning in continuous action spaces is enabling the learning agent to adequately explore the space in order to generate an adequate model. One method to achieve this is to add noise to the action. As was done by Lillicrap et al., an Ornstein-Uhlenbeck noise process was utilized. In the pendulum code, one of the terms for the noise process is a sampled for a uniform distribution. I changed this to a normal variate, which I think is the correct method. In any case, this change seems to help the learning converge faster. The two noise parameters that I varied were theta and sigma. In an Ornstein–Uhlenbeck process, theta is the rate at which a perturbation to the system reverts towards the mean, which is assumed to be zero for our purposes. The parameter sigma, which is the standard deviation of the normal variate component, is a measure of the volatility of the perturbations. 

​	The default values provided in the code for the pendulum environment were theta = 0.15 and sigma = 0.2 I tried a wide range of values above and below these values. Early on in this project, I observed that my agent or agents would often fail by get stuck repeating the same motion, usually a constant circling in the same plane with "arm" outstretched. I found that I had much more success by using relatively large values of theta = 0.3 to 0.5 and sigma = 0.6. Perhaps large perturbations helped prevent the agents from getting stuck in a local minimum. I thought that the large amount of noise might interfere with learning convergence, so I included a scaling parameter that would gradually decrease the magnitude of the added noise at each episode of training.

#### Gradient Clipping

As suggested by the "Benchmark Implementation" in the Udacity course materials for the project, gradient clipping was utilized in the critic network to improve stability of the learning process. The magnitude of the gradients are limited to 1.

#### Experience Replay

The original code included an implementation of experience replay, which stores prior states and actions. Training is performed on a random sampling of the the stored items. The basic version uniformly samples the items. I was going to implement a prioritized version of the experience replay as I had for our previous project on deep Q learning. In prioritized experience replay, each item is weighted by a function of its error. Since the agent may be able to learn more from the actions which result in a relatively larger error, those with larger errors are weighted more during sampling of the replay memory buffer. However, my results for the current project were satisfactory without, so I did not proceed with this. I hope to implement this later to see if it improves the learning.

The benchmark implementation described in the Udacity course materials suggested updating the networks 10 times after every 20 timesteps. This presumably allows the algorithm to sample relatively uncorrelated data in the buffer since it was accumulated over more timesteps. I varied the interval for updating from 1 to 20 timesteps and also adjusted the number of updates per interval. I obtained good results updating only 5 times every 20 timesteps.

The very first update during training is delayed until there is adequate data stored in the experience replay buffer for training in the neural networks. The original code started training after enough data was obtained for one mini batch. I extended this to 10 times the batch size to accumulated more data before training. This may be excessive but the learning worked, so I did not further adjust this.

#### Evolving Learning Parameters


Gamma is the discount factor for Q learning for indicating a preference for current rewards over potential future rewards. In the setting of deep Q learning, Francois-Lavet et al. (How to Discount Deep Reinforcement Learning: Towards New Dynamic Strategies, https://arxiv.org/abs/1512.02011) recommend gradually increasing gamma as learning progresses. I though this may also hold true for DDPG. I used a variation of their method and allow gamma to increase according to:

`gamma(k + 1) = gamma_final + (1 - gamma_rate) * (gamma_final - gamma(k))`

The following were my final choices for values but I tried different values for the rate:

```
gamma_initial = 0.95,
gamma_final = 0.99,
gamma_rate = 0.01
```

As I mentioned above that I allowed  `tau`, the weighting factor for soft updating, to evolve as well:

`tau(k + 1) = tau_final + (1 - tau_rate) * (tau_final - tau(k))`

In the previously project with deep Q learning, I was having problems with the improvement in learning leveling off too early. It seemed to help to have `tau` gradually increase so that the changes in the network would be greater at later stages. However, this could also conceivably result in more instability, in which having `tau` decay would be more preferable. I suspect that it may depend on the specific learning task at hand but have not yet explored this parameter adequately. When using RELU activations, a relatively larger `tau` of 0.01 compared to the value of 0.001 in the original code seemed to work better. For this project, I made tau decay from 0.01 to 0.001.

For decaying the magnitude of noise, I just use as multiplicative factor, where factor < 1:

`scaling_factor(k+1) = scaling_factor(k) * noise_factor`

The noise is then multiplied by `scaling_factor`.

#### The Neural Networks

Lillicrap et al., suggest using batch normalization for the state input and between every layer in the actor network. For the critic network, they suggest using batch normalization at every layer before the the action input. I tried doing it this way, but my impression for our current learning task is that having batch normalization just at the state input works as well or even better, although I would have to more formally investigate this to be sure.

When I use RELU activations I settled on the following actor neural network structure:

1. Batch normalization of the state input (size 33)
2. fully connected layer  with a number of inputs equal to the state size and 128 outputs
3. Rectified Linear Unit (ReLU)
4. fully connected layer (128 in and 64 out)
5. ReLU
6. fully connected layer(64 in. The number of outputs is the action size)

The critic network was similar:

1. Batch normalization of the state input (size 33)
2. fully connected layer  with a number of inputs equal to the state size and 128 outputs
3. Concatenate the action
4. Rectified Linear Unit (ReLU)
5. fully connected layer. The number of inputs is 128 + the action size.  64 outputs
6. ReLU
7. fully connected layer(64 in and a single output representing the Q value)

When I used SELU activations, the actor network had the following structure:

1. fully connected layer  with a number of inputs equal to the state size and 128 outputs
2. SELU
3. fully connected layer (128 in and 64 out)
4. SELU
5. fully connected layer (64 in and 32 out)
6. SELU
7. fully connected layer (32 in and action size out)

The critic with SELU:

1. fully connected layer  with a number of inputs equal to the state size and 128 outputs
2. SELU
3. Concatenate the action
4. fully connected layer (inputs: 128 + action size; outputs: 64)
5. SELU
6. fully connected layer (64 in and 32 out)
7. SELU
8. fully connected layer (32 in and 1 out)

The learning utilizes the Adam optimizer for both types.

## Learning Algorithm

#### Agent parameters

The following parameters determine the learning agent:

```
    state_size: Number of parameters defining the environmen state
    action_size: Number of pameters definine the actions
    num_agents: Number of learning agents
    random_seed: Random seed number
    batch_size: Batch size for neural network training
    lr_actor: Learning rate for the actor neural network
    lr_critic: Learning rate for the critic neural network
    noise_theta (float): theta for Ornstein-Uhlenbeck noise process
    noise_sigma (float): sigma for Ornstein-Uhlenbeck noise process
    actor_fc1 (int): Number of hidden units in the first fully connected layer of the
    	actor network
    actor_fc2: Units in second layer
    actor_fc3: Units in third fully connected layer. This parameter does nothing for
    	the "RELU" network
    critic_fc1: Number of hidden units in the first fully connected layer of the critic
    	network
    critic_fc2: Units in second layer
    critic_fc3: Units in third layer. This parameter does nothing for the "RELU" 			network
    update_every: The number of time steps between each updating of the neural networks 
    num_updates: The number of times to update the networks at every update_every 			interval
    buffer_size: Buffer size for experience replay. Default 2e6.
    network (string): The name of the neural networks that are used for learning.
        There are 	only 2 choices, one with only 2 fully connected layers and RELU
        activations and one with 3 fully connected layers with SELU activations.
        Their names are "RELU" and "SELU," respectively. Default is "RELU."

```

#### Training parameters

These parameters adjust the learning:

```
agent (Agent): The learning agent
checkpoint_name (string): A prefix string for naming all of the checkpoints of the actor and critic neural
    networks that are saved.
n_episodes (int): Maximum number of training episodes
max_t (int): Maximum number of timesteps per episode
gamma_initial (float): Initial gamma discount factor (0 to 1). Higher values favor long term over current rewards.
gamma_final (float): Final gamma discount factor (0 to 1).
gammma_rate (float): A rate (0 to 1) for increasing gamma.
tau_initial (float): Initial value for tau, the weighting factor for soft updating the neural networks.
tau_final (float): Final value of tau.
tau_rate (float): Rate (0 to 1) for increasing tau each episode.
noise_factor (float<=1): The value for scaling the noise every episode to gradually decrease it.
```

The training is performed by looping through multiple episodes for the environment.

For each episode

1. Get an action
2. Send the action to the environment
3. Get the next state
4. Get the reward
5. Add the state, action, reward, and next state to the experience replay buffer. 
6. Every 20 time steps in the environment, randomly sample the experience replay buffer and perform 5 learning steps. Each learning step consists of updating the critic network, then updating the actor network.
7. The reward is added to the score
8. Update the state to the next state and loop back to step 1.



This is implemented in section 3 of the Jupyter notebook, Continuous_Control_SNH.ipynb , as `ddpg`. Running `ddpg` returns the scores from all of the episodes.

The code runs for both version 1 (single agent) and version 2 (20 agents) of the project environment. However, with the parameters I provided, the RELU network does not achieve success with the single agent case. The SELU works for both versions but is significantly slower than RELU for the multi-agent version.



## Results and Plots of Rewards

I found this project challenging in terms of finding a set of parameters that worked well. Sometimes training would seem to be progressing well, then suddenly, would completely fail with the scores dropping to zero. I first achieved success with the SELU network, but then eventually achieved faster results by going back to the RELU network. 

The run plotted below used the following parameters:

`agent = Agent(
​    state_size = state_size,
​    action_size = action_size,
​    num_agents = num_agents,
​    random_seed = 0,
​    batch_size = 1024, 
​    lr_actor = 0.001,
​    lr_critic = 0.001,
​    noise_theta = 0.45,
​    noise_sigma = 0.6,
​    actor_fc1 = 128,
​    actor_fc2 = 64,
​    critic_fc1 = 128,
​    critic_fc2 = 64,
​    update_every = 20,
​    num_updates = 5,
​    buffer_size = int(2e6),
​    network = 'RELU'
)`

`scores = ddpg(
​    agent,
​    checkpoint_name = 'v2_RELU',
​    n_episodes = 150,
​    max_t = 10000,
​    gamma_initial = 0.95,
​    gamma_final = 0.99,
​    gamma_rate = 0.01,
​    tau_initial = 0.01,
​    tau_final = 0.001,
​    tau_rate = 0.01,
​    noise_factor = 0.995
)`



![img](data:image/png;base64,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)

A score of 30 was first surpassed after 28 episodes. During training, the score averaged over an episode and the 99 preceding episodes surpassed 30 at episode 110. The average score over 100 episodes using the final set of weights after 150 episodes of training was 38.97. The best 100-episode average achieved was also 34.87, which was the score for the very last episode. The weights were stored in the files v2_RELU_actor_checkpoint_final.pth and v2_RELU_critic_checkpoint_final.pth and should be identical to v2_RELU_actor_best_agent_average.pth and v2_RELU_critic_best_agent_average.pth.  A run of 100 episodes without training resulting in a 100-episode average of 38.72 as shown in the notebook. I had some runs that leveled off at at high level, around 38, but this run was the fastest to reach an agent-average score of >30. Its 100-episode average without further training was also one of the highest.

## Ideas for Future Work

A basic method for improvement would be to do a more systematic search for the optimal hyperparameters.

One particular aspect of the parameters that  I would be interested in investigating further is the action noise process. I found that the values of theta and sigma had a significant impact on the learning process. Rather than just scaling the noise with a single factor, adding control over theta and sigma individually would provide more options to optimize.

Also in regard to noise, adding parameter noise to the neural network weights in place of or in addition to noise to the actions may also provide benefit.

I was going to implement prioritized experience replay as this has been shown to improve learning but did not since I achieved the desired score without it. It would be interesting to see how much improvement could be obtained.

For the multi-agent version, other algorithms such as  [PPO](https://arxiv.org/pdf/1707.06347.pdf), [A3C](https://arxiv.org/pdf/1602.01783.pdf), and [D4PG](https://openreview.net/pdf?id=SyZipzbCb) use multiple, non-interacting, and parallel copies of the same agent to distribute the task of gathering experience.  These methods may provide improvements.



