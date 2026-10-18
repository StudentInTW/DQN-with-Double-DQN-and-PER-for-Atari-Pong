# Spring 2026, 535518 Deep Learning
# Lab5: Value-based RL
# Contributors: Kai-Siang Ma and Alison Wen
# Instructor: Ping-Chun Hsieh

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import gymnasium as gym
import cv2
import ale_py
import os
from collections import deque
import wandb
import argparse
import time
import torch.nn.functional as F

gym.register_envs(ale_py)


def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

class DQN(nn.Module):
    """
        Design the architecture of your deep Q network
        - Input size is the same as the state dimension; the output size is the same as the number of actions
        - Feel free to change the architecture (e.g. number of hidden layers and the width of each hidden layer) as you like
        - Feel free to add any member variables/functions whenever needed
    """
    def __init__(self, input_dim ,num_actions , mode = "cartpole"):
        super(DQN, self).__init__()
        self.mode = mode

        
        # An example: 
        #self.network = nn.Sequential(
        #    nn.Linear(input_dim, 64),
        #    nn.ReLU(),
        #    nn.Linear(64, 64),
        #    nn.ReLU(),
        #    nn.Linear(64, num_actions)
        #)       
        ########## YOUR CODE HERE (5~10 lines) ##########

        if mode == "cartpole":
            self.network = nn.Sequential(
                # 4 dim
                nn.Linear(input_dim , 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64 , num_actions)   
            )
        elif mode == "atari":
            self.network = nn.Sequential(
                #  4,84,84
                nn.Conv2d(input_dim, 32, kernel_size=8, stride=4),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=4, stride=2),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(64 * 7 * 7, 512),
                nn.ReLU(),
                nn.Linear(512, num_actions)
                )
            
        ########## END OF YOUR CODE ##########

    def forward(self, x):
        if self.mode =="atari":
            # do normalization for atari image pixels
            return self.network( x/ 255.0)
        else:
            return self.network(x)


class AtariPreprocessor:
    """
        Preprocesing the state input of DQN for Atari
    """    
    def __init__(self, frame_stack=4):
        self.frame_stack = frame_stack
        self.frames = deque(maxlen=frame_stack)

    def preprocess(self, obs):
        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
        return resized

    def reset(self, obs):
        frame = self.preprocess(obs)
        self.frames = deque([frame for _ in range(self.frame_stack)], maxlen=self.frame_stack)
        return np.stack(self.frames, axis=0)

    def step(self, obs):
        frame = self.preprocess(obs)
        self.frames.append(frame)
        return np.stack(self.frames, axis=0)


class PrioritizedReplayBuffer:
    """
        Prioritizing the samples in the replay memory by the Bellman error
        See the paper (Schaul et al., 2016) at https://arxiv.org/abs/1511.05952
    """ 
    def __init__(self, capacity, alpha=0.6, beta=0.4):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.buffer = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.pos = 0

    def add(self, transition, error):
        ########## YOUR CODE HERE (for Task 3) ########## 
        if error is None:
          priority = self.priorities.max() if self.buffer else 1.0
        else:
            priority = abs(error) + 1e-5

        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.pos] = transition

        self.priorities[self.pos] = priority
        self.pos = (self.pos + 1) % self.capacity

        ########## END OF YOUR CODE (for Task 3) ########## 
        return 
    def sample(self, batch_size):
        ########## YOUR CODE HERE (for Task 3) ########## 
        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:self.pos]
            
        # 1. 計算每個記憶的機率 P(i) = p_i^alpha / sum(p_k^alpha)
        probs = prios ** self.alpha
        probs /= probs.sum()
        
        # 2. 根據機率抽出 batch_size 個 index
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]
        
        # 3. 計算 Importance Sampling (IS) 權重，用來校正優先採樣帶來的偏差
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-self.beta)
        weights /= weights.max() # 正規化確保最大值為 1
        
        return samples, indices, np.array(weights, dtype=np.float32)

        ########## END OF YOUR CODE (for Task 3) ########## 
        return
    def update_priorities(self, indices, errors):
        ########## YOUR CODE HERE (for Task 3) ########## 
        # 每次訓練完，根據新的 TD Error 更新優先權：p_i = |error| + epsilon
        for idx, error in zip(indices, errors):
            self.priorities[idx] = abs(error) + 1e-5

        ########## END OF YOUR CODE (for Task 3) ########## 
        return

    def __len__(self):
      return len(self.buffer)
        

class DQNAgent:
    def __init__(self, env_name="CartPole-v1", args=None):
        self.env = gym.make(env_name, render_mode="rgb_array")
        self.test_env = gym.make(env_name, render_mode="rgb_array")
        self.num_actions = self.env.action_space.n
        
        # for task 3
        self.n_step = 3
        self.n_step_buffer  = deque(maxlen= self.n_step)


        self.mode = "cartpole" if "CartPole" in env_name else "atari"
        input_dim = 4 if self.mode == "cartpole" else 4
        self.preprocessor = AtariPreprocessor() if self.mode == "atari" else None
    
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device:", self.device)

        # initialize double q network
        self.q_net = DQN(input_dim,self.num_actions,self.mode).to(self.device)
        self.q_net.apply(init_weights)
        self.target_net = DQN(input_dim,self.num_actions,self.mode).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=args.lr)


        # resume weights
        if args.resume_path is not None:
          print(f"Loading checkpoint from {args.resume_path}")
          state_dict = torch.load(args.resume_path, map_location=self.device)
          self.q_net.load_state_dict(state_dict)
          self.target_net.load_state_dict(self.q_net.state_dict())


        self.memory_size = args.memory_size
        #self.memory = deque(maxlen=self.memory_size)
        self.memory = PrioritizedReplayBuffer(self.memory_size)
        
        self.batch_size = args.batch_size
        self.gamma = args.discount_factor
        self.epsilon = args.epsilon_start
        self.epsilon_decay = args.epsilon_decay
        self.epsilon_min = args.epsilon_min

        self.env_count = args.start_env_count
        self.train_count = 0
        self.best_reward = 0  # Initilized to 0 for CartPole and to -21 for Pong
        self.max_episode_steps = args.max_episode_steps
        self.replay_start_size = args.replay_start_size
        self.target_update_frequency = args.target_update_frequency
        self.train_per_step = args.train_per_step
        self.save_dir = args.save_dir
        os.makedirs(self.save_dir, exist_ok=True)

        self.save_dir = args.save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        
        # Task 3  Checkpoint
        self.save_steps = [600000, 1000000, 1500000, 2000000, 2500000]
        self.saved_steps = {s for s in self.save_steps if s <= self.env_count}

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.num_actions - 1)
        state_tensor = torch.from_numpy(np.array(state)).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_net(state_tensor)
        return q_values.argmax().item()
    
    
    def get_n_step_transition(self):
      first_state, first_action, _, _, _ = self.n_step_buffer[0]
      last_next_state = self.n_step_buffer[-1][3]
      last_done = self.n_step_buffer[-1][4]

      n_step_reward = 0.0
      for i, (_, _, reward, _, done) in enumerate(self.n_step_buffer):
          n_step_reward += (self.gamma ** i) * reward
          if done:
              break

      return first_state, first_action, n_step_reward, last_next_state, last_done
    
    def evaluate_n_episodes(self, n=5, seed_start=0):
      scores = []

      for i in range(n):
          seed = seed_start + i
          obs, _ = self.test_env.reset(seed=seed)
          self.test_env.action_space.seed(seed)

          if self.mode == "atari":
              state = self.preprocessor.reset(obs)
          else:
              state = obs

          done = False
          total_reward = 0

          while not done:
              state_tensor = torch.from_numpy(np.array(state)).float().unsqueeze(0).to(self.device)

              with torch.no_grad():
                  action = self.q_net(state_tensor).argmax().item()

              next_obs, reward, terminated, truncated, _ = self.test_env.step(action)
              done = terminated or truncated
              total_reward += reward

              if self.mode == "atari":
                  state = self.preprocessor.step(next_obs)
              else:
                  state = next_obs

          scores.append(total_reward)

      return np.mean(scores), np.std(scores)


    
    def run(self, episodes=10000):
        for ep in range(episodes):
            obs, _ = self.env.reset()

            if self.mode == "atari":
                state = self.preprocessor.reset(obs) 
            else:
                state = obs
            done = False
            total_reward = 0
            step_count = 0

            while not done and step_count < self.max_episode_steps:
                action = self.select_action(state)
                next_obs, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated


                # 
                if self.mode =="atari":
                    next_state = self.preprocessor.step(next_obs)
                else:
                    next_state = next_obs

                self.n_step_buffer.append((state, action, reward, next_state, done))
                # self.memory.append((state, action, reward, next_state, done))
                # if full or end of episode => memory
                
                if done:
                    while len(self.n_step_buffer) > 0:
                        transition = self.get_n_step_transition()
                        self.memory.add(transition, error=None)
                        self.n_step_buffer.popleft()

                elif len(self.n_step_buffer) == self.n_step:
                    transition = self.get_n_step_transition()
                    self.memory.add(transition, error=None)
                    self.n_step_buffer.popleft()
                
                
                if done:
                    self.n_step_buffer.clear()


                for _ in range(self.train_per_step):
                    self.train()

                state = next_state
                total_reward += reward
                self.env_count += 1
                step_count += 1


                #  Task 3  Snapshot
                for target_step in self.save_steps:
                    if self.env_count >= target_step and target_step not in self.saved_steps:
                        model_path = os.path.join(self.save_dir, f"LAB5_E84111201_task3_{target_step}.pt")
                        torch.save(self.q_net.state_dict(), model_path)
                        print(f"🌟 Reached {target_step} steps! Saved model to {model_path}")
                        self.saved_steps.add(target_step)
            

                if self.env_count % 1000 == 0:                 
                    print(f"[Collect] Ep: {ep} Step: {step_count} SC: {self.env_count} UC: {self.train_count} Eps: {self.epsilon:.4f}")
                    wandb.log({
                        "Episode": ep,
                        "Step Count": step_count,
                        "Env Step Count": self.env_count,
                        "Update Count": self.train_count,
                        "Epsilon": self.epsilon
                    })
                    ########## YOUR CODE HERE  ##########
                    # Add additional wandb logs for debugging if needed 
                    
                    ########## END OF YOUR CODE ##########   
            print(f"[Eval] Ep: {ep} Total Reward: {total_reward} SC: {self.env_count} UC: {self.train_count} Eps: {self.epsilon:.4f}")
            wandb.log({
                "Episode": ep,
                "Total Reward": total_reward,
                "Env Step Count": self.env_count,
                "Update Count": self.train_count,
                "Epsilon": self.epsilon
            })
            ########## YOUR CODE HERE  ##########
            # Add additional wandb logs for debugging if needed 
            
            ########## END OF YOUR CODE ##########  
            # for task 3
            if ep % 10 == 0:
              latest_path = os.path.join(self.save_dir, "latest_model.pt")
              torch.save(self.q_net.state_dict(), latest_path)
              print(f"Saved latest model to {latest_path}")
          

            if ep % 100 == 0:
                model_path = os.path.join(self.save_dir, f"model_ep{ep}.pt")
                torch.save(self.q_net.state_dict(), model_path)
                print(f"Saved model checkpoint to {model_path}")

            if ep % 20 == 0:
              eval_mean, eval_std = self.evaluate_n_episodes(n=5, seed_start=0)

              if eval_mean > self.best_reward:
                  self.best_reward = eval_mean
                  model_path = os.path.join(self.save_dir, "LAB5_E84111201_task3_best.pt")
                  torch.save(self.q_net.state_dict(), model_path)
                  print(f"Saved new best model to {model_path} with mean reward {eval_mean:.2f}")

              wandb.log({
                  "Env Step Count": self.env_count,
                  "Update Count": self.train_count,
                  "Eval Mean Reward": eval_mean,
                  "Eval Std Reward": eval_std
              })

    def evaluate(self):
        obs, _ = self.test_env.reset()

        if self.mode =="atari":
            state = self.preprocessor.reset(obs)
        else :
            state = obs
        done = False
        total_reward = 0

        while not done:
            state_tensor = torch.from_numpy(np.array(state)).float().unsqueeze(0).to(self.device)
            with torch.no_grad():
                action = self.q_net(state_tensor).argmax().item()
            next_obs, reward, terminated, truncated, _ = self.test_env.step(action)
            done = terminated or truncated
            total_reward += reward
            if self.mode =="atari":
                state = self.preprocessor.step(next_obs)
            else:
                state = next_obs
        return total_reward


    def train(self):

        if len(self.memory) < self.replay_start_size:
            return 
        
        # Decay function for epsilin-greedy exploration
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        self.train_count += 1
       
        ########## YOUR CODE HERE (<5 lines) ##########
        # Sample a mini-batch of (s,a,r,s',done) from the replay buffer

        # batch = random.sample(self.memory, self.batch_size)
        # states,actions, rewards, next_states, dones = zip(*batch)
        samples, indices, weights = self.memory.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*samples)

        weights = torch.tensor(weights, dtype=torch.float32).to(self.device)
            
        ########## END OF YOUR CODE ##########

        # Convert the states, actions, rewards, next_states, and dones into torch tensors
        # NOTE: Enable this part after you finish the mini-batch sampling
        states = torch.from_numpy(np.array(states).astype(np.float32)).to(self.device)
        next_states = torch.from_numpy(np.array(next_states).astype(np.float32)).to(self.device)
        actions = torch.tensor(actions, dtype=torch.int64).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)


        q_values = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # task 3
        use_ddqn = True
        ########## YOUR CODE HERE (~10 lines) ##########
        # Implement the loss function of DQN and the gradient updates 
        with torch.no_grad():
            if use_ddqn:
              # use q network to find next max q for action
              best_actions_next = self.q_net(next_states).argmax(dim = 1, keepdim = True)

              # use target netwrok to evaluate
              max_next_q_values = self.target_net(next_states).gather(1, best_actions_next).squeeze(1)
            else:
            # max(1)[0]: find max value along with action channel 
              max_next_q_values = self.target_net(next_states).max(1)[0]

            target_q_values = rewards +(self.gamma** self.n_step) * max_next_q_values*( 1- dones)
        
        # elementwise_loss = F.mse_loss(q_values, target_q_values, reduction='none')
        
        # 換Huber loss 對outlier TD error比較穩 ( from gpt)
        elementwise_loss = F.smooth_l1_loss(q_values, target_q_values, reduction='none')



        # [PER] 2. 將獨立的 Loss 乘上各自的 IS Weight，然後才取平均
        loss = (weights * elementwise_loss).mean()

        # loss = nn.MSELoss()(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 10.0)
        self.optimizer.step()

        td_errors = torch.abs(target_q_values - q_values).detach().cpu().numpy()
        self.memory.update_priorities(indices, td_errors)
        
        ########## END OF YOUR CODE ##########  

        if self.train_count % self.target_update_frequency == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        # NOTE: Enable this part if "loss" is defined
        if self.train_count % 1000 == 0:
           print(f"[Train #{self.train_count}] Loss: {loss.item():.4f} Q mean: {q_values.mean().item():.3f} std: {q_values.std().item():.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-name", type=str, default="CartPole-v1")
    parser.add_argument("--save-dir", type=str, default="./results")
    parser.add_argument("--wandb-run-name", type=str, default="cartpole-run")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--memory-size", type=int, default=100000)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--discount-factor", type=float, default=0.99)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-decay", type=float, default=0.999999)
    parser.add_argument("--epsilon-min", type=float, default=0.05)
    parser.add_argument("--target-update-frequency", type=int, default=1000)
    parser.add_argument("--replay-start-size", type=int, default=50000)
    parser.add_argument("--max-episode-steps", type=int, default=10000)
    parser.add_argument("--train-per-step", type=int, default=1)
    parser.add_argument("--resume-path", type=str, default=None)
    parser.add_argument("--start-env-count", type=int, default=0)
    args = parser.parse_args()

    # 讓 wandb 的 project 名稱根據環境動態改變，才不會把 CartPole 和 Pong 的數據混在一起
    safe_env_name = args.env_name.replace("/", "-")
    project_name = f"DLP-Lab5-DQN-{safe_env_name}"
    wandb.init(project=project_name, name=args.wandb_run_name, save_code=True)

    agent = DQNAgent(env_name=args.env_name,args=args)
    agent.run()