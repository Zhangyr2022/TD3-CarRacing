import cv2
import gym
import random
from collections import deque
import torch
from torch import nn
import numpy as np
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from matplotlib import pyplot as plt


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim) -> None:
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.Mish(),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.Mish(),
        )
        self.down1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.Mish(),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.Mish()
        )
        self.down2 = nn.MaxPool2d(kernel_size=2)

        self.conv3 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.Mish(),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.Mish()
        )
        self.down3 = nn.MaxPool2d(kernel_size=2)

        self.linear = nn.Sequential(
            nn.Linear(16*(state_dim//8)**2, 256),
            nn.LayerNorm(256),
            nn.Mish(),
            nn.Linear(256, action_dim),
            nn.LayerNorm(action_dim),
            nn.Tanh()
        )

    def forward(self, state: np.ndarray | torch.Tensor | list, sigma=0, brake_rate=1):

        h = self.conv1(state)
        h = self.down1(h)
        h = self.conv2(h)
        h = self.down2(h)
        h = self.conv3(h)
        h = self.down3(h)
        h = torch.flatten(h, start_dim=1)

        h = self.linear(h)

        h_clone = h.clone()+torch.randn_like(h)*sigma
        h_clone[:, 0] = (h_clone[:, 0])
        h_clone[:, 1] = (h_clone[:, 1]+1)*0.5+0.1
        h_clone[:, 2] = (h_clone[:, 2]+1)*0.015
        return h_clone


class Critic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int) -> None:
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.Mish(),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.Mish(),
        )
        self.down1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.Mish()
        )
        self.down2 = nn.MaxPool2d(kernel_size=2)

        self.conv3 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.Mish(),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.Mish()
        )
        self.down3 = nn.MaxPool2d(kernel_size=2)

        self.action_linear = nn.Sequential(
            nn.Linear(action_dim, 256),
            nn.LayerNorm(256),
            nn.Mish()
        )

        self.state_linear = nn.Sequential(
            nn.Linear(16*(state_dim//8)**2, 256),
            nn.LayerNorm(256),
            nn.Mish(),
        )

        self.concat_linear = nn.Sequential(
            nn.Linear(512, 64),
            nn.LayerNorm(64),
            nn.Mish(),
            nn.Linear(64, 64),
            nn.LayerNorm(64),
            nn.Mish(),
            nn.Linear(64, 1)
        )

    def forward(self, state: np.ndarray | torch.Tensor | list, action: np.ndarray | torch.Tensor | list):
        # extract the state features
        state_h = self.conv1(state)
        state_h = self.down1(state_h)
        state_h = self.conv2(state_h)
        state_h = self.down2(state_h)
        state_h = self.conv3(state_h)
        state_h = self.down3(state_h)
        state_h = torch.flatten(state_h, start_dim=1)

        state_h = self.state_linear(state_h)
        # action features
        action_h = self.action_linear(action)

        # concat
        h = self.concat_linear(torch.concat((state_h, action_h), dim=1))

        return h


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, next_state, reward, done):
        experience = (state, action, next_state, reward, done)
        self.buffer.append(experience)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, next_state, reward, done = map(np.stack, zip(*batch))

        return torch.FloatTensor(state), torch.FloatTensor(action), torch.FloatTensor(next_state), torch.FloatTensor(reward), torch.FloatTensor(done)

    def __len__(self):
        return len(self.buffer)


class Agent:
    def __init__(self, state_dim: int, action_dim: int, lr: float, device="cpu") -> None:
        self.brake_rate = 0

        self.device = device

        self.actor = Actor(state_dim, action_dim).to(device)
        self.critic1 = Critic(state_dim, action_dim).to(device)
        self.critic2 = Critic(state_dim, action_dim).to(device)

        self.target_actor = Actor(state_dim, action_dim).to(device)
        self.target_actor.load_state_dict(self.actor.state_dict())

        self.target_critic1 = Critic(state_dim, action_dim).to(device)
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2 = Critic(state_dim, action_dim).to(device)
        self.target_critic2.load_state_dict(self.critic2.state_dict())

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic1_optimizer = torch.optim.Adam(
            self.critic1.parameters(), lr=lr)
        self.critic2_optimizer = torch.optim.Adam(
            self.critic2.parameters(), lr=lr)

        self.critic_mse_loss = nn.MSELoss().to(device)

    def select_action(self, state, sigma=0):
        state = torch.FloatTensor(state).unsqueeze(0)
        action = self.actor.forward(
            state.to(self.device), sigma=sigma, brake_rate=self.brake_rate)
        return action.cpu().detach().numpy()[0]

    def train(self, replay_buffer: ReplayBuffer, batch_size: int, gamma: float, tau: float, update_time=0):
        (state, action, next_state, reward, done) = replay_buffer.sample(batch_size)

        state = state.to(self.device)
        action = action.to(self.device)
        next_state = next_state.to(self.device)
        reward = reward.to(self.device)
        done = done.to(self.device)

        predicted_q1 = self.critic1.forward(state, action)
        predicted_q2 = self.critic2.forward(state, action)

        target_action = self.target_actor.forward(
            next_state, brake_rate=self.brake_rate).detach()+0.005*torch.randn_like(action)
        target_q1 = self.target_critic1.forward(
            next_state, target_action).detach()
        target_q2 = self.target_critic2.forward(
            next_state, target_action).detach()

        TD_target = reward.unsqueeze(
            1)+gamma*torch.min(target_q1, target_q2)*(1-done.unsqueeze(1))
        # TD_error = predicted_q-(TD_target)

        # Update critic networks
        critic1_loss = self.critic_mse_loss.forward(predicted_q1, TD_target)

        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()

        critic2_loss = self.critic_mse_loss.forward(predicted_q2, TD_target)

        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()

        # Update actor networks
        if update_time % 3 == 0:
            predicted_action = self.actor.forward(
                state, brake_rate=self.brake_rate)
            actor_loss = -self.critic1.forward(state, predicted_action).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update target networks
            for param, target_param in zip(self.actor.parameters(), self.target_actor.parameters()):
                target_param.data.copy_(
                    tau * param.data + (1 - tau) * target_param.data)

            for param, target_param in zip(self.critic1.parameters(), self.target_critic1.parameters()):
                target_param.data.copy_(
                    tau * param.data + (1 - tau) * target_param.data)

            for param, target_param in zip(self.critic2.parameters(), self.target_critic2.parameters()):
                target_param.data.copy_(
                    tau * param.data + (1 - tau) * target_param.data)


def car_in_road_reward(image):
    # car_range: ()
    # 转换为HSV颜色空间
    # hsv_image = cv2.cvtColor(image[60:84, 40:60, :], cv2.COLOR_RGB2HSV)
    part_image = image[60:84, 40:60, :]
    # 定义道路、小车和草地的颜色范围（根据实际情况调整）
    road_color_lower = np.array([90, 90, 90], dtype=np.uint8)
    road_color_upper = np.array([120, 120, 120], dtype=np.uint8)
    grass_color_lower = np.array([90, 180, 90], dtype=np.uint8)
    grass_color_upper = np.array([120, 255, 120], dtype=np.uint8)

    # 根据颜色范围创建掩膜
    road_mask = cv2.inRange(part_image, road_color_lower, road_color_upper)
    # car_mask = cv2.inRange(hsv_image, car_color_lower, car_color_upper)
    grass_mask = cv2.inRange(part_image, grass_color_lower, grass_color_upper)

    # 对掩膜进行图像处理，例如腐蚀和膨胀，以去除噪声和平滑边界

    # 在道路区域和草地区域内计算像素数量
    road_pixel_count = cv2.countNonZero(road_mask)
    grass_pixel_count = cv2.countNonZero(grass_mask)

    if road_pixel_count+grass_pixel_count == 0:
        return 0

    reward = (road_pixel_count)/(road_pixel_count+grass_pixel_count) - 0.5

    return reward*6


if __name__ == "__main__":
    device = "cuda"
    env = gym.make('CarRacing-v2', render_mode="human")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    lr = 4.5e-5
    gamma = 0.985
    batch_size = 96
    max_epoch = 2000
    max_step_per_epoch = 10000
    replay_capacity = 2000
    tau_initial = 0.500
    tau_final = 0.001
    tau_decay = 0.0005

    agent = Agent(state_dim, action_dim, lr, device)
    replay_buffer = ReplayBuffer(replay_capacity)
    transformer = transforms.ToTensor()
    writer = SummaryWriter('./data')

    total_steps = 0

    for epoch in range(max_epoch):
        state = transformer(env.reset()[0])
        episode_reward = 0
        continue_negative_reward_time = 0
        continue_out_of_road_time = 0
        tau = max(tau_final, tau_initial - epoch * tau_decay)
        update_time = 0

        for step in range(max_step_per_epoch):
            action = agent.select_action(state, sigma=0.1*(1-epoch/max_epoch))

            next_state_image, reward, done, truncated, info = env.step(action)
            next_state = transformer(next_state_image)

            if reward < 0:
                continue_negative_reward_time += 1
            else:
                continue_negative_reward_time = 0

            episode_reward += reward

            if step > 30:
                in_road_reward = min(
                    0.01, car_in_road_reward(next_state_image))
                if in_road_reward < 0:
                    continue_out_of_road_time += 1
                else:
                    continue_out_of_road_time = 0
                reward += in_road_reward

            if step % 3 == 0:
                replay_buffer.add(state, action, next_state, reward, done)
                if replay_buffer.__len__() > batch_size:
                    agent.train(replay_buffer, batch_size,
                                gamma, tau, update_time)
                    update_time += 1
                    # print(is_car_in_center(next_state_image))
                # Check action value
                with torch.no_grad():
                    action_value1 = agent.critic1.forward(
                        state.unsqueeze(0).to(device), torch.FloatTensor(action).unsqueeze(0).to(device))
                    action_value2 = agent.critic2.forward(
                        state.unsqueeze(0).to(device), torch.FloatTensor(action).unsqueeze(0).to(device))
                    writer.add_scalar(
                        'action value 1', action_value1, total_steps)
                    writer.add_scalar(
                        'action value 2', action_value2, total_steps)

            state = next_state
            total_steps += 1

            if done or truncated or episode_reward < -5 \
                    or continue_negative_reward_time > 60 or continue_out_of_road_time > 10:
                writer.add_scalar(
                    'episode reward', episode_reward, epoch)
                agent.brake_rate = min(1, epoch / max_epoch * 0.5)
                break

        if epoch % 100 == 0 and epoch > 0:
            torch.save(agent.actor, f"./model/agent{epoch}.pt")
