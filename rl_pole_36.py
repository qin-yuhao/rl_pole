import random
import gym
import numpy as np
from tensorflow.keras import models, layers
import time

env = gym.make('CartPole-v0') # 创建倒立摆模型

STATE_DIM, ACTION_DIM = 4, 2 # State 维度 4, Action 维度 2

model = models.Sequential([
    layers.Dense(64, input_dim=STATE_DIM, activation='relu'),
    layers.Dense(20, activation='relu'),
    layers.Dense(ACTION_DIM, activation='linear')
]) # 简单的MLP

model.summary()
def generate_data_one_episode():
    '''生成单次游戏的训练数据'''
    x, y, score = [], [], 0
    obs = env.reset()
    while True:
        action = env.action_space.sample()
        x.append(obs)
        y.append([1,0] if action == 0 else [0,1])
        obs, reward, done, info = env.step(action)
        score += reward
        if done:
            break
    return x, y, score

def generate_training_data(expected_score=100):
    '''生成N次游戏的训练数据，并进行筛选，选择 > 100 的数据作为训练集'''
    data_X, data_Y, scores = [], [], []
    for i in range(10000):
        x, y, score = generate_data_one_episode()
        if score > expected_score:
            data_X += x
            data_Y += y
            scores.append(score)
    print('dataset size: {}, max score: {}'.format(len(data_X), max(scores)))
    return np.array(data_X), np.array(data_Y)
data_X, data_Y = generate_training_data()
model.compile(loss='mse', optimizer='adam')
model.fit(data_X, data_Y, epochs=5)
model.save('CartPole-v0-nn.h5')
saved_model = models.load_model('CartPole-v0-nn.h5')  # 加载模型
env = gym.make("CartPole-v0")  # 加载游戏环境

for i in range(5):
    obs = env.reset()
    score = 0
    while True:
        time.sleep(0.01)
        #env.render()   # 显示画面
        action = np.argmax(saved_model.predict(np.array([obs]))[0])  # 预测动作
        obs, reward, done, info = env.step(action)  # 执行这个动作
        score += reward     # 每回合的得分
        if done:       # 游戏结束
            print('using nn, score: ', score)  # 打印分数
            break
env.close()