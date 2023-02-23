#완벽한 baseline


# import packages
import numpy as np
import math
import random
import time
from datetime import datetime, timedelta
from IPython.core.debugger import set_trace

import scipy.stats as stats
from scipy.stats import norm

import itertools
from itertools import count
from itertools import product
from collections import namedtuple

import pandas as pd
import matplotlib
import matplotlib.pylab as plt

from bokeh.io import show, output_notebook
from bokeh.palettes import PuBu4
from bokeh.plotting import figure
from bokeh.models import Label
from IPython.display import clear_output

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import pickle
from collections import Counter

# ### 자주 수정하는 parameter 앞으로

# In[360]:
"""BASELINE"""

# 자주 수정하는 parameter 앞으로
result_numpy_path = '1205_successive'
now = datetime.now().timestamp()
#PATH = result_numpy_path + '_model_' + str(now)  # model path
PATH = result_numpy_path + '_model_'  # model path
# op = 0.3 # stock capacity 기준으로 몇% 이하이면 재주문을 할 것인지(연속형 주문에 쓰인다)

# 공급이 수요보다 많도록!
q = 250  # 보관 가능한 총 item 수(창고 크기)
T = 15  # 하루 영업시간(재고보충시간 포함)
n = 20  # 시간당 방문 고객 수
# n = random.randrange(15, 25) # 한 시간에 입장하는 고객 수. 15부터 25 사이의 난수 생성


min_p = 30  # minimum price
max_p = 40  # maximum price #이거 변경할 때는 새로 train 해야함.

memory_capacity = 5000
BATCH_SIZE = 32
GAMMA = 0.99
TARGET_UPDATE = 10  # 10일마다 타겟 업데이트.
REPLAY_START_SIZE = 5000 #일단 500.

test_trial = 5

num_episodes = 10000 # 10000일. 이제 제대로 돌려야지. # 일단 1000
#num_episodes = 10000
num_episodes_test = 300
num_episodes_test_2 = 365 * 40

# num_episodes, num_episodes_test, num_episodes_test_2  = 3, 3, 3

pin = time.strftime('%m/%d_%H:%M', time.localtime(time.time()))
print(pin)



# In[361]:


# 수정 빈도 낮은 parameter
eps_start = 0.9
eps_end = 0.05
eps_decay = 200

lr = 0.0001  # policy net learning rate

# cost per unit
# c1 = 0.03 # 재고유지비용 per a product per a day
c1 = 0.6  # 재고 유지 비용
c2 = 0.5  # 구매가 일어나지 않아서 발생하는 손실
# c3 = 0.035 # 폐기비용
c3 = 1  # 구매비용;발주비용; 고정비! (원래는 oc)
c4 = 0.2  # 메뉴비용
c5 = 8  # 구매원가


# 몇 시간 간격으로 time index를 나누는지
t_interval = 1




# ### plot_return_trace visualization

# In[362]:


# Visualization functions

output_notebook()


def plot_return_trace(returns, smoothing_window=10, range_std=2):
    plt.figure(figsize=(16, 5))
    plt.xlabel("Episode")
    plt.ylabel("Return ($)")
    returns_df = pd.Series(returns)
    ma = returns_df.rolling(window=smoothing_window).mean()
    mstd = returns_df.rolling(window=smoothing_window).std()
    plt.plot(ma, c='blue', alpha=1.00, linewidth=1)
    plt.fill_between(mstd.index, ma - range_std * mstd, ma + range_std * mstd, color='blue', alpha=0.2)


def plot_loss_trace(loss_trace):
    plt.figure(figsize=(16, 5))
    plt.xlabel("Episode")
    plt.ylabel("Loss")
    plt.plot(loss_trace, c='blue', alpha=1.00, linewidth=1)


def plot_cumulative_trace(cumulative_trace, cumulative_result_list):
    plt.figure(figsize=(16, 5))
    plt.xlabel("Episode")
    plt.ylabel("Cumulative reward")
    plt.ylim([min(list(map(lambda x: int(min(x)), cumulative_result_list))),
              max(list(map(lambda x: int(max(x)), cumulative_result_list)))])
    plt.plot(cumulative_trace, c='blue', alpha=1.00, linewidth=1)


# ### item state 정의

# In[363]:


# item state 정의


# f_sigma = 0.05 # sigma for freshness threshold
p_sigma = 0.5  # sigma for price threshold

# sales index
si = 1

# initial state
# [0]'stock level', [1]'price', [2]'last time sales', [3]'time index', [4]'sales index'

first_si = [i for i in range(si + 1)]
first_si = first_si.pop(random.randrange(len(first_si)))

state0 = [q, max_p, 0, 0]
state = state0
# state_shape = state0.shape


# In[365]:


# normalization function
# [0]'group' = 3, [1]'freshness' = 1, [2]'stock level' = sl , [3]'price' = 12, [4]'time index' = 6, [5]'customer' = 4
# [0]'stock level', [1]'price', [2]'last time sales', [3]'time index', [4]'sales index'

norm_list = [q, max_p - min_p, n, t_interval]


def to_norm(state):
    state_norm = []
    state_norm.append(state[0] / norm_list[0])
    state_norm.append((state[1] - min_p) / norm_list[1])
    state_norm.append(state[2] / norm_list[2])
    state_norm.append(state[3] / norm_list[3])
    return np.array(state_norm).T



def norm_reward(reward):
    if reward >= -(max_p + 10):
        n_r = reward / max_p
    else:
        n_r = -10
    return n_r


# In[367]:


# tensor 변환 함수
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def to_tensor(x):
    return torch.from_numpy(np.array(x).astype(np.float32)).to(device)


def to_tensor_long(x):
    return torch.tensor([[x]], device=device, dtype=torch.long)



def state_to_nn(state):  # nn에 전달해주는 값은  [2]'stock level',[5]'customer' // 'last time sales', '
    norm_state = to_norm(state)
    x = to_tensor(norm_state[:])  # 전체를 일단 보내보자. -> sales index는 제외하고 진행.
    return x



# ### DQN class


Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


# In[370]:


class PolicyNetworkDQN(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=10):
        super(PolicyNetworkDQN, self).__init__()
        # nn layers
        linear1 = torch.nn.Linear(state_size, hidden_size, bias=True)
        linear2 = torch.nn.Linear(hidden_size, hidden_size, bias=True)
        linear3 = torch.nn.Linear(hidden_size, action_size, bias=True)
        relu = torch.nn.ReLU()
        # kaiming initialization
        torch.nn.init.kaiming_normal_(linear1.weight)
        torch.nn.init.kaiming_normal_(linear2.weight)
        torch.nn.init.kaiming_normal_(linear3.weight)

        self.model = torch.nn.Sequential(linear1, relu, linear2, relu, linear3)

    def forward(self, x):
        q_values = self.model(x)
        return q_values

    # In[371]:


class AnnealedEpsGreedyPolicy(object):
    def __init__(self, eps_start=eps_start, eps_end=eps_end, eps_decay=eps_decay):
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.steps_done = 0

    def select_action(self, q_values):
        sample = random.random()
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * math.exp(-1. * self.steps_done / self.eps_decay)
        #eps_threshold = ((self.eps_end - self.eps_start)/(num_episodes*T))* self.steps_done +self.eps_start
        self.steps_done += 1
        if sample > eps_threshold:
            return torch.argmax(q_values).item()  # torch.argmax 그대로 수정
        else:
            return random.randrange(len(q_values))


def update_model(memory, policy_net, target_net):
    if len(memory) < REPLAY_START_SIZE:
        return
    else:
        transitions = memory.sample(BATCH_SIZE)
        batch = Transition(*zip(*transitions))

        next_states_batch = torch.stack(batch.next_state)
        state_batch = torch.stack(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.stack(batch.reward)

        state_action_values = policy_net(state_batch).gather(1, action_batch)  # policy net이 계산한 value

        next_state_values = target_net(next_states_batch).max(1)[0].detach()  # target net이 계산한 value; goal
        # Compute the expected Q values
        expected_state_action_values = reward_batch[:, 0] + (GAMMA * next_state_values)  # 사이즈 안 맞으면 확인해보기.

        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        optimizer.zero_grad()
        loss.backward()
        for param in policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        optimizer.step()
        return loss.item()


# In[373]:



def env_initial_state(): # 초반 state로 반환.
    return [q,max_p,0,0]


# freshness, price threshold 만들기

def clamp(num, min_value, max_value):
    return max(min(num, max_value), min_value)


# sensitive customer에게 upper bound, insensitive customer에게 lower bound 설정
def p_t_low():
    return clamp(np.random.normal(min_p+(max_p-min_p)*0.25, p_sigma), min_p, (min_p+max_p)/2) # sigma = 0.7

def p_t_high():
    return clamp(np.random.normal(max_p-(max_p-min_p)*0.25, p_sigma), (min_p+max_p)/2, max_p)


# 1단위로 discrete한 price
action_list = [i for i in range(min_p, max_p + 1)]  # 초반부터 할인을 해야 한다.



def env_step(t, state, action):
    customer = [random.randrange(0, 0.2 * n), random.randrange(0, n),
                random.randrange(0.8 * n, n)]  # 아예 안 팔리는 상황, 몇 개라도 팔리는 상황, n개가 다 팔리는 상황
    sales, C1, C2, C3, C4  = 0, 0, 0, 0, 0  # C1:holding, C2 : lost sales, C3 : 뭔 비용이었음.  C4:ordering cost
    goods_sold_rcd = [None for _ in range(3)] # 3개짜리 홀이 있는 리스트 제작.

    #state[1] = action_list[action]
    #next_state = np.repeat(0,len(state))
    next_state = state
    # 구매 발생
    if state[1] >= p_t_high(): # 가격 바운더
        purchase_group = customer[0]
        next_state[0] = state[0] - purchase_group if state[0] >= purchase_group else 0 # 재고 빼
        if state[2] <= int(q/T):
            next_state[1] = state[1] -1 if state[1] > min_p else min_p
        else:
            next_state[1] = state[1]
        next_state[2] = purchase_group # 이전 시간대에 팔린 만큼만 표현해내야지.
        next_state[3] = t
        # sales 계산
        sales = purchase_group * state[1]
        C2 = c2 * (n-purchase_group)
        C1 = c1 * (state[0]-purchase_group)/T
        C4 = c4
#        state[4] = 0

    elif p_t_low()<= state[1] <= p_t_high(): # 가격 바운더리.
        purchase_group = customer[1]
        next_state[0] = state[0] - purchase_group if state[0] >= purchase_group else 0 # 재고 빼
        if state[2] <= int(q/T):
            next_state[1] = state[1] -1 if state[1] > min_p else min_p
        else:
            next_state[1] = state[1]
        next_state[2] = purchase_group # 이전 시간대에 팔린 만큼만 표현해내야지.
        next_state[3] = t
        # sales 계산
        sales = purchase_group * state[1]
        C2 = c2 * (n-purchase_group)
        #C1 = c1 * len(m) / T  # holding cost C1 발생; 균일하게
        C1 = c1 * (state[0]-purchase_group)/T
        C4 = c4
#        state[4] = 1


    elif state[1] <= p_t_low(): # 가격 바운더리.
        purchase_group = customer[2]
        next_state[0] = state[0] - purchase_group if state[0] >= purchase_group else 0 # 재고 빼
        if state[2] <= int(q/T):
            next_state[1] = state[1] -1 if state[1] > min_p else min_p
        else:
            next_state[1] = state[1]
        next_state[2] = purchase_group # 이전 시간대에 팔린 만큼만 표현해내야지.
        next_state[3] = t
        # state[3] = t
        # sales 계산
        sales = purchase_group * state[1]
        C2 = c2 * (n-purchase_group)
        C1 = c1 * (state[0]-purchase_group)/T # 팔고 남은 것을 시간으로 나누어서 매기
        C4 = c4
#        state[4] = 2

    # 재고가 없을 때
    elif state[0] == 0:
        purchase_group = 0
        next_state[0] = q
        next_state[1] = state[1]  # 가격은 그대로 가져감.
        next_state[2] = purchase_group  # 구매자 없음.
        sales = purchase_group  # 매출도 없음.
        C2 = n * c2  # 손실만 발생
        C3 = c3 * q  # 주문비용만 발생
        next_state[3] = t

    # 재고보충 발생
    elif t == T-1:
        if state[1] >= p_t_high():  # 가격 바운더
            purchase_group = customer[0]
            next_state[0] = state[0] - purchase_group if state[0] >= purchase_group else 0  # 재고 빼
            if state[2] <= int(q / T):
                next_state[1] = state[1] - 1 if state[1] > min_p else min_p
            else:
                next_state[1] = state[1]
            next_state[2] = purchase_group  # 이전 시간대에 팔린 만큼만 표현해내야지.
            next_state[3] = t
            # sales 계산
            sales = purchase_group * state[1]
            C2 = c2 * (n - purchase_group)
            C3 = c3 * (q - state[0])  # 주문비용만 발생. 주문비용 * (최대재고-남은재고)
            C1 = c1 * state[0]/(24-T)


        elif p_t_low() <= state[1] <= p_t_high():  # 가격 바운더리.
            purchase_group = customer[1]
            next_state[0] = state[0] - purchase_group if state[0] >= purchase_group else 0  # 재고 빼
            if state[2] <= int(q / T):
                next_state[1] = state[1] - 1 if state[1] > min_p else min_p
            else:
                next_state[1] = state[1]
            next_state[2] = purchase_group  # 이전 시간대에 팔린 만큼만 표현해내야지.
            next_state[3] = t
            # sales 계산
            sales = purchase_group * state[1]
            C2 = c2 * (n - purchase_group)
            C3 = c3 * (q - state[0])  # 주문비용만 발생. 주문비용 * (최대재고-남은재고)
            C1 = c1 * state[0]/(24-T)


        elif state[1] <= p_t_low():  # 가격 바운더리.
            purchase_group = customer[2]
            next_state[0] = state[0] - purchase_group if state[0] >= purchase_group else 0  # 재고 빼
            if state[2] <= int(q / T):
                next_state[1] = state[1] - 1 if state[1] > min_p else min_p
            else:
                next_state[1] = state[1]
            next_state[2] = purchase_group  # 이전 시간대에 팔린 만큼만 표현해내야지.
            next_state[3] = t
            # sales 계산
            sales = purchase_group * state[1]
            C2 = c2 * (n - purchase_group)
            C1 = c1 * state[0]/(24-T)
            C3 = c3 * (q - state[0])  # 주문비용만 발생. 주문비용 * (최대재고-남은재고)
            #C4 = c4
        # 이 다음에는 원래대로 돌아간다.


    # reward calculate
    reward = sales - (C1 + C2 + C3 + C4)
    reward_decompose = [sales, C1, C2, C3, C4]

    return next_state, reward, reward_decompose


# In[376]:


# policy_net & target_net 생성
policy_net = PolicyNetworkDQN(len(state_to_nn(state0)), len(action_list)).to(device)
target_net = PolicyNetworkDQN(len(state_to_nn(state0)), len(action_list)).to(device)

optimizer = optim.AdamW(policy_net.parameters(), lr=lr)
policy = AnnealedEpsGreedyPolicy()
memory = ReplayMemory(memory_capacity)

target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

# ### train

# In[377]:


print("training starts")

# train

return_trace, loss_trace = [[] for i in range(2)]


for i_episode in range(num_episodes):  # self 넣고 싶으면 class로 만들어라. 일단 10
    reward_trace, loss_list, si_queue = [[] for i in range(3)]
    state = env_initial_state()  # 가격이 원래대로 돌아옴.
    for t in range(T):  # 한 에피소드 내에서는 T만큼의 finite time horizon # hours

        # lts에 따라서 state가 취하는 방향이 달라진다. 많이 팔았다. --> 0, 많이 못 팔았다 --> 1
        # 구매량이 얼만큼 되느냐에 따라서 각각 업데이트
        with torch.no_grad():
            q_values = policy_net(state_to_nn(state))  # 현재 state에서의 모든 action에 대한 q value를 구함
        action = policy.select_action(q_values.detach())  # policy에 따라 q value를 보고 action을 선택
        print("sojung") if action == 20 else None
        next_state, reward, reward_decompose = env_step(t, state, action)
        # Store the transition in memory
        memory.push(to_tensor(state_to_nn(state)),
                    #to_tensor_long(action / (len(action_list))),
                    to_tensor_long(action),
                    to_tensor(state_to_nn(next_state)), # if t != T - 1 else None,
                    to_tensor([norm_reward(reward)]))  # state, action, reward 모두 정규화해서 memory에 저장
        state = next_state
        # Perform one step of the optimization (on the target network)
        loss_value = update_model(memory, policy_net, target_net)  # target_net을 따라 policy_net update
        loss_list.append(loss_value)
        reward_trace.append(reward)

    return_trace.append(sum(reward_trace))
    loss_trace.extend(loss_list)

    # Update the target network, copying all weights and biases in DQN
    if i_episode % TARGET_UPDATE == 0:  # 50 에피소드마다 한번씩 update
        target_net.load_state_dict(policy_net.state_dict())

        clear_output(wait=True)
        print(f"Episode {i_episode} of {num_episodes} ({i_episode / num_episodes * 100:.2f}%)")

plot_return_trace(return_trace)
fig = plt.figure(figsize=(16, 5))
plot_loss_trace(loss_trace)
fig = plt.figure(figsize=(16, 5))


# train 완료된 모델 저장하기
torch.save(policy_net.state_dict(), PATH)
print(PATH)

# In[ ]:
print("training finished")

print("test starts")

# In[ ]:


# train model 불러오기
test_net = PolicyNetworkDQN(len(state_to_nn(state0)), len(action_list)).to(device)
test_net.load_state_dict(torch.load(PATH))
print(PATH)
test_net.eval() # 매개변수로 받은 expression (=식)을 문자열로 받아서, 실행하는 함수 입니다. 일단 있게 진행을 해보자.




# test는 20번 돌려보고 평균값을 사용함

total_return_test = []
for i in range(test_trial):
    return_trace_test, full_reward_trace_test, a_trace_test, reward_dcp_trace_test = [[] for i in range(4)]
    stock_l, price_l, customer_l, time_l = [[] for i in range(4)]
    #state = state0  # initial state
    state_l = []
    # customer queue
    for i_episode in range(num_episodes_test):
        reward_trace_test, si_test_queue = [[] for i in range(2)]
        state = env_initial_state()
        for t in range(T):  # 한 에피소드 내에서는 T만큼의 finite time horizon # hours
            with torch.no_grad():
                q_values = test_net(state_to_nn(state))
            action = torch.argmax(q_values).item()
            stock_l.append(state[0])
            price_l.append(state[1])
            customer_l.append(state[2])
            time_l.append(state[3])
            a_trace_test.append(action)
            next_state, reward, reward_decompose = env_step(t, state, action)
            reward_dcp_trace_test.append(reward_decompose)
            reward_trace_test.append(reward)
            state = next_state  # Move to the next state


        return_trace_test.append(np.sum(reward_trace_test))  # 에피소드별 return; 일 매출
        full_reward_trace_test.extend(reward_trace_test)  # 모든 고객이 입장할 때마다 발생하는 reward 저장; event 매출

        if i_episode % TARGET_UPDATE == 0:
            clear_output(wait=True)
            print(f"Episode {i_episode} of {num_episodes_test} ({i_episode / num_episodes_test * 100:.2f}%)")
            print(f"{i + 1}/20")

        plot_return_trace(return_trace_test)
        fig = plt.figure(figsize=(16,5))

    total_return_test.append(np.sum(np.array(return_trace_test)))  # 매 test의 return의 합; 30000일 매출
    total_test = pd.DataFrame({"stock": stock_l})
    total_test_1 = pd.DataFrame({"price": price_l})
    total_test_2 = pd.DataFrame({"time": time_l})
    total_test_3 = pd.DataFrame({"time": customer_l})

    #q_action = pd.concat([pd.DataFrame(total_test)], axis=1)
    total_test.to_csv(PATH + 'test_stock.csv')
    total_test_1.to_csv(PATH + 'test_price.csv')
    total_test_2.to_csv(PATH + 'test_time.csv')
    total_test_2.to_csv(PATH + 'customer.csv')
print(np.mean(np.array(total_return_test)))
print("test finished")


# 비교모델 1. 랜덤random
print("comp1 random starts")

def env_step_random(t, state, action):
    customer = [random.randrange(0, 0.2 * n), random.randrange(0, n),
                random.randrange(0.8 * n, n)]  # 아예 안 팔리는 상황, 몇 개라도 팔리는 상황, n개가 다 팔리는 상황
    sales, C1, C2, C3, C4  = 0, 0, 0, 0, 0  # C1:holding, C2 : lost sales, C3 : 뭔 비용이었음.  C4:ordering cost
    goods_sold_rcd = [None for _ in range(3)] # 3개짜리 홀이 있는 리스트 제작.
    action = random.randrange(len(action_list))
    #state[1] = action_list[action]
    #next_state = np.repeat(0,len(state))
    next_state = state
    #state[1] = action
    # 구매 발생
    if state[1] >= p_t_high(): # 가격 바운더
        purchase_group = customer[0] if state[0] > 0 else 0
        next_state[0] = state[0] - purchase_group if state[0] >= purchase_group else 0 # 재고 빼
        #next_state[1] = action_list[action] if t != 0 else action_list[10]
        if next_state[1] > state[1]:  # 다음 시점 가격이 높아지면 안되니까.
            # print (“haha”)
            next_state[1] = state[1] if t != 0 else action_list[10]
        else:
            next_state[1] = action_list[action]
        next_state[2] = purchase_group # 이전 시간대에 팔린 만큼만 표현해내야지.
        next_state[3] = t
        # sales 계산
        sales = purchase_group * state[1]
        C2 = c2 * (n-purchase_group)
        #C1 = c1 * len(m) / T  # holding cost C1 발생; 균일하게
        C1 = c1 * (state[0]-purchase_group)/T
        C4 = c4
#        state[4] = 0

    elif p_t_low()<= state[1] <= p_t_high(): # 가격 바운더리.
        purchase_group = customer[1] if state[0] > 0 else 0
        next_state[0] = state[0] - purchase_group if state[0] >= purchase_group else 0 # 재고 빼
        #next_state[1] = action_list[action] if t != 0 else action_list[10]
        if next_state[1] > state[1]:  # 다음 시점 가격이 높아지면 안되니까.
            # print (“haha”)
            next_state[1] = state[1] if t != 0 else action_list[10]
        else:
            next_state[1] = action_list[action]
        next_state[2] = purchase_group # 이전 시간대에 팔린 만큼만 표현해내야지.
        next_state[3] = t
        # sales 계산
        sales = purchase_group * state[1]
        C2 = c2 * (n-purchase_group)
        #C1 = c1 * len(m) / T  # holding cost C1 발생; 균일하게
        C1 = c1 * (state[0]-purchase_group)/T
        C4 = c4
#        state[4] = 1


    elif state[1] <= p_t_low(): # 가격 바운더리.
        purchase_group = customer[2] if state[0] > 0 else 0
        next_state[0] = state[0] - purchase_group if state[0] >= purchase_group else 0 # 재고 빼
        #next_state[1] = action_list[action] if t != 0 else action_list[10]
        if next_state[1] > state[1]:  # 다음 시점 가격이 높아지면 안되니까.
            # print (“haha”)
            next_state[1] = state[1] if t != 0 else action_list[10]
        else:
            next_state[1] = action_list[action]
        next_state[2] = purchase_group # 이전 시간대에 팔린 만큼만 표현해내야지.
        next_state[3] = t
        # state[3] = t
        # sales 계산
        sales = purchase_group * state[1]
        C2 = c2 * (n-purchase_group)
        C1 = c1 * (state[0]-purchase_group)/T
        C4 = c4
#        state[4] = 2

    # 재고가 없을 때
    elif state[0] == 0:
        purchase_group = 0
        next_state[0] = q
        next_state[1] = state[1]  # 가격은 그대로 가져감.
        next_state[2] = purchase_group  # 구매자 없음.
        sales = purchase_group  # 매출도 없음.
        C2 = n * c2  # 손실만 발생
        C3 = c3 * q  # 주문비용만 발생
        next_state[3] = t


    # 재고보충 발생
    elif t == T-1:
        if state[1] >= p_t_high():  # 가격 바운더
            purchase_group = customer[0]
            next_state[0] = state[0] - purchase_group if state[0] >= purchase_group else 0  # 재고 빼
            if next_state[1] > state[1]:  # 다음 시점 가격이 높아지면 안되니까.
                # print (“haha”)
                next_state[1] = state[1] if t != 0 else action_list[10]
            else:
                next_state[1] = action_list[action]
            next_state[2] = purchase_group  # 이전 시간대에 팔린 만큼만 표현해내야지.
            next_state[3] = t
            # sales 계산
            sales = purchase_group * state[1]
            C2 = c2 * (n - purchase_group)
            C3 = c3 * (q - state[0])  # 주문비용만 발생. 주문비용 * (최대재고-남은재고)
            C1 = c1 * state[0] / (24 - T)
            # C4 = c4
        #        state[4] = 0

        elif p_t_low() <= state[1] <= p_t_high():  # 가격 바운더리.
            purchase_group = customer[1]
            next_state[0] = state[0] - purchase_group if state[0] >= purchase_group else 0  # 재고 빼
            if next_state[1] > state[1]:  # 다음 시점 가격이 높아지면 안되니까.
                # print (“haha”)
                next_state[1] = state[1] if t != 0 else action_list[10]
            else:
                next_state[1] = action_list[action] # 이전 시간대에 팔린 만큼만 표현해내야지.
            next_state[3] = t
            # sales 계산
            sales = purchase_group * state[1]
            C2 = c2 * (n - purchase_group)
            C3 = c3 * (q - state[0])  # 주문비용만 발생. 주문비용 * (최대재고-남은재고)
            # C1 = c1 * len(m) / T  # holding cost C1 발생; 균일하게
            C1 = c1 * state[0] / (24 - T)
            # C4 = c4
        #        state[4] = 1

        elif state[1] <= p_t_low():  # 가격 바운더리.
            purchase_group = customer[2]
            next_state[0] = state[0] - purchase_group if state[0] >= purchase_group else 0  # 재고 빼
            if next_state[1] > state[1]:  # 다음 시점 가격이 높아지면 안되니까.
                # print (“haha”)
                next_state[1] = state[1] if t != 0 else action_list[10]
            else:
                next_state[1] = action_list[action]
            next_state[2] = purchase_group  # 이전 시간대에 팔린 만큼만 표현해내야지.
            next_state[3] = t
            # state[3] = t
            # sales 계산
            sales = purchase_group * state[1]
            C2 = c2 * (n - purchase_group)
            C1 = c1 * state[0] / (24 - T)
            C3 = c3 * (q - state[0])  # 주문비용만 발생. 주문비용 * (최대재고-남은재고)
            # C4 = c4
        # 이 다음에는 원래대로 돌아간다.


    # reward calculate
    reward = sales - (C1 + C2 + C3 + C4)
    reward_decompose = [sales, C1, C2, C3, C4]

    return next_state, reward, reward_decompose


total_return_comp1 = []
for i in range(test_trial):
    return_trace_comp1, full_reward_trace_comp1, reward_dcp_trace_comp1, state_trace_comp1 = [[] for i in range(4)]
    print(f"test {i} of {test_trial} ({i / test_trial}%)")

#    si_queue = si_for_test[i].copy()
    for i_episode in range(num_episodes_test):  # self 넣고 싶으면 class로 만들어라.
        reward_trace_comp1 = []
        # Select and perform an action
        state = env_initial_state()
        for t in range(T):  # 한 에피소드 내에서는 T만큼의 finite time horizon # hours
            # 구매량이 얼만큼 되느냐에 따라서 각각 업데이트
            with torch.no_grad():
                q_values = test_net(state_to_nn(state))
            action = torch.argmax(q_values).item()
            next_state, reward, reward_decompose = env_step_random(t, state, action)
            state = next_state
            reward_trace_comp1.append(reward)
            reward_dcp_trace_comp1.append(reward_decompose)

        return_trace_comp1.append(sum(reward_trace_comp1))
        full_reward_trace_comp1.extend(reward_trace_comp1)


        # Update the target network, copying all weights and biases in DQN
        if i_episode % TARGET_UPDATE == 0:  # 50 에피소드마다 한번씩 update
            clear_output(wait = True)
            print(f"Episode {i_episode} of {num_episodes_test} ({i_episode/num_episodes_test*100:.2f}%)")
            print(f"{i+1}/20")


    total_return_comp1.append(np.sum(np.array(return_trace_comp1)))  # 매 test의 return의 합; 30000일 매출

print(np.mean(np.array(total_return_comp1)))
print("comp1 random finished")


# 비교모델 2. # clearance sales
print("comp2 clearance starts")
def env_step_clearance(t, state, action):
    customer = [random.randrange(0, 0.2 * n), random.randrange(0, n),
                random.randrange(0.8 * n, n)]  # 아예 안 팔리는 상황, 몇 개라도 팔리는 상황, n개가 다 팔리는 상황
    sales, C1, C2, C3, C4  = 0, 0, 0, 0, 0  # C1:holding, C2 : lost sales, C3 : 뭔 비용이었음.  C4:ordering cost
    goods_sold_rcd = [None for _ in range(3)] # 3개짜리 홀이 있는 리스트 제작.
    next_state = state

    # 구매 발생

    if 0 <= t < T - 7: # 가격 바운더
        purchase_group = customer[0] if state[0] > 0 else 0
        next_state[0] = state[0] - purchase_group if state[0] >= purchase_group else 0 # 재고 빼
        next_state[1] = action_list[10]
        next_state[2] = purchase_group # 이전 시간대에 팔린 만큼만 표현해내야지.
        next_state[3] = t
        # sales 계산
        sales = purchase_group * state[1]
        C2 = c2 * (n-purchase_group)
        #C1 = c1 * len(m) / T  # holding cost C1 발생; 균일하게
        C1 = c1 * (state[0]-purchase_group)/T
#        state[4] = 0


    elif T - 7 <= t < T - 1: # 가격 바운더리.
        purchase_group = customer[2] if state[0] > 0 else 0
        next_state[0] = state[0] - purchase_group if state[0] >= purchase_group else 0 # 재고 빼
        next_state[1] = action_list[0]
        #if next_state[1] >= state[1]: # 다음 시점껏이 더 커져버리면 출력해라
        #    next_state[1] = state[1] # 1달러 할인
        next_state[2] = purchase_group # 이전 시간대에 팔린 만큼만 표현해내야지.
        next_state[3] = t
        # state[3] = t
        # sales 계산
        sales = purchase_group * state[1]
        C2 = c2 * (n-purchase_group)
        C1 = c1 * (state[0]-purchase_group)/T
#        state[4] = 2

    # 재고가 없을 때
    elif state[0] == 0:
        purchase_group = 0
        next_state[0] = q
        next_state[1] = state[1]  # 가격은 그대로 가져감.
        next_state[2] = purchase_group  # 구매자 없음.
        sales = purchase_group  # 매출도 없음.
        C2 = n * c2  # 손실만 발생
        C3 = c3 * q  # 주문비용만 발생
        next_state[3] = t

    # 재고보충 발생
    elif t == T-1:
        if state[1] >= p_t_high():  # 가격 바운더
            purchase_group = customer[2]
            next_state[0] = state[0] - purchase_group if state[0] >= purchase_group else 0  # 재고 빼
            next_state[1] = action_list[10]
            next_state[2] = purchase_group  # 이전 시간대에 팔린 만큼만 표현해내야지.
            next_state[3] = t
            # sales 계산
            sales = purchase_group * state[1]
            C2 = c2 * (n - purchase_group)
            C3 = c3 * (q - state[0])  # 주문비용만 발생. 주문비용 * (최대재고-남은재고)
            C1 = c1 * state[0] / (24 - T)
            # C4 = c4
        #
        # 이 다음에는 원래대로 돌아간다.

    # reward calculate
    reward = sales - (C1 + C2 + C3 + C4)
    reward_decompose = [sales, C1, C2, C3, C4]

    return next_state, reward, reward_decompose

total_return_comp2 = []
for i in range(test_trial):
    return_trace_comp2, full_reward_trace_comp2, reward_dcp_trace_comp2, state_trace_comp2 = [[] for i in range(4)]
    print(f"test {i} of {test_trial} ({i / test_trial}%)")
    for i_episode in range(num_episodes_test):  # self 넣고 싶으면 class로 만들어라.
        reward_trace_comp2 = []
        # Select and perform an action
        state = env_initial_state()
        for t in range(T):  # 한 에피소드 내에서는 T만큼의 finite time horizon # hours

            # 구매량이 얼만큼 되느냐에 따라서 각각 업데이트
                with torch.no_grad():
                    q_values = test_net(state_to_nn(state))
                action = torch.argmax(q_values).item()
                next_state, reward, reward_decompose = env_step_clearance(t, state, action)
                state = next_state
                reward_trace_comp2.append(reward)
                reward_dcp_trace_comp2.append(reward_decompose)

        return_trace_comp2.append(sum(reward_trace_comp2))
        full_reward_trace_comp2.extend(reward_trace_comp2)


        # Update the target network, copying all weights and biases in DQN
        if i_episode % TARGET_UPDATE == 0:
            clear_output(wait = True)
            print(f"Episode {i_episode} of {num_episodes_test} ({i_episode/num_episodes_test*100:.2f}%)")
            print(f"{i+1}/20")



    total_return_comp2.append(np.sum(np.array(return_trace_comp2)))  # 매 test의 return의 합; 30000일 매출

print(np.mean(np.array(total_return_comp2)))
print("comp2 clearance finished")

# ### result

# In[ ]:


# 누적 합계로 result 확인
def cumulative_sum(trace_list):
    cum_sum_list = [0]
    for i in range(len(trace_list)):
        cum_sum_list.append(cum_sum_list[i] + trace_list[i])
    del cum_sum_list[0]
    return cum_sum_list


# In[ ]:


result_list = [return_trace_test]
result_1 = [return_trace_comp1]
result_2 = [return_trace_comp2]
#result_list = [return_trace_test, return_trace_comp2]
cumulative_result_list = []
for i in result_list:
    cumulative_result_list.append(cumulative_sum(i))
for i in result_1:
    cumulative_result_list.append(cumulative_sum(i))
for i in result_2:
    cumulative_result_list.append(cumulative_sum(i))

# In[ ]:


total_return_result_test, total_return_result_comp1, total_return_result_comp2 = [],[],[]

for i in range(test_trial):
    total_return_result_test.append(np.mean(np.array(total_return_test[i:i + 20])))
    total_return_result_comp1.append(np.mean(np.array(total_return_comp1[i:i + 20])))
    total_return_result_comp2.append(np.mean(np.array(total_return_comp2[i:i + 20])))

total_return_result = pd.DataFrame({"test": total_return_result_test})
total_return_result_1 = pd.DataFrame({"random": total_return_result_comp1})
total_return_result_2 = pd.DataFrame({"clearance": total_return_result_comp2})

print(total_return_result)
print(total_return_result_1)
print(total_return_result_2)



df = pd.concat([pd.DataFrame(total_return_result)], axis=1)
df.to_csv(PATH + 'total_test_result.csv')
df = pd.concat([pd.DataFrame(total_return_result_1)], axis=1)
df.to_csv(PATH + 'total_test_result_1.csv')
df = pd.concat([pd.DataFrame(total_return_result_2)], axis=1)
df.to_csv(PATH + 'total_test_result_2.csv')
# In[ ]:


