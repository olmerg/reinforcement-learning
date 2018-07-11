# -*- coding: utf-8 -*-
"""
Created on Tue Jul 10 15:05:10 2018
@author: olmer.garciab
"""

Action=['study','facebook','quit','sleep','pub']
states=['c1','c2','c3','fb','sleep']

#assuming 0.5 the policy
policy=[[0.5,0.5,0,0,0],
        [0.5,0,0,0.5,0],
        [0.5,0,0,0,0.5],
        [0,0.5,0.5,0,0],
        [0,0,0,0,0]]



#all the actions have change to only one state except pub

probablitiy_map={}

probablitiy_map['study']=[[0,1,0,0,0],
   [0,0,1,0,0,0],
   [0,0,0,0,1],
   [0,0,0,0,0],
   [0,0,0,0,1]]
probablitiy_map['facebook']=[[0,0,0,1,0],
   [0,0,0,0,0,0],
   [0,0,0,0,0],
   [0,0,0,1,0],
   [0,0,0,0,1]]
probablitiy_map['quit']=[[0,0,0,0,0],
   [0,0,0,0,0,0],
   [0,0,0,0,0],
   [1,0,0,0,0],
   [0,0,0,0,1]]

probablitiy_map['sleep']=[[0,0,0,0,0],
   [0,0,0,0,0,1],
   [0,0,0,0,0],
   [0,0,0,0,0],
   [0,0,0,0,1]]
probablitiy_map['pub']=[[0,0,0,0,0],
   [0,0,0,0,0,0],
   [0.2,0.4,0.4,0,0],
   [0,0,0,0,0],
   [0,0,0,0,1]]
 
R=[[-2,-1.,0,0,0],
   [-2,0,0,0,0],
   [10,0,0,0,1],
   [0,-1,0,0,0],
   [0,0,0,0,0]]
gamma=1


def iterative_policy_evaluation(states,policy,Action, theta=0.001, gamma=1):
    V_s = [0,0,0,0,0] # 1. initial value function
#    probablitiy_map = create_probability_map() # 2.
    N_a=len(Action)
    N=len(states)
#dynamic programming method
    delta = 100 # 3.  
    while not delta < theta: # 4.
        delta = 0 # 5.
#implement the equation
        for state in range(N): # 6. 
            v = V_s[state] # 7.
            total = 0 # 8.
            for action,k in zip(Action,range(N_a)):
                action_total = R[state][k]
                for state_prime in range(N):
                    action_total += probablitiy_map[action][state][state_prime] * (gamma * V_s[state_prime])
                total += policy[state][k] * action_total   # just an average               
            V_s[state] = total # 9.
#dynamic programming part
            delta = max(delta, abs(v - V_s[state])) # 10.
    V_s=[round(v,2) for v in V_s]
    return V_s # 11.
V_s=iterative_policy_evaluation(states,policy,Action, theta=.00001, gamma=1)
print(V_s)
