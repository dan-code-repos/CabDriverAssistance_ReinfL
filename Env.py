# Author    :   Danish Ansari
# Date      :   28.12.2021
# Version   :   1.4
# Description:  Create the environment

# Import routines

import numpy as np
import math
import random

# Defining hyperparameters
m = 5 # number of cities, ranges from 0 ..... m-1
t = 24 # number of hours, ranges from 0 .... t-1
d = 7  # number of days, ranges from 0 ... d-1
C = 5 # Per hour fuel and other costs
R = 9 # per hour revenue from a passenger

class CabDriver():

    def __init__(self):
        """initialise your state and define your action space and state space"""        
        self.state_space = [(loc,time,day) for loc in range(m) for time in range(t) for day in range(d)]
        self.action_space = [(pick,drop) for pick in range(m) for drop in range(m) if pick!=drop or pick==0]
        self.state_init = random.choice(self.state_space)
        self.state_size = m + t + d
        self.action_size = len(self.action_space)        
        self.time_elapsed_episode = 0
        self.isTerminal = False
        # Start the first round
        self.reset()
        


    # Encoding state (or state-action) for NN input
    def state_encod_arch2(self, state):
        # state = [i , t , d]
        """convert the state into a vector so that it can be fed to the NN. This method converts a given state into a vector format. Hint: The vector is of size m + t + d."""
        loc,time,day = state

        locArray = np.zeros(m, dtype=int)
        timeArray = np.zeros(t, dtype=int)
        dayArray = np.zeros(d, dtype=int)

        locArray[loc] = 1
        timeArray[time] = 1
        dayArray[day] = 1

        state_encod = np.concatenate((locArray,timeArray,dayArray))

        return state_encod



    # Use this function if you are using architecture-2 
    def state_encod_arch1(self, state, action):
        """convert the (state-action) into a vector so that it can be fed to the NN. This method converts a given state-action pair into a vector format. Hint: The vector is of size m + t + d + m + m."""
        #(x,t,d) , (p,q)

        loc, time, day = state
        fromLoc, toLoc = action

        locArray = np.zeros(m, dtype=int)
        timeArray = np.zeros(t, dtype=int)
        dayArray = np.zeros(d, dtype=int)

        fromLocArray = np.zeroes(m, dtype=int)
        toLocArray = np.zeroes(m, dtype=int)

        locArray[loc] = 1
        timeArray[time] = 1
        dayArray[day] = 1

        fromLocArray[fromLoc] = 1
        toLocArray[toLoc] = 1

        state_encod = np.concatenate((locArray, timeArray, dayArray, fromLocArray, toLocArray))

        return state_encod
    

    ## Getting number of requests
    def requests(self, state):
        """Determining the number of requests basis the location. 
        Use the table specified in the MDP and complete for rest of the locations"""
        location = state[0]
        if location == 0:
            requests = np.random.poisson(2)
        if location == 1:
            requests = np.random.poisson(12)
        if location == 2:
            requests = np.random.poisson(4)
        if location == 3:
            requests = np.random.poisson(7)
        if location == 4:
            requests = np.random.poisson(8)
        if requests > 15:
            requests = 15
            
        #[1,2,3,4,5,6,7,] -> [5,3]
        #Action space size  = (m-1)*m +1 
        #                   = self.action_size
        possible_actions_index = random.sample(range(1, (m-1)*m +1), requests) + [0] # (0,0) is not considered as customer request
        actions = [self.action_space[i] for i in possible_actions_index]   
        #actions.append((0,0))

        return possible_actions_index,actions   



    def shift_time_day(self, time, days, shiftByTime):
        shiftByTime = int(shiftByTime)
        updatedTime = time+shiftByTime
        updatedDays = days
        if updatedTime >= 24:
            updatedTime = updatedTime%24
            daysShift = updatedTime//24
            updatedDays += daysShift
        return updatedTime, updatedDays



    def get_next_state_and_times(self, state, action, Time_matrix):
        """Takes state and action as input and returns next state"""
        currLoc,currTime,currDay = state
        pickLoc,dropLoc = action

        timeIdle = 0
        timeToReachCust = 0
        timeOfRide = 0
        
        if (pickLoc==0 and dropLoc==0):
            timeIdle = 1
            timeToReachCust = 0
            timeOfRide = 0            
            nextLoc = currLoc
        elif (currLoc==pickLoc):
            timeIdle = 0
            timeToReachCust = 0
            timeOfRide = Time_matrix[currLoc][dropLoc][currTime][currDay]
            nextLoc = dropLoc
        else:
            timeIdle = 0
            timeToReachCust = Time_matrix[currLoc][pickLoc][currTime][currDay]
            updatedTime,updatedDay = self.shift_time_day(currTime, 
                                                        currDay, 
                                                        timeToReachCust)
            timeOfRide = Time_matrix[currLoc][dropLoc][updatedTime][updatedDay]           
            nextLoc = dropLoc

        totalTime = timeToReachCust + timeOfRide + timeIdle
        nextTime, nextDay = self.shift_time_day(currTime,
                                                currDay,
                                                totalTime)
        next_state = [nextLoc, nextTime, nextDay]
        return next_state, timeIdle, timeToReachCust, timeOfRide



    def get_reward(self, timeIdle, timeToReachCust, timeOfRide):
        """Takes in state, action and Time-matrix and returns the reward"""
        #timeIdle, timeToReachCust, timeOfRide = self.get_next_state_and_times(state,action,Time_matrix)[1:3]
        reward = (R*timeOfRide) - (C*(timeIdle + timeToReachCust + timeOfRide))
        return reward



    def step(self, state, action, Time_matrix):
        next_state, timeIdle, timeToReachCust, timeOfRide = self.get_next_state_and_times(state, action, Time_matrix)
        reward = self.get_reward(timeIdle, timeToReachCust, timeOfRide)
        timeTotalForStep = timeIdle + timeToReachCust + timeOfRide
        self.time_elapsed_episode  += timeTotalForStep
        if self.time_elapsed_episode > 720:
            self.isTerminal = True
        return next_state, reward, self.isTerminal



    def reset(self):
        return self.action_space, self.state_space, self.state_init
