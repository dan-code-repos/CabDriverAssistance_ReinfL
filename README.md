# CabDriverAssistance_ReinfL
The goal of this project is to build an RL-based algorithm which can help cab drivers maximise their profits by improving their decision-making process on the field.


Assumptions:
1. The taxis are electric cars. It can run for 30 days non-stop, i.e., 24*30 hrs. Then it needs to 
recharge itself. If the cab driver is completing his trip at that time, he’ll finish that trip and 
then stop for recharging. So, the terminal state is independent of the number of rides 
covered in a month, it is achieved as soon as the cab driver crosses 24*30 hours.
2. There are only 5 locations in the city where the cab can operate.
3. All decisions are made at hourly intervals. We won’t consider minutes and seconds for this 
project. So for example, the cab driver gets requests at 1:00 pm, then at 2:00 pm, then at 
3:00 pm and so on. So, he can decide to pick among the requests only at these times. A 
request cannot come at (say) 2.30 pm.
4. The time taken to travel from one place to another is considered in integer hours (only) and 
is dependent on the traffic. Also, the traffic is dependent on the hour of the day and the 
day of the week

State:

The state space is defined by the driver’s current location along with the time components (hour of
the day and the day of the week). A terminal state is achieved when the cab completes his 30 days, i.e., an episode is 30 days long.


Actions:

Every hour, ride requests come from customers in the form of (pick-up, drop) location. Based on the 
current ‘state’, he needs to take an action that could maximise his monthly revenue. If some 
passenger is already on board, then the driver won’t get any requests.

Therefore, an action is represented by the tuple (pick-up, drop) location. In a general scenario, the 
number of requests the cab driver can get at any state is not the same. We can model the number of 
requests as follows:

The number of requests (possible actions) at a state is dependent on the location. Say, at location A, 
you get 2 requests on average and at location B, you get 12 requests on average. That means when
at A, the cab driver is likely to get 2 customer requests from anywhere to anywhere of the form
(𝑝, 𝑞).

For each location, you can sample the number of requests from a Poisson distribution using the 
mean λ defined for each location as below:

- Location λ (of Poisson Distribution)
- Location A 2
- Location B 12
- Location C 4
- Location D 7
- Location E 8

The upper limit on these customers’ requests (𝑝, 𝑞) is 15.

Apart from these requests, the driver always has the option to go ‘offline’ (accept no ride). The noride action just moves the time component by 1 hour. So, you need to append (0,0) action to the customer requests.

There’ll never be requests of the sort where pickup and drop locations are the same. So, the action 
space A will be: (𝑚 − 1) ∗ 𝑚 + 1 for m locations. Each action will be a tuple of size 2. You can 
define action space as below:

- pick up and drop locations (𝑝, 𝑞) where p and q both take a value between 1 and m;
- (0, 0) tuple that represents ’no-ride’ action.



Time matrix:

Time Matrix is a 4-D matrix. The 4 dimensions are as below:
- Start location
- End location
- Time of the day
- Day of the week
