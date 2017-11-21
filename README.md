# Swain Bot
Created by Devin Light

## Introduction
### What is League of Legends?
League of Legends (abbreviated as LoL, or League) is a multiplayer online battle arena (MOBA) game developed by Riot Games which features two teams of five players each competing in head-to-head matches with the ultimate goal of destroying the opposing teams nexus structure. The game boasts millions of monthly players and a large competitive scene involving dozens of teams participating in both national and international tournaments. The game takes place across two broadly defined phases. In the first phase (or Drafting phase), each side takes turns assembling their team by selecting a unique character (called a champion) from a pool of almost 140 (as of this writing) without replacement. Then, in the second phase (or Play phase), each player in the match takes control of one of the champions chosen by their team and attempts to claim victory. Although not strictly required by the game, over the years players usually elect to play their champion in one of five roles named after the location on the map in which they typically start the game, and often corresponding to the amount of resources that player will have devoted to them:

- Position 1 (primary farm)-> ADC/Marksman<sup>1</sup>
- Position 2 (secondary farm)-> Middle
- Position 3 (tertiary farm)-> Top
- Position 4 (farming support)-> Jungle
- Position 5 (primary support)-> Support<sup>1</sup>

<sup>1</sup> Typically the ADC and Support begin the game together same lane and are collectively called 'Bottom'.

Each champion has distinct set of characteristics and abilities that allows them excel in certain situations while struggling in others. In order to maximize the odds of victory, it is important that the team assembled during the drafting phase simultaneously plays into one cohesive set of strengths and disrupts or plays well against the strengths of the opposing draft. There are two types of submissions made during the drafting phase. In the banning portions of the drafting phase champions are removed from the pool of allowed submissions, whereas champions are added to the roster of the submitting team during the pick phases. The draft phase alternates between banning and picking until both teams have a full roster of five champions, at which point the game is played. The structure of the drafting phase is displayed in Figure 1. Note the asymmetry between teams (for example Blue bans first in ban phase one, while Red bans first in ban phase two) and between the phases themselves (ban phases always alternate sides, while pick phases "snake" between teams).

![Figure 1](common/images/league_draft_structure.png "Figure 1")

### What is Swain Bot?
Swain Bot (named after the champion Swain whose moniker is "The Master Tactician") is a machine learning application built in Python and using Google's Tensorflow framework. Swain Bot is designed to analyze the drafting phase of competitive League of Legends matches. Given a state of the draft which includes full information of our team's submissions (champions and positions) and partial information of the opponent's submissions (champions only), Swain Bot attemps to suggest picks and bans that are well-suited for our draft.

### What do we hope to do with Swain Bot?
Our objective with Swain Bot is to be able to provide insight into a few questions concering League's draft phase:
- Can we estimate how valuable certain submissions are for a given state of the draft?
- Is there a common structure or theme to how professional League teams draft?
- Can we identify where losing drafts go awry?
- Can we estimate the "winner" of a completed draft?

## Assumptions and Limitations
Every model tasked with approaching a difficult problem is predicated on some number assumptions which in turn define the boundaries that the model can safely be applied. Swain Bot is no exception, so here we outline and discuss some of the explicit assumptions being made going into the construction of the underlying model Swain Bot uses to make its predictions. Some of the assumptions are more impactful than others and some could be removed in the future to improve Swain Bot's performance, but are in place for now for various reasons.

1. Swain Bot is limited to data from recorded professionally played games from the "Big 5" regions (NALCS, EULCS, LCK, LPL, and LMS). Limiting potential data sources to competitive leagues is very restrictive when compared to the pool of amature matches played on local servers across the world. However, this assumption is in place as a result of changes in Riot's (otherwise exceptional) API which effectively randomizes the order in which the champion submissions for a draft are presented, rendering it impossible to recover the sequence of draft states that make up the complete information regarding the draft. Should the API be changed in the future Swain Bot will be capable of learning from amature matches as well. 

2. Swain Bot does not recieve information about either the patch the game was played on or the teams involved in the match. Not including the patch allows us to stretch the data as much as we can given the restricted pool. Although the effectiveness of a champion might change as they are tuned between patches, it is unlikely that they are changed so much that the situations that the champion would normally be picked in are dramatically different. Nevertheless substantial champion changes have occured in the past, usually in the form of a total redesign. Additionally, although team data for competitive matches is available during the draft, Swain Bot's primary objective is to identify the most effective submissions for a given draft state rather than predict what a specific team might select in that situation.  Nevertheless it would be possible to combine Swain Bot's output with information about a team's drafting tendencies (using ensemble techniques like stacking) to produce a final prediction which both suits the draft and is likely to be chosen by the team. However we will leave this work for later.

3. Swain Bot's objective is to associate the combination of a state and a potential submission with a value and to suggest taking the action which has the highest value. This valuation should be based primarily on what is likely to win the draft (or move us towards a winning state), and partly on what is likely to be done. 
Although these two goals may be correllated (a champion that is highly-valued might also be the one selected most frequently) they are not necessarily the same since, for example, teams may be biased towards or against specific strategies or champions.

4. Swain Bot's objective to estimate the value of submissions for a given draft state is commonly approached using techniques from Reinforcement Learning (RL). RL methods have been successfully used in a variety of situations such as teaching robots how to move, playing ATARI games, and even [playing DOTA2](https://blog.openai.com/dota-2/). A common element to most RL applications is the ability to automatically explore and evaluate states as they are encountered in a simulated environment. However, Swain Bot is not capable of automatically playing out the drafts in order to evaluate them (yet..) and so is dependent on the data it observes originating from games that were previously played. This scenario is reminiscent of a Supervised Learning (SL) problem called behavioral cloning, where the task is to learn and replicate the policy outlined by an expert. However, behavioral cloning attempts to directly mimic the expert policy and does not include the estimation of action values. As a result, Swain Bot implements an RL algorithm to estimate action values (Q-Learning), but trained using expertly-generated data. In practice, this means that the predictions made by Swain Bot can only have an expectation of accuracy when following trajectories that are similiar to the paths prescribed by the training data (which we will see later).

## Methods
This section is not designed to be too technical, but rather give some insight into how Swain Bot is implemented and some important modifications that helped with the learning process. For some excellent and thorough discussions on RL, check out the following:
- [David Silver's course on RL](http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching.html) [(with video lectures)](https://www.youtube.com/watch?v=2pWv7GOvuf0)
- [Reinforcement Learning](http://incompleteideas.net/sutton/book/the-book.html) By Sutton and Barto
- [The DeepMind ATARI paper](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf)
- [Dueling DQNs](https://arxiv.org/pdf/1511.06581.pdf)
- And finally a [few](http://outlace.com/rlpart3.html), [useful](https://www.intelnervana.com/demystifying-deep-reinforcement-learning/), [tutorials](https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-0-q-learning-with-tables-and-neural-networks-d195264329d0)

### Representing Draft States and Actions
Each of the _N_ eligible champions (138 as of this writing) in a draft is represented by a unique `champion_id` integer and every position in the game (five positions per team plus banned champions) is given by a `position_id`. An _action_ (or _submission_) to the draft is defined as tuple of the form `(champion_id, position_id) = (i,j)` representing the selection of champion `i` to position `j` in the draft. We can represent the _draft state_ as a boolean matrix _S_ where _S_(i,j) = 1 if the ith champion has been submitted to the draft at the jth position. The size of _S_ is determined by how much information about the positions is available to the drafter:
- In a _completely informed_ draft all position information is known so _S_ is an `N x 11` matrix (10 positions + bans).
- In a _partially informed_ draft position information is only known for the drafter's team whereas only the `champion_id`s are known for the opponent's team. As a result _S_ is given by an `N x 7` matrix (5 positions + bans + enemy champions).
Note that the _S_ is a sparse matrix since for any given state of a draft, there are no more than 10 picks and 10 bans that have been submitted so there are no more than 20 non-zero entries in _S_ at any given time. Swain Bot operates using partially informed draft states as inputs which may be obtained by projecting the five columns in the completely informed state corresponding to the positions in the opponent's draft onto a single column. Finally, we define the _actionable state_ to be the submatrix of _S_ corresponding to the actions the drafter may submit-- that is the columns corresponding to bans as well as the drafter's five submittable positions. 

For either completely or partially informed states, the draft can be fully recovered using the sequence of states `(S_0, S_1,..., S_n)` transitioned through during the draft. This sequence defines a Markov chain since given the current state _s_, the value of the succesor state _s'_ is independent of the states that were transitioned through before _s_. In other words, the states we are able to transition to away from _s_ depend only on _s_ itself, and not on the states that were seen on the way to _s_. To complete the framework of drafting as a Markov Decision Process (MDP) we must still define a reward schedule and discount factor. 

The discount factor is a scalar value between 0 and 1 that governs the present value of future expected rewards. Two common reasons to use a discount factor are to express uncertainty about the future and to capture the extra value of immediate rewards over delayed rewards (e.g. if the reward is financial, an immediate reward is worth more than a delayed reward becuase that immediate reward can be used to earn interest). Typical discount factor values are in the `0.9` to `0.99`. Swain Bot uses a dicount factor of 

<img src="https://raw.githubusercontent.com/lightd22/swainBot/master/common/images/discount_factor.png" height="50%" width="50%">

The reward schedule is a vital component of the MDP and ultimately determines what policy the model will converge towards. As previously discussed, Swain Bot's objective is to first and foremost select the action which moves the state 

## Disclaimer
Swain Bot isn’t endorsed by Riot Games and doesn’t reflect the views or opinions of Riot Games or anyone officially involved in producing or managing League of Legends. League of Legends and Riot Games are trademarks or registered trademarks of Riot Games, Inc. League of Legends © Riot Games, Inc.
