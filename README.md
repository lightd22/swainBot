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

## Disclaimer
Swain Bot isn’t endorsed by Riot Games and doesn’t reflect the views or opinions of Riot Games or anyone officially involved in producing or managing League of Legends. League of Legends and Riot Games are trademarks or registered trademarks of Riot Games, Inc. League of Legends © Riot Games, Inc.
