## Maze Crawler
Navigate an infinite scrolling maze with fog of war in this 1v1 simulation

## Overview
The goal of this competition is to create and/or train AI bots to play a novel maze crawling game against other submitted agents.

## Description
Welcome to Maze Crawler! In this infinite scrolling strategy game, you must lead a robotic expedition through a shifting, fog-shrouded labyrinth. The floor is literally disappearing beneath you, so you must stay ahead of the southern boundary, manage your energy, and outmaneuver your opponent to be the last factory standing.

## The Challenge
You control a mobile Factory and a fleet of specialized robots. Your objective is survival in a mirrored maze. As the game progresses, the bottom of the map advances faster and faster, destroying everything in its path. You must push upward, but the path is blocked by walls, fog, and an enemy player with the exact same goal.

The match ends when a factory is destroyed by the scrolling boundary or by a direct collision with the enemy factory.

Can you balance exploration, resource management, and aggressive expansion before the ground vanishes?

## Evaluation
Each day your team is able to submit up to 5 agents (bots) to the competition. Each submission will play Episodes (games) against other bots on the ladder that have a similar skill rating. Over time skill ratings will go up with wins or down with losses and evened out with ties. To reduce the number of bots playing and increase the number of episodes each team participates in, we only track the latest 2 submissions and use those for final submissions.

Every bot submitted will continue to play episodes until the end of the competition, with newer bots playing a much more frequent number of episodes. On the leaderboard, only your best scoring bot will be shown, but you can track the progress of all of your submissions on your Submissions page.

Each Submission has an estimated Skill Rating which is modeled by a Gaussian N(μ,σ2) where μ is the estimated skill and σ represents the uncertainty of that estimate which will decrease over time.

When you upload a Submission, we first play a Validation Episode where that Submission plays against copies of itself to make sure it works properly. If the Episode fails, the Submission is marked as Error and you can download the agent logs to help figure out why. Otherwise, we initialize the Submission with μ0=600 and it joins the pool of All Submissions for ongoing evaluation.

We repeatedly run Episodes from the pool of All Submissions, and try to pick Submissions with similar ratings for fair matches. Newly submitted agents will be given an increased rate in the number of episodes run to give you faster feedback.

## How to Play Maze Crawler
Overview
Each player starts with a single Factory robot near the bottom of a 20x20 maze. The maze scrolls northward over time — the southern boundary advances, destroying anything left behind. The last player with a surviving factory wins.

The maze has east/west symmetry: the left half mirrors the right half, with occasional doors connecting the two sides. Players start on opposite halves.

## Robot Types
Type	Cost	Max Energy	Move Period	Vision	Special Abilities
Factory	—	unlimited	2 turns	4	BUILD, JUMP (20-turn CD), indestructible
Scout	50	100	1 turn	5	Fast explorer
Worker	200	300	2 turns	3	BUILD_DIR / REMOVE_DIR walls (100 energy)
Miner	300	500	2 turns	3	TRANSFORM into energy mine (requires mining node)
All robots consume 1 energy per turn. Robots with 0 energy are forced idle.

## Actions
Each turn, you return a dictionary mapping robot UIDs to action strings.

## Movement
NORTH, SOUTH, EAST, WEST — Move one cell in that direction (blocked by walls). A unit that successfully moves off the north or south edge of the board (no wall blocking) is destroyed. East/west are always blocked by perimeter walls.

## Factory Actions
BUILD_SCOUT, BUILD_WORKER, BUILD_MINER — Spawn a new robot in the cell north of the factory. Requires no wall between factory and spawn cell. 10-turn cooldown between builds. The new robot is placed before the movement phase, so it counts as a stationary occupant during combat — if an enemy on that cell moves away the same turn, the new robot lands safely; otherwise crush combat resolves on the spawn cell.
JUMP_NORTH, JUMP_SOUTH, JUMP_EAST, JUMP_WEST — Leap 2 cells in a direction, ignoring all walls. The jump always happens and the cooldown is consumed. If the landing cell is off the board, the factory is destroyed. 20-turn cooldown.

## Worker Actions
BUILD_NORTH, BUILD_SOUTH, BUILD_EAST, BUILD_WEST — Add a wall between the worker's cell and the adjacent cell in that direction. Costs 100 energy. The worker survives.
REMOVE_NORTH, REMOVE_SOUTH, REMOVE_EAST, REMOVE_WEST — Remove the wall between the worker's cell and the adjacent cell. Costs 100 energy. The worker survives.
Fixed walls (cannot be modified): the outer perimeter (E/W of the leftmost and rightmost columns) and the central mirror axis (E of column width/2 - 1 and W of column width/2). BUILD/REMOVE on a fixed wall, or where the wall is already in the requested state, still costs 100 energy but has no effect. Fixed walls are drawn as double lines in the visualizer.

## Miner Actions
TRANSFORM — Destroy the miner and create an energy mine at its position. Requires the miner to be standing on a mining node. Costs 100 energy. The mine receives the miner's remaining energy (up to mine max).

## Other
TRANSFER_NORTH, TRANSFER_SOUTH, TRANSFER_EAST, TRANSFER_WEST — Send all energy to an adjacent friendly robot. Blocked by walls. Target's energy is capped at its max (factory has no cap).
IDLE — Do nothing.

## Combat
When two or more robots end the turn on the same cell, crush rules apply — ownership doesn't matter; friendly fire is real.

Crush hierarchy: Factory > Miner > Worker > Scout. The stronger type destroys the weaker.
Same type: Both (or all) robots of that type are destroyed. Two friendly scouts walking onto the same cell mutually annihilate.
Factory: Indestructible against any non-factory unit (friendly or enemy) and crushes them. Two enemy factories on the same cell mutually destroy each other (game ends, see Win Conditions).
Crystal on combat cell: The surviving robot (if any) collects the crystal energy. If no robot survives, the crystal is consumed.
Spawning a robot onto an occupied cell triggers combat normally — including friendly fire if the spawn cell is held by your own unit.

### Map Features
## Crystals
Scattered throughout the maze (6% density per cell). Any robot moving onto a crystal collects its energy (10-50). Crystals are visible only within vision range and are not remembered after leaving range.

## Mining Nodes
Rare locations (3% density) marked on the map where miners can transform into mines. A mining node is consumed when a mine is created on it. Mining nodes never overlap with crystals.

## Mines
Created by miners using TRANSFORM on a mining node. Mines generate 50 energy per turn up to a maximum of 1000 energy. Friendly robots standing on a mine collect energy from it. Mines are remembered once discovered (even outside vision range).

## Fog of War
Each robot has a vision range (Manhattan distance). You can only see what's within the combined vision of all your robots.

Data	    Visible in range	     Remembered after leaving range
Walls/layout	  Yes	               Yes (permanent)
Crystals	      Yes	                      No
Enemy robots	  Yes	                      No
Own robots	     Always	                      N/A
Mines (any owner)	Yes	            Yes (permanent, last-known state)
Mining nodes	    Yes	                      No

## Maze Scrolling
The southern boundary advances over time, destroying all robots, mines, and crystals below it.

Start: Scrolls once every 4 turns
Ramp: Linearly increases speed over 400 steps
End: Scrolls every turn from step 400 onward (until game end at step 500)
If a factory falls below the southern boundary, that player is eliminated.

## Turn Processing Order
Cooldown tick — Decrement move, jump, and build cooldowns
Action validation — Verify action legality
Energy consumption — Each robot loses 1 energy; 0-energy robots forced idle
Special actions — TRANSFORM (miner), BUILD_DIR/REMOVE_DIR (worker walls), BUILD_SCOUT/WORKER/MINER (factory), TRANSFER (in that order)
Movement + combat — Simultaneous movement, then resolve collisions
Crystal collection — Robots on crystal cells collect energy
Mine energy fill — Robots on friendly mines collect energy
Mine generation — Each mine gains 50 energy (up to max 1000)
Scroll advancement — Advance boundaries, generate new row, place crystals/nodes
Boundary destruction — Destroy robots/mines below southern boundary
Win condition check
Update observations — Compute fog of war, build per-player views

## Win Conditions
Survival: If one factory is destroyed (by scrolling below the boundary), the other player wins.
Simultaneous elimination: If both factories are destroyed on the same turn (e.g. mutual factory collision, or both scrolled off), apply the tiebreaker cascade.
Time limit (step 500): If both factories are still alive, apply the tiebreaker cascade.

## Tiebreaker cascade
Total energy across all robots — higher wins
Unit count across all robots — higher wins
True draw — both players receive reward 0.5

## Reward
Alive (mid-game): Total energy across all your robots
Win by tiebreaker cascade: 1
Loss by tiebreaker cascade: 0
Draw: 0.5
Eliminated (opponent survives): step_eliminated - episodeSteps - 1 (negative value); winner gets total energy

## Observation Format
def agent(obs, config):
    obs.player        # Your player index (0 or 1)
    obs.walls         # Flat array: index = (row - southBound) * width + col
                      # Values: wall bitfield, -1 = undiscovered
    obs.crystals      # {"col,row": energy} — only currently visible
    obs.robots        # {"uid": [type, col, row, energy, owner, move_cd, jump_cd, build_cd]}
    obs.mines         # {"col,row": [energy, maxEnergy, owner]} — remembered once seen
    obs.miningNodes   # {"col,row": 1} — only currently visible
    obs.southBound    # Current southern boundary row
    obs.northBound    # Current northern boundary row


## Wall Bitfield
N = 1, E = 2, S = 4, W = 8
Check for a wall: if wall_value & 1: means there's a north wall. Fixed walls (perimeter and middle axis) have the same bitfield representation but cannot be modified by workers; the visualizer renders them as double lines.

## Configuration Defaults
Parameter	Default 	Description
episodeSteps	501	Max turns
width	20	Maze width
height	20	Visible window height
factoryEnergy	1000	Starting factory energy
scoutCost	50	Energy to build scout (also the energy a freshly-built scout spawns with)
workerCost	200	Energy to build worker (also the energy a freshly-built worker spawns with)
minerCost	300	Energy to build miner (also the energy a freshly-built miner spawns with)
scoutMaxEnergy	100	Max energy a scout can carry
workerMaxEnergy	300	Max energy a worker can carry
minerMaxEnergy	500	Max energy a miner can carry
wallBuildCost	100	Energy per worker BUILD_DIR (charged even on no-op)
wallRemoveCost	100	Energy per worker REMOVE_DIR (charged even on no-op)
transformCost	100	Energy for miner transform
mineMaxEnergy	1000	Max energy a mine stores
mineRate	50	Mine energy generation per turn
energyPerTurn	1	Energy consumed per robot per turn
factoryBuildCooldown	10	Turns between builds
factoryJumpCooldown	20	Turns between jumps
factoryMovePeriod	2	Factory moves every N turns
workerMovePeriod	2	Worker moves every N turns
minerMovePeriod	2	Miner moves every N turns
visionFactory	4	Factory vision range
visionScout	5	Scout vision range
visionWorker	3	Worker vision range
visionMiner	3	Miner vision range
scrollStartInterval	4	Initial turns between scrolls
scrollEndInterval	1	Final turns between scrolls
scrollRampSteps	400	Step when max scroll speed reached
crystalDensity	0.06	Crystal spawn probability per cell
miningNodeDensity	0.03	Mining node spawn probability per cell
doorProbability	0.08	Door probability between maze halves


## Dataset Description
This is the folder for the Python kit. Please make sure to read the instructions as they are important regarding how you will write a bot and submit it to the competition.

For kits in other languages please see this example from the Lux AI Challenge Github repository

