"""Manual local match harness. Run: python tests/match_random.py"""

import sys

from kaggle_environments import make


SEEDS = [42, 7, 13, 99, 256]
WIN_THRESHOLD = 0.8


def play_one(seed):
    env = make("crawl", configuration={"randomSeed": seed}, debug=True)
    env.run(["main.py", "random"])
    rewards = [env.steps[-1][i].reward for i in range(2)]
    if rewards[0] is None or rewards[1] is None:
        return None
    return rewards[0] > rewards[1]


def main():
    results = []
    for seed in SEEDS:
        outcome = play_one(seed)
        results.append(outcome)
        print(f"seed={seed} → {'WIN' if outcome else 'LOSS/TIE'}")
    wins = sum(1 for r in results if r)
    rate = wins / len(SEEDS)
    print(f"win rate: {rate:.0%} ({wins}/{len(SEEDS)})")
    sys.exit(0 if rate >= WIN_THRESHOLD else 1)


if __name__ == "__main__":
    main()
