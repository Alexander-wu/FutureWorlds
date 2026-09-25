"""Full-data episode order and mixed short/distant target sampling."""

import random


def target_times(length, seed):
    if length < 2:
        raise ValueError(length)
    rng = random.Random(seed)
    short = list(range(1, min(length, 8)))
    medium = list(range(8, min(length, 17))) or short
    long = list(range(17, min(length, 41))) or medium
    pools = [short] * 3 + [medium] * 2 + [long] * 2
    return sorted(rng.choice(pool) for pool in pools)


def is_generator_step(shared_step_before_update, discriminator_start):
    return (
        shared_step_before_update < discriminator_start
        or (shared_step_before_update - discriminator_start) % 2 == 0
    )


def episode_order(count, epoch, seed):
    order = list(range(count))
    random.Random(seed + epoch).shuffle(order)
    return order


VALIDATION_TIMES = (1, 4, 7, 8, 12, 16, 24, 32, 40)


def gap_group(gap):
    return "short" if gap <= 7 else "medium" if gap <= 16 else "long"
