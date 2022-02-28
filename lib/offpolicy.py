import time
from collections import defaultdict
from typing import Callable, Dict, Optional, Union

import numpy as np
import tqdm

from tianshou.data import Collector
from tianshou.policy import BasePolicy
from tianshou.trainer import gather_info, test_episode
from tianshou.utils import BaseLogger, LazyLogger, MovAvg, tqdm_config


def offpolicy_trainer(
    policy: BasePolicy,
    train_collector: Collector,
    test_collector: Optional[Collector],
    max_epoch: int,
    step_per_epoch: int,
    step_per_collect: int,
    episode_per_test: int,
    batch_size: int,
    update_per_step: Union[int, float] = 1,
    test_fn: Optional[Callable[[int, Optional[int]], None]] = None,
    stop_fn: Optional[Callable[[float], bool]] = None,
    save_fn: Optional[Callable[[BasePolicy], None]] = None,
    save_checkpoint_fn: Optional[Callable[[int, int, int], None]] = None,
    resume_from_log: bool = False,
    reward_metric: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    logger=None,
    writer=None, 
    verbose: bool = True,
    test_in_train: bool = True,
) -> Dict[str, Union[float, str]]:
    start_epoch, env_step, gradient_step = 0, 0, 0
    if resume_from_log:
        start_epoch, env_step, gradient_step = logger.restore_data()
    last_rew, last_len = 0.0, 0
    stat: Dict[str, MovAvg] = defaultdict(MovAvg)
    start_time = time.time()
    train_collector.reset_stat()
    test_in_train = test_in_train and (
        train_collector.policy == policy and test_collector is not None
    )

    if test_collector is not None:
        test_c: Collector = test_collector  # for mypy
        test_collector.reset_stat()
        test_result = test_episode(
            policy, test_c, test_fn, start_epoch, episode_per_test, logger, env_step,
            reward_metric
        )
        best_epoch = start_epoch
        best_reward, best_reward_std = test_result["rew"], test_result["rew_std"]
    total_step_count = 0
    if save_fn:
        save_fn(policy)

    for epoch in range(1 + start_epoch, 1 + max_epoch):
        # train
        policy.train()
        with tqdm.tqdm(
            total=step_per_epoch, desc=f"Epoch #{epoch}", **tqdm_config
        ) as t:
            while t.n < t.total:
                result = train_collector.collect(n_step=step_per_collect)
                if result["n/ep"] > 0 and reward_metric:
                    rew = reward_metric(result["rews"])
                    result.update(rews=rew, rew=rew.mean(), rew_std=rew.std())
                env_step += int(result["n/st"])
                t.update(result["n/st"])
                logger.log_train_data(result, env_step)
                writer.add_scalar("mean_reward", result["True_rew"].mean(), total_step_count)
                writer.add_scalar("std_reward", result["True_rew"].std(), total_step_count)
                last_rew = result['rew'] if result["n/ep"] > 0 else last_rew
                last_len = result['len'] if result["n/ep"] > 0 else last_len
                data = {
                    "env_step": str(env_step),
                    "rew": f"{last_rew:.2f}",
                    "len": str(int(last_len)),
                    "n/ep": str(int(result["n/ep"])),
                    "n/st": str(int(result["n/st"])),
                    "rew_t": str(int(result["True_rew"].mean())),
                }
                if result["n/ep"] > 0:
                    if test_in_train and stop_fn and stop_fn(result["rew"]):
                        test_result = test_episode(
                            policy, test_c, test_fn, epoch, episode_per_test, logger,
                            env_step
                        )
                        if stop_fn(test_result["rew"]):
                            if save_fn:
                                save_fn(policy)
                            logger.save_data(
                                epoch, env_step, gradient_step, save_checkpoint_fn
                            )
                            t.set_postfix(**data)
                            return gather_info(
                                start_time, train_collector, test_collector,
                                test_result["rew"], test_result["rew_std"]
                            )
                        else:
                            policy.train()
                for _ in range(round(update_per_step * result["n/st"])):
                    gradient_step += 1
                    losses = policy.update(batch_size, train_collector.buffer)
                    for k in losses.keys():
                        stat[k].add(losses[k])
                        losses[k] = stat[k].get()
                        data[k] = f"{losses[k]:.3f}"
                    logger.log_update_data(losses, gradient_step)
                    t.set_postfix(**data)
            if t.n <= t.total:
                t.update()
        logger.save_data(epoch, env_step, gradient_step, save_checkpoint_fn)
        # test
        if test_collector is not None:
            test_result = test_episode(
                policy, test_c, test_fn, epoch, episode_per_test, logger, env_step,
                reward_metric
            )
            rew, rew_std = test_result["rew"], test_result["rew_std"]
            if best_epoch < 0 or best_reward < rew:
                best_epoch, best_reward, best_reward_std = epoch, rew, rew_std
                if save_fn:
                    save_fn(policy)
            if verbose:
                print(
                    f"Epoch #{epoch}: test_reward: {rew:.6f} ± {rew_std:.6f}, best_rew"
                    f"ard: {best_reward:.6f} ± {best_reward_std:.6f} in #{best_epoch}"
                )
            if stop_fn and stop_fn(best_reward):
                break

    if test_collector is None and save_fn:
        save_fn(policy)

    if test_collector is None:
        return gather_info(start_time, train_collector, None, 0.0, 0.0)
    else:
        return gather_info(
            start_time, train_collector, test_collector, best_reward, best_reward_std
        )
