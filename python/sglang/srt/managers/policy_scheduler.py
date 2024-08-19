"""
Copyright 2023-2024 SGLang Team
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

"""Request policy scheduler"""

import random
from collections import defaultdict
from contextlib import contextmanager

from sglang.srt.managers.schedule_batch import Req, ScheduleBatch


class PolicyScheduler:
    def __init__(
        self,
        policy,
        max_running_seqs,
        max_prefill_num_tokens,
        max_total_num_tokens,
        tree_cache,
    ):
        if tree_cache.disable and policy == "lpm":
            # LMP is meaningless when the tree cache is disabled.
            policy = "fcfs"

        self.policy = policy
        self.max_running_seqs = max_running_seqs
        self.max_prefill_num_tokens = max_prefill_num_tokens
        self.max_total_num_tokens = max_total_num_tokens
        self.tree_cache = tree_cache

    def get_priority_queue(self, waiting_queue):
        if self.policy == "lpm":
            # longest prefix match
            waiting_queue.sort(key=lambda x: -len(x.prefix_indices))
            return waiting_queue
        elif self.policy == "fcfs":
            # first come first serve
            return waiting_queue
        elif self.policy == "lof":
            # longest output first
            waiting_queue.sort(key=lambda x: -x.sampling_params.max_new_tokens)
            return waiting_queue
        elif self.policy == "random":
            random.shuffle(waiting_queue)
            return waiting_queue
        elif self.policy == "dfs-weight":
            last_node_to_reqs = defaultdict(list)
            for req in waiting_queue:
                last_node_to_reqs[req.last_node].append(req)

            node_to_weight = defaultdict(int)
            for node in last_node_to_reqs:
                node_to_weight[node] = len(last_node_to_reqs[node])
            self.calc_weight(self.tree_cache.root_node, node_to_weight)

            q = []
            self.get_dfs_priority(
                self.tree_cache.root_node, node_to_weight, last_node_to_reqs, q
            )
            assert len(q) == len(waiting_queue)
            return q
        else:
            raise ValueError(f"Unknown schedule_policy: {self.policy}")

    def calc_weight(self, cur_node, node_to_weight):
        for child in cur_node.children.values():
            self.calc_weight(child, node_to_weight)
            node_to_weight[cur_node] += node_to_weight[child]

    def get_dfs_priority(self, cur_node, node_to_priority, last_node_to_reqs, q):
        childs = [child for child in cur_node.children.values()]
        childs.sort(key=lambda x: -node_to_priority[x])
        for child in childs:
            self.get_dfs_priority(child, node_to_priority, last_node_to_reqs, q)
        q.extend(last_node_to_reqs[cur_node])


class PrefillAdder:
    def __init__(
        self,
        tree_cache,
        rem_total_tokens,
        rem_input_tokens,
        rem_chunk_tokens,
    ):
        self.tree_cache = tree_cache
        self.rem_total_tokens = rem_total_tokens
        self.rem_input_tokens = rem_input_tokens
        self.rem_chunk_tokens = rem_chunk_tokens

        self.can_run_list = []
        self.new_inflight_req = None
        self.log_hit_tokens = 0
        self.log_input_tokens = 0

    def no_remaining_tokens(self):
        return (
            self.rem_total_tokens <= 0
            or self.rem_input_tokens <= 0
            or (
                self.rem_chunk_tokens <= 0
                if self.rem_chunk_tokens is not None
                else False
            )
        )

    def remove_running_tokens(
        self, running_batch: ScheduleBatch, new_token_ratio: float
    ):
        self.rem_total_tokens -= sum(
            [
                (r.sampling_params.max_new_tokens - len(r.output_ids)) * new_token_ratio
                for r in running_batch.reqs
            ]
        )

    def _prefill_one_req(
        self, prefix_len: int, extend_input_len: int, max_new_tokens: int
    ):
        self.rem_total_tokens -= extend_input_len + max_new_tokens
        self.rem_input_tokens -= extend_input_len
        if self.rem_chunk_tokens is not None:
            self.rem_chunk_tokens -= extend_input_len

        self.log_hit_tokens += prefix_len
        self.log_input_tokens += extend_input_len

    def add_inflight_req(self, req: Req):
        req.input_ids = req.origin_input_ids + req.output_ids
        req.extend_input_len = len(req.input_ids) - len(req.prefix_indices)
        truncated = req.extend_input_len > self.rem_chunk_tokens
        req.extend_input_len = min(req.extend_input_len, self.rem_chunk_tokens)
        req.input_ids = req.input_ids[: len(req.prefix_indices) + req.extend_input_len]
        self.can_run_list.append(req)

        self._prefill_one_req(
            len(req.prefix_indices),
            req.extend_input_len,
            req.sampling_params.max_new_tokens if not truncated else 0,
        )

        # Return if chunked prefill not finished
        return req if truncated else None

    @contextmanager
    def _lock_node(self, last_node):
        try:
            delta = self.tree_cache.inc_lock_ref(last_node)
            self.rem_total_tokens += delta
            yield None
        finally:
            delta = self.tree_cache.dec_lock_ref(last_node)
            self.rem_total_tokens += delta

    def add_one_req(self, req: Req):
        total_tokens = req.extend_input_len + req.sampling_params.max_new_tokens
        input_tokens = req.extend_input_len
        prefix_len = len(req.prefix_indices)

        if total_tokens >= self.rem_total_tokens:
            return False

        if input_tokens > self.rem_input_tokens and len(self.can_run_list) != 0:
            return False

        with self._lock_node(req.last_node):
            if total_tokens > self.rem_total_tokens:
                return False

            if (
                self.rem_chunk_tokens is None
                or input_tokens <= self.rem_chunk_tokens
                or (req.return_logprob and req.normalized_prompt_logprob is None)
            ):
                # Non-chunked prefill
                self.can_run_list.append(req)
                self.tree_cache.inc_lock_ref(req.last_node)
                self._prefill_one_req(
                    prefix_len, input_tokens, req.sampling_params.max_new_tokens
                )
            else:
                # Chunked prefill
                trunc_len = self.rem_chunk_tokens
                if trunc_len == 0:
                    return False

                req.extend_input_len = trunc_len
                req.input_ids = req.input_ids[: len(req.prefix_indices) + trunc_len]
                self.can_run_list.append(req)
                self.new_inflight_req = req
                self.tree_cache.inc_lock_ref(req.last_node)
                self._prefill_one_req(prefix_len, trunc_len, 0)

        return True
