"""
# Anthropic's Original Performance Engineering Take-home (Release version)

Copyright Anthropic PBC 2026. Permission is granted to modify and use, but not
to publish or redistribute your solutions so it's hard to find spoilers.

# Task

- Optimize the kernel (in KernelBuilder.build_kernel) as much as possible in the
  available time, as measured by test_kernel_cycles on a frozen separate copy
  of the simulator.

Validate your results using `python tests/submission_tests.py` without modifying
anything in the tests/ folder.

We recommend you look through problem.py next.
"""

from collections import defaultdict
import random
import unittest

from problem import (
    Engine,
    DebugInfo,
    SLOT_LIMITS,
    VLEN,
    N_CORES,
    SCRATCH_SIZE,
    Machine,
    Tree,
    Input,
    HASH_STAGES,
    reference_kernel,
    build_mem_image,
    reference_kernel2,
)


class KernelBuilder:
    def __init__(self):
        self.instrs = []
        self.scratch = {}
        self.scratch_debug = {}
        self.scratch_ptr = 0
        self.const_map = {}

    def debug_info(self):
        return DebugInfo(scratch_map=self.scratch_debug)

    def build(self, slots: list[tuple[Engine, tuple]], vliw: bool = False):
        # Simple slot packing that just uses one slot per instruction bundle
        instrs = []
        for engine, slot in slots:
            instrs.append({engine: [slot]})
        return instrs

    def add(self, engine, slot):
        self.instrs.append({engine: [slot]})

    def alloc_scratch(self, name=None, length=1):
        addr = self.scratch_ptr
        if name is not None:
            self.scratch[name] = addr
            self.scratch_debug[addr] = (name, length)
        self.scratch_ptr += length
        assert self.scratch_ptr <= SCRATCH_SIZE, "Out of scratch space"
        return addr

    def scratch_const(self, val, name=None):
        if val not in self.const_map:
            addr = self.alloc_scratch(name)
            self.add("load", ("const", addr, val))
            self.const_map[val] = addr
        return self.const_map[val]

    def build_hash(self, val_hash_addr, tmp1, tmp2, round, i):
        slots = []

        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            slots.append(("alu", (op1, tmp1, val_hash_addr, self.scratch_const(val1))))
            slots.append(("alu", (op3, tmp2, val_hash_addr, self.scratch_const(val3))))
            slots.append(("alu", (op2, val_hash_addr, tmp1, tmp2)))
            slots.append(("debug", ("compare", val_hash_addr, (round, i, "hash_stage", hi))))

        return slots

    def build_kernel(
        self, forest_height: int, n_nodes: int, batch_size: int, rounds: int
    ):
        """
        Optimized kernel:
        - Cache the entire indices/values arrays in scratch so we only touch
          main memory at the start/end.
        - Vectorize all arithmetic over VLEN lanes.
        - Avoid flow engine entirely by using arithmetic equivalents.
        - Bundle ops across chunks to overlap load and valu work.
        """
        instrs = self.instrs

        def emit(instr):
            instrs.append(instr)

        const_map = {}
        const_loads = []

        def const(val, name=None):
            if val not in const_map:
                addr = self.alloc_scratch(name)
                const_map[val] = addr
                const_loads.append((addr, val))
            return const_map[val]

        full_chunks = batch_size // VLEN
        tail = batch_size % VLEN
        padded_batch = full_chunks * VLEN
        chunk_count = full_chunks

        # Scalar constants
        forest_values_p_val = 7
        inp_indices_p_val = forest_values_p_val + n_nodes
        inp_values_p_val = inp_indices_p_val + batch_size

        zero = const(0, "zero")
        one = const(1, "one")
        two = const(2, "two")
        three = const(3, "three")
        four = const(4, "four")
        n_nodes_c = const(n_nodes, "n_nodes")
        forest_values_p = const(forest_values_p_val, "forest_values_p")

        # Vector constants
        zero_vec = self.alloc_scratch("zero_vec", VLEN)
        one_vec = self.alloc_scratch("one_vec", VLEN)
        two_vec = self.alloc_scratch("two_vec", VLEN)
        three_vec = self.alloc_scratch("three_vec", VLEN)
        four_vec = self.alloc_scratch("four_vec", VLEN)
        n_nodes_vec = self.alloc_scratch("n_nodes_vec", VLEN)
        forest_vec = self.alloc_scratch("forest_vec", VLEN)
        forest0_scalar = self.alloc_scratch("forest0_scalar")
        forest0_vec = self.alloc_scratch("forest0_vec", VLEN)
        forest1_scalar = self.alloc_scratch("forest1_scalar")
        forest2_scalar = self.alloc_scratch("forest2_scalar")
        forest1_vec = self.alloc_scratch("forest1_vec", VLEN)
        forest2_vec = self.alloc_scratch("forest2_vec", VLEN)
        forest_diff_vec = self.alloc_scratch("forest_diff_vec", VLEN)
        use_level2_const = forest_height >= 2 and rounds > 2
        if use_level2_const:
            forest3_scalar = self.alloc_scratch("forest3_scalar")
            forest4_scalar = self.alloc_scratch("forest4_scalar")
            forest5_scalar = self.alloc_scratch("forest5_scalar")
            forest6_scalar = self.alloc_scratch("forest6_scalar")
            forest3_vec = self.alloc_scratch("forest3_vec", VLEN)
            forest4_vec = self.alloc_scratch("forest4_vec", VLEN)
            forest5_vec = self.alloc_scratch("forest5_vec", VLEN)
            forest6_vec = self.alloc_scratch("forest6_vec", VLEN)

        def emit_vbcasts(pairs):
            for i in range(0, len(pairs), SLOT_LIMITS["valu"]):
                slots = [
                    ("vbroadcast", dest, src)
                    for dest, src in pairs[i : i + SLOT_LIMITS["valu"]]
                ]
                emit({"valu": slots})

        emit_vbcasts(
            [
                (zero_vec, zero),
                (one_vec, one),
                (two_vec, two),
                (three_vec, three),
                (four_vec, four),
                (n_nodes_vec, n_nodes_c),
                (forest_vec, forest_values_p),
            ]
        )
        forest1_addr = const(forest_values_p_val + 1)
        forest2_addr = const(forest_values_p_val + 2)
        if use_level2_const:
            forest3_addr = const(forest_values_p_val + 3)
            forest4_addr = const(forest_values_p_val + 4)
            forest5_addr = const(forest_values_p_val + 5)
            forest6_addr = const(forest_values_p_val + 6)
        emit(
            {
                "load": [
                    ("load", forest0_scalar, forest_values_p),
                    ("load", forest1_scalar, forest1_addr),
                ]
            }
        )
        emit({"load": [("load", forest2_scalar, forest2_addr)]})
        if use_level2_const:
            emit(
                {
                    "load": [
                        ("load", forest3_scalar, forest3_addr),
                        ("load", forest4_scalar, forest4_addr),
                    ]
                }
            )
            emit(
                {
                    "load": [
                        ("load", forest5_scalar, forest5_addr),
                        ("load", forest6_scalar, forest6_addr),
                    ]
                }
            )
        emit_vbcasts(
            [
                (forest0_vec, forest0_scalar),
                (forest1_vec, forest1_scalar),
                (forest2_vec, forest2_scalar),
            ]
        )
        if use_level2_const:
            emit_vbcasts(
                [
                    (forest3_vec, forest3_scalar),
                    (forest4_vec, forest4_scalar),
                    (forest5_vec, forest5_scalar),
                    (forest6_vec, forest6_scalar),
                ]
            )

        emit({"valu": [("-", forest_diff_vec, forest2_vec, forest1_vec)]})

        hash_plan = []
        hash_bcasts = []
        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            hv1 = self.alloc_scratch(f"hash{hi}_a", VLEN)
            hash_bcasts.append((hv1, const(val1)))
            if op1 == "+" and op2 == "+" and op3 == "<<":
                mul_val = (1 << val3) + 1
                mul_vec = self.alloc_scratch(f"hash{hi}_mul", VLEN)
                hash_bcasts.append((mul_vec, const(mul_val)))
                hash_plan.append(("muladd", hv1, mul_vec))
            else:
                hv3 = self.alloc_scratch(f"hash{hi}_b", VLEN)
                hash_bcasts.append((hv3, const(val3)))
                hash_plan.append((op1, hv1, op2, op3, hv3))
        emit_vbcasts(hash_bcasts)

        expected_nodes = (1 << (forest_height + 1)) - 1
        use_round_wrap = n_nodes == expected_nodes
        wrap_period = forest_height + 1

        tail_extra_consts = 0
        if tail:
            for _, val1, _, _, val3 in HASH_STAGES:
                if val1 not in const_map:
                    tail_extra_consts += 1
                if val3 not in const_map:
                    tail_extra_consts += 1
        tail_scratch = 0 if tail == 0 else (7 + tail + tail_extra_consts)
        scratch_limit = SCRATCH_SIZE - tail_scratch
        base_scratch = self.scratch_ptr
        full_fixed = base_scratch + 2 * padded_batch + chunk_count
        full_remaining = scratch_limit - full_fixed
        max_group_full = full_remaining // (2 * VLEN) if full_remaining >= 0 else 0
        use_full_buf = max_group_full >= 1

        if use_full_buf:
            group_size = min(chunk_count, max_group_full)
            idx_buf = self.alloc_scratch("idx_buf", padded_batch)
            val_buf = self.alloc_scratch("val_buf", padded_batch)
        else:
            stream_remaining = scratch_limit - (base_scratch + chunk_count)
            max_group_stream = (
                stream_remaining // (4 * VLEN) if stream_remaining >= 0 else 0
            )
            group_size = min(chunk_count, max_group_stream) if max_group_stream > 0 else 0
            if chunk_count and group_size < 1:
                raise AssertionError("Batch size too large for scratch buffers")
            idx_buf = self.alloc_scratch("idx_buf", group_size * VLEN)
            val_buf = self.alloc_scratch("val_buf", group_size * VLEN)

        # Per-chunk constants for input pointers
        val_ptrs = []
        for ci in range(chunk_count):
            offset = ci * VLEN
            val_ptrs.append(const(inp_values_p_val + offset))
        addr_vecs = [self.alloc_scratch(f"addr_vec_{i}", VLEN) for i in range(group_size)]
        node_vecs = [self.alloc_scratch(f"node_vec_{i}", VLEN) for i in range(group_size)]

        def build_ops(idx_chunk, val_chunk, regs, val_ptr):
            addr_vec, node_vec = regs
            tmp1_vec = node_vec
            tmp2_vec = addr_vec
            cond_vec = node_vec
            ops = []
            ops.append(("load", [("vload", val_chunk, val_ptr)], "load"))
            for round_i in range(rounds):
                depth = round_i
                if use_round_wrap:
                    depth = round_i % wrap_period
                if depth == 0:
                    ops.append(
                        ("valu", [("^", val_chunk, val_chunk, forest0_vec)], "valu")
                    )
                elif depth == 1 and rounds > 1:
                    ops.append(("valu", [("^", val_chunk, val_chunk, addr_vec)], "valu"))
                elif use_level2_const and depth == 2 and rounds > 2:
                    ops.append(("valu", [("^", val_chunk, val_chunk, addr_vec)], "valu"))
                else:
                    for start in range(0, VLEN, 2):
                        ops.append(
                            (
                                "load",
                                [
                                    ("load_offset", node_vec, addr_vec, start),
                                    ("load_offset", node_vec, addr_vec, start + 1),
                                ],
                                "load",
                            )
                        )
                    ops.append(("valu", [("^", val_chunk, val_chunk, node_vec)], "valu"))
                for stage in hash_plan:
                    if stage[0] == "muladd":
                        _, hv1, mul_vec = stage
                        ops.append(
                            (
                                "valu",
                                [("multiply_add", val_chunk, val_chunk, mul_vec, hv1)],
                                "valu",
                            )
                        )
                    else:
                        op1, hv1, op2, op3, hv3 = stage
                        ops.append(
                            (
                                "valu",
                                [
                                    (op1, tmp1_vec, val_chunk, hv1),
                                    (op3, tmp2_vec, val_chunk, hv3),
                                ],
                                "valu",
                            )
                        )
                        ops.append(
                            (
                                "valu",
                                [(op2, val_chunk, tmp1_vec, tmp2_vec)],
                                "valu",
                            )
                        )
                if use_level2_const and depth == 1 and rounds > 2:
                    ops.append(
                        (
                            "valu",
                            [
                                ("%", cond_vec, val_chunk, two_vec),
                                ("-", idx_chunk, idx_chunk, one_vec),
                            ],
                            "valu",
                        )
                    )
                else:
                    ops.append(("valu", [("%", cond_vec, val_chunk, two_vec)], "valu"))
                if depth == 0 and rounds > 1:
                    ops.append(
                        (
                            "valu",
                            [
                                (
                                    "multiply_add",
                                    addr_vec,
                                    cond_vec,
                                    forest_diff_vec,
                                    forest1_vec,
                                )
                            ],
                            "valu",
                        )
                    )
                if use_level2_const and depth == 1 and rounds > 2:
                    ops.append(
                        (
                            "flow",
                            [("vselect", addr_vec, cond_vec, forest6_vec, forest5_vec)],
                            "flow",
                        )
                    )
                    ops.append(
                        (
                            "flow",
                            [("vselect", node_vec, cond_vec, forest4_vec, forest3_vec)],
                            "flow",
                        )
                    )
                    ops.append(
                        (
                            "flow",
                            [("vselect", addr_vec, idx_chunk, addr_vec, node_vec)],
                            "flow",
                        )
                    )
                    ops.append(("valu", [("%", cond_vec, val_chunk, two_vec)], "valu"))
                    ops.append(
                        ("flow", [("vselect", cond_vec, cond_vec, four_vec, three_vec)], "flow")
                    )
                    ops.append(
                        (
                            "valu",
                            [
                                (
                                    "multiply_add",
                                    idx_chunk,
                                    idx_chunk,
                                    two_vec,
                                    cond_vec,
                                )
                            ],
                            "valu",
                        )
                    )
                else:
                    if depth == 0:
                        ops.append(
                            ("flow", [("vselect", idx_chunk, cond_vec, two_vec, one_vec)], "flow")
                        )
                    else:
                        ops.append(
                            ("flow", [("vselect", cond_vec, cond_vec, two_vec, one_vec)], "flow")
                        )
                        ops.append(
                            (
                                "valu",
                                [
                                    (
                                        "multiply_add",
                                        idx_chunk,
                                        idx_chunk,
                                        two_vec,
                                        cond_vec,
                                    )
                                ],
                                "valu",
                            )
                        )
                if use_round_wrap:
                    if (round_i + 1) % wrap_period == 0:
                        ops.append(
                            (
                                "flow",
                                [("vselect", idx_chunk, zero_vec, idx_chunk, zero_vec)],
                                "flow",
                            )
                        )
                else:
                    ops.append(
                        ("valu", [("<", cond_vec, idx_chunk, n_nodes_vec)], "valu")
                    )
                    ops.append(("valu", [("*", idx_chunk, idx_chunk, cond_vec)], "valu"))
                if round_i != rounds - 1 and round_i > 0:
                    if depth == 0:
                        continue
                    if use_level2_const and depth == 1 and rounds > 2:
                        continue
                    ops.append(("valu", [("+", addr_vec, idx_chunk, forest_vec)], "addr"))
            ops.append(
                (
                    "store",
                    [
                        ("vstore", val_ptr, val_chunk),
                    ],
                    "store",
                )
            )
            return ops

        class OpQueue:
            def __init__(self, ops):
                self.ops = ops
                self.idx = 0

            def ready(self):
                return self.idx < len(self.ops)

            def peek(self):
                return self.ops[self.idx]

            def pop(self):
                self.idx += 1

            def load_block_len(self):
                if not self.ready():
                    return 0
                if self.ops[self.idx][0] != "load":
                    return 0
                n = 0
                for i in range(self.idx, len(self.ops)):
                    if self.ops[i][0] != "load":
                        break
                    n += 1
                return n
            def load_distance(self):
                if not self.ready():
                    return 1 << 30
                for i in range(self.idx, len(self.ops)):
                    if self.ops[i][0] == "load":
                        return i - self.idx
                return 1 << 30

        def schedule_wave(queues):
            start = 0
            while any(q.ready() for q in queues):
                load_supply = sum(q.load_block_len() for q in queues)
                need_addr = load_supply <= 10
                if load_supply <= 2:
                    addr_budget = 3
                elif load_supply <= 4:
                    addr_budget = 2
                else:
                    addr_budget = 1
                instr = {}
                used = set()
                # Schedule valu ops, prioritizing addr ops when load supply is low.
                slots = []
                limit = SLOT_LIMITS["valu"]
                if need_addr:
                    addr_used = 0
                    for off in range(len(queues)):
                        if len(slots) == limit:
                            break
                        if addr_used >= addr_budget:
                            break
                        gi = (start + off) % len(queues)
                        if gi in used:
                            continue
                        q = queues[gi]
                        if not q.ready():
                            continue
                        eng, op_slots, kind = q.peek()
                        if eng != "valu" or kind != "addr":
                            continue
                        if len(slots) + len(op_slots) > limit:
                            continue
                        slots.extend(op_slots)
                        q.pop()
                        used.add(gi)
                        addr_used += 1
                candidates = []
                for off in range(len(queues)):
                    gi = (start + off) % len(queues)
                    if gi in used:
                        continue
                    q = queues[gi]
                    if not q.ready():
                        continue
                    eng, op_slots, _ = q.peek()
                    if eng != "valu":
                        continue
                    candidates.append((q.load_distance(), off, gi, op_slots))
                for _, _, gi, op_slots in sorted(
                    candidates, key=lambda item: (item[0], len(item[3]), item[1])
                ):
                    if len(slots) == limit:
                        break
                    if gi in used:
                        continue
                    if len(slots) + len(op_slots) > limit:
                        continue
                    slots.extend(op_slots)
                    queues[gi].pop()
                    used.add(gi)
                if slots:
                    instr["valu"] = slots

                # Schedule load ops up to the slot limit; bias toward finishing load blocks.
                slots = []
                limit = SLOT_LIMITS["load"]
                candidates = []
                for off in range(len(queues)):
                    gi = (start + off) % len(queues)
                    if gi in used:
                        continue
                    q = queues[gi]
                    if not q.ready():
                        continue
                    eng, op_slots, _ = q.peek()
                    if eng != "load":
                        continue
                    candidates.append((q.load_block_len(), off, gi))
                for _, _, gi in sorted(candidates):
                    if len(slots) == limit:
                        break
                    q = queues[gi]
                    eng, op_slots, _ = q.peek()
                    if eng != "load":
                        continue
                    if len(slots) + len(op_slots) > limit:
                        continue
                    slots.extend(op_slots)
                    q.pop()
                    used.add(gi)
                if slots:
                    instr["load"] = slots

                # Schedule one store op if available.
                slots = []
                limit = SLOT_LIMITS["store"]
                for off in range(len(queues)):
                    gi = (start + off) % len(queues)
                    if gi in used:
                        continue
                    q = queues[gi]
                    if not q.ready():
                        continue
                    eng, op_slots, _ = q.peek()
                    if eng != "store":
                        continue
                    if len(op_slots) > limit:
                        continue
                    slots.extend(op_slots)
                    q.pop()
                    used.add(gi)
                    break
                if slots:
                    instr["store"] = slots
                # Schedule one flow op if available.
                slots = []
                limit = SLOT_LIMITS["flow"]
                for off in range(len(queues)):
                    gi = (start + off) % len(queues)
                    if gi in used:
                        continue
                    q = queues[gi]
                    if not q.ready():
                        continue
                    eng, op_slots, _ = q.peek()
                    if eng != "flow":
                        continue
                    if len(op_slots) > limit:
                        continue
                    slots.extend(op_slots)
                    q.pop()
                    used.add(gi)
                    break
                if slots:
                    instr["flow"] = slots
                if not instr:
                    for q in queues:
                        if q.ready():
                            eng, op_slots, _ = q.peek()
                            instr[eng] = op_slots
                            q.pop()
                            break
                emit(instr)
                start = (start + 1) % len(queues)

        group_step = max(1, group_size)
        for base_chunk in range(0, chunk_count, group_step):
            active = min(group_size, chunk_count - base_chunk)
            chunk_base = base_chunk if use_full_buf else 0
            idx_chunks = [idx_buf + (chunk_base + i) * VLEN for i in range(active)]
            val_chunks = [val_buf + (chunk_base + i) * VLEN for i in range(active)]
            queues = []
            for gi in range(active):
                regs = (
                    addr_vecs[gi],
                    node_vecs[gi],
                )
                queues.append(
                    OpQueue(
                        build_ops(
                            idx_chunks[gi],
                            val_chunks[gi],
                            regs,
                            val_ptrs[base_chunk + gi],
                        )
                    )
                )
            schedule_wave(queues)

        if tail:
            tail_val = self.alloc_scratch("tail_val")
            tail_idx = self.alloc_scratch("tail_idx")
            tail_tmp1 = self.alloc_scratch("tail_tmp1")
            tail_tmp2 = self.alloc_scratch("tail_tmp2")
            tail_cond = self.alloc_scratch("tail_cond")
            tail_addr = self.alloc_scratch("tail_addr")
            tail_node = self.alloc_scratch("tail_node")
            tail_base = padded_batch
            for ti in range(tail):
                val_ptr = const(inp_values_p_val + tail_base + ti)
                emit({"load": [("load", tail_val, val_ptr)]})
                emit({"alu": [("+", tail_idx, zero, zero)]})
                for _ in range(rounds):
                    emit({"alu": [("+", tail_addr, forest_values_p, tail_idx)]})
                    emit({"load": [("load", tail_node, tail_addr)]})
                    emit({"alu": [("^", tail_val, tail_val, tail_node)]})
                    for op1, val1, op2, op3, val3 in HASH_STAGES:
                        emit(
                            {
                                "alu": [
                                    (op1, tail_tmp1, tail_val, const(val1)),
                                    (op3, tail_tmp2, tail_val, const(val3)),
                                ]
                            }
                        )
                        emit({"alu": [(op2, tail_val, tail_tmp1, tail_tmp2)]})
                    emit({"alu": [("%", tail_cond, tail_val, two)]})
                    emit({"flow": [("select", tail_cond, tail_cond, two, one)]})
                    emit({"alu": [("*", tail_idx, tail_idx, two)]})
                    emit({"alu": [("+", tail_idx, tail_idx, tail_cond)]})
                    emit({"alu": [("<", tail_cond, tail_idx, n_nodes_c)]})
                    emit({"alu": [("*", tail_idx, tail_idx, tail_cond)]})
                emit({"store": [("store", val_ptr, tail_val)]})

        if const_loads:
            const_instrs = []
            slots = []
            for addr, val in const_loads:
                slots.append(("const", addr, val))
                if len(slots) == SLOT_LIMITS["load"]:
                    const_instrs.append({"load": slots})
                    slots = []
            if slots:
                const_instrs.append({"load": slots})
            self.instrs = const_instrs + self.instrs

BASELINE = 147734

def do_kernel_test(
    forest_height: int,
    rounds: int,
    batch_size: int,
    seed: int = 123,
    trace: bool = False,
    prints: bool = False,
):
    print(f"{forest_height=}, {rounds=}, {batch_size=}")
    random.seed(seed)
    forest = Tree.generate(forest_height)
    inp = Input.generate(forest, batch_size, rounds)
    mem = build_mem_image(forest, inp)

    kb = KernelBuilder()
    kb.build_kernel(forest.height, len(forest.values), len(inp.indices), rounds)
    # print(kb.instrs)

    value_trace = {}
    machine = Machine(
        mem,
        kb.instrs,
        kb.debug_info(),
        n_cores=N_CORES,
        value_trace=value_trace,
        trace=trace,
    )
    machine.prints = prints
    for i, ref_mem in enumerate(reference_kernel2(mem, value_trace)):
        machine.run()
        inp_values_p = ref_mem[6]
        if prints:
            print(machine.mem[inp_values_p : inp_values_p + len(inp.values)])
            print(ref_mem[inp_values_p : inp_values_p + len(inp.values)])
        assert (
            machine.mem[inp_values_p : inp_values_p + len(inp.values)]
            == ref_mem[inp_values_p : inp_values_p + len(inp.values)]
        ), f"Incorrect result on round {i}"
        inp_indices_p = ref_mem[5]
        if prints:
            print(machine.mem[inp_indices_p : inp_indices_p + len(inp.indices)])
            print(ref_mem[inp_indices_p : inp_indices_p + len(inp.indices)])
        # Updating these in memory isn't required, but you can enable this check for debugging
        # assert machine.mem[inp_indices_p:inp_indices_p+len(inp.indices)] == ref_mem[inp_indices_p:inp_indices_p+len(inp.indices)]

    print("CYCLES: ", machine.cycle)
    print("Speedup over baseline: ", BASELINE / machine.cycle)
    return machine.cycle


class Tests(unittest.TestCase):
    def test_ref_kernels(self):
        """
        Test the reference kernels against each other
        """
        random.seed(123)
        for i in range(10):
            f = Tree.generate(4)
            inp = Input.generate(f, 10, 6)
            mem = build_mem_image(f, inp)
            reference_kernel(f, inp)
            for _ in reference_kernel2(mem, {}):
                pass
            assert inp.indices == mem[mem[5] : mem[5] + len(inp.indices)]
            assert inp.values == mem[mem[6] : mem[6] + len(inp.values)]

    def test_kernel_trace(self):
        # Full-scale example for performance testing
        do_kernel_test(10, 16, 256, trace=True, prints=False)

    # Passing this test is not required for submission, see submission_tests.py for the actual correctness test
    # You can uncomment this if you think it might help you debug
    # def test_kernel_correctness(self):
    #     for batch in range(1, 3):
    #         for forest_height in range(3):
    #             do_kernel_test(
    #                 forest_height + 2, forest_height + 4, batch * 16 * VLEN * N_CORES
    #             )

    def test_kernel_cycles(self):
        do_kernel_test(10, 16, 256)


# To run all the tests:
#    python perf_takehome.py
# To run a specific test:
#    python perf_takehome.py Tests.test_kernel_cycles
# To view a hot-reloading trace of all the instructions:  **Recommended debug loop**
# NOTE: The trace hot-reloading only works in Chrome. In the worst case if things aren't working, drag trace.json onto https://ui.perfetto.dev/
#    python perf_takehome.py Tests.test_kernel_trace
# Then run `python watch_trace.py` in another tab, it'll open a browser tab, then click "Open Perfetto"
# You can then keep that open and re-run the test to see a new trace.

# To run the proper checks to see which thresholds you pass:
#    python tests/submission_tests.py

if __name__ == "__main__":
    unittest.main()
