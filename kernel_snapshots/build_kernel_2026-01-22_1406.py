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
        assert (
            batch_size % VLEN == 0
        ), "This optimized kernel expects batch_size to be divisible by VLEN"

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

        chunk_count = batch_size // VLEN

        # Scalar constants
        forest_values_p_val = 7
        inp_indices_p_val = forest_values_p_val + n_nodes
        inp_values_p_val = inp_indices_p_val + batch_size

        zero = const(0, "zero")
        one = const(1, "one")
        two = const(2, "two")
        three = const(3, "three")
        n_nodes_c = const(n_nodes, "n_nodes")
        forest_values_p = const(forest_values_p_val, "forest_values_p")
        inp_indices_p = const(inp_indices_p_val, "inp_indices_p")
        inp_values_p = const(inp_values_p_val, "inp_values_p")

        # Vector constants
        zero_vec = self.alloc_scratch("zero_vec", VLEN)
        one_vec = self.alloc_scratch("one_vec", VLEN)
        two_vec = self.alloc_scratch("two_vec", VLEN)
        three_vec = self.alloc_scratch("three_vec", VLEN)
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

        # Scratch buffers for the working set
        idx_buf = self.alloc_scratch("idx_buf", batch_size)
        val_buf = self.alloc_scratch("val_buf", batch_size)

        # Per-chunk constants for input pointers
        idx_ptrs = []
        val_ptrs = []
        for ci in range(chunk_count):
            offset = ci * VLEN
            idx_ptrs.append(const(inp_indices_p_val + offset))
            val_ptrs.append(const(inp_values_p_val + offset))

        group_size = chunk_count
        addr_vecs = [self.alloc_scratch(f"addr_vec_{i}", VLEN) for i in range(group_size)]
        node_vecs = [self.alloc_scratch(f"node_vec_{i}", VLEN) for i in range(group_size)]

        # Load inputs into scratch buffers once
        for ci in range(chunk_count):
            offset = ci * VLEN
            emit(
                {
                    "load": [
                        ("vload", idx_buf + offset, idx_ptrs[ci]),
                        ("vload", val_buf + offset, val_ptrs[ci]),
                    ]
                }
            )

        def build_ops(idx_chunk, val_chunk, regs, idx_ptr, val_ptr):
            addr_vec, node_vec = regs
            tmp1_vec = node_vec
            tmp2_vec = addr_vec
            cond_vec = node_vec
            ops = []
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
                    ops.append(("valu", [("-", idx_chunk, idx_chunk, one_vec)], "valu"))
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
                    ops.append(("valu", [("+", idx_chunk, idx_chunk, three_vec)], "valu"))
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
                        ("vstore", idx_ptr, idx_chunk),
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

        def schedule_wave(queues):
            start = 0
            while any(q.ready() for q in queues):
                load_supply = sum(q.load_block_len() for q in queues)
                need_addr = load_supply <= 6
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
                for off in range(len(queues)):
                    if len(slots) == limit:
                        break
                    gi = (start + off) % len(queues)
                    if gi in used:
                        continue
                    q = queues[gi]
                    if not q.ready():
                        continue
                    eng, op_slots, _ = q.peek()
                    if eng != "valu":
                        continue
                    if len(slots) + len(op_slots) > limit:
                        continue
                    slots.extend(op_slots)
                    q.pop()
                    used.add(gi)
                if slots:
                    instr["valu"] = slots

                # Schedule one load op if available.
                slots = []
                limit = SLOT_LIMITS["load"]
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
                    if len(op_slots) > limit:
                        continue
                    slots.extend(op_slots)
                    q.pop()
                    used.add(gi)
                    break
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

        for base_chunk in range(0, chunk_count, group_size):
            active = min(group_size, chunk_count - base_chunk)
            idx_chunks = [idx_buf + (base_chunk + i) * VLEN for i in range(active)]
            val_chunks = [val_buf + (base_chunk + i) * VLEN for i in range(active)]
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
                            idx_ptrs[base_chunk + gi],
                            val_ptrs[base_chunk + gi],
                        )
                    )
                )
            schedule_wave(queues)

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
