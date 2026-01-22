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
    Instruction,
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

    def emit(self, instr: Instruction):
        """
        Append a full instruction bundle. Each engine entry should contain
        a list of slots already respecting SLOT_LIMITS.
        """
        self.instrs.append(instr)

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
        assert (
            batch_size % VLEN == 0
        ), "This optimized kernel expects batch_size to be divisible by VLEN"

        chunk_count = batch_size // VLEN

        # Scalar constants
        forest_values_p_val = 7
        inp_indices_p_val = forest_values_p_val + n_nodes
        inp_values_p_val = inp_indices_p_val + batch_size

        zero = self.scratch_const(0, "zero")
        one = self.scratch_const(1, "one")
        two = self.scratch_const(2, "two")
        n_nodes_c = self.scratch_const(n_nodes, "n_nodes")
        forest_values_p = self.scratch_const(forest_values_p_val, "forest_values_p")
        inp_indices_p = self.scratch_const(inp_indices_p_val, "inp_indices_p")
        inp_values_p = self.scratch_const(inp_values_p_val, "inp_values_p")

        # Vector constants
        zero_vec = self.alloc_scratch("zero_vec", VLEN)
        one_vec = self.alloc_scratch("one_vec", VLEN)
        two_vec = self.alloc_scratch("two_vec", VLEN)
        n_nodes_vec = self.alloc_scratch("n_nodes_vec", VLEN)
        forest_vec = self.alloc_scratch("forest_vec", VLEN)

        for dest, src in [
            (zero_vec, zero),
            (one_vec, one),
            (two_vec, two),
            (n_nodes_vec, n_nodes_c),
            (forest_vec, forest_values_p),
        ]:
            self.emit({"valu": [("vbroadcast", dest, src)]})

        hash_plan = []
        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            hv1 = self.alloc_scratch(f"hash{hi}_a", VLEN)
            self.emit({"valu": [("vbroadcast", hv1, self.scratch_const(val1))]})
            if op1 == "+" and op2 == "+" and op3 == "<<":
                mul_val = (1 << val3) + 1
                mul_vec = self.alloc_scratch(f"hash{hi}_mul", VLEN)
                self.emit({"valu": [("vbroadcast", mul_vec, self.scratch_const(mul_val))]})
                hash_plan.append(("muladd", hv1, mul_vec))
            else:
                hv3 = self.alloc_scratch(f"hash{hi}_b", VLEN)
                self.emit({"valu": [("vbroadcast", hv3, self.scratch_const(val3))]})
                hash_plan.append((op1, hv1, op2, op3, hv3))

        # Scratch buffers for the working set
        idx_buf = self.alloc_scratch("idx_buf", batch_size)
        val_buf = self.alloc_scratch("val_buf", batch_size)

        # Per-chunk constants for input pointers
        idx_ptrs = []
        val_ptrs = []
        for ci in range(chunk_count):
            offset = ci * VLEN
            idx_ptrs.append(self.scratch_const(inp_indices_p_val + offset))
            val_ptrs.append(self.scratch_const(inp_values_p_val + offset))

        group_size = min(chunk_count, SLOT_LIMITS["valu"] * 2 + 4)
        addr_vecs = [self.alloc_scratch(f"addr_vec_{i}", VLEN) for i in range(group_size)]
        node_vecs = [self.alloc_scratch(f"node_vec_{i}", VLEN) for i in range(group_size)]
        tmp1_vecs = [self.alloc_scratch(f"tmp1_vec_{i}", VLEN) for i in range(group_size)]
        tmp2_vecs = [self.alloc_scratch(f"tmp2_vec_{i}", VLEN) for i in range(group_size)]
        cond_vecs = [self.alloc_scratch(f"cond_vec_{i}", VLEN) for i in range(group_size)]

        # Load inputs into scratch buffers once
        for ci in range(chunk_count):
            offset = ci * VLEN
            self.emit(
                {
                    "load": [
                        ("vload", idx_buf + offset, idx_ptrs[ci]),
                        ("vload", val_buf + offset, val_ptrs[ci]),
                    ]
                }
            )

        def build_ops(idx_chunk, val_chunk, regs):
            addr_vec, node_vec, tmp1_vec, tmp2_vec, cond_vec = regs
            ops = []
            for _ in range(rounds):
                ops.append(("valu", [("+", addr_vec, idx_chunk, forest_vec)]))
                for start in range(0, VLEN, 2):
                    ops.append(
                        (
                            "load",
                            [
                                ("load_offset", node_vec, addr_vec, start),
                                ("load_offset", node_vec, addr_vec, start + 1),
                            ],
                        )
                    )
                ops.append(("valu", [("^", val_chunk, val_chunk, node_vec)]))
                for stage in hash_plan:
                    if stage[0] == "muladd":
                        _, hv1, mul_vec = stage
                        ops.append(
                            (
                                "valu",
                                [("multiply_add", val_chunk, val_chunk, mul_vec, hv1)],
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
                            )
                        )
                        ops.append(
                            (
                                "valu",
                                [(op2, val_chunk, tmp1_vec, tmp2_vec)],
                            )
                        )
                ops.append(("valu", [("%", cond_vec, val_chunk, two_vec)]))
                ops.append(("valu", [("+", cond_vec, cond_vec, one_vec)]))
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
                    )
                )
                ops.append(("valu", [("<", cond_vec, idx_chunk, n_nodes_vec)]))
                ops.append(("valu", [("*", idx_chunk, idx_chunk, cond_vec)]))
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

        def schedule_wave(queues):
            start = 0
            while any(q.ready() for q in queues):
                instr = {}
                used = set()
                for engine in ("valu", "load"):
                    slots = []
                    limit = SLOT_LIMITS[engine]
                    for off in range(len(queues)):
                        gi = (start + off) % len(queues)
                        if gi in used:
                            continue
                        q = queues[gi]
                        if not q.ready():
                            continue
                        eng, op_slots = q.peek()
                        if eng != engine:
                            continue
                        if len(slots) + len(op_slots) > limit:
                            continue
                        slots.extend(op_slots)
                        q.pop()
                        used.add(gi)
                        if len(slots) == limit:
                            break
                    if slots:
                        instr[engine] = slots
                if not instr:
                    for q in queues:
                        if q.ready():
                            eng, op_slots = q.peek()
                            instr[eng] = op_slots
                            q.pop()
                            break
                self.instrs.append(instr)
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
                    tmp1_vecs[gi],
                    tmp2_vecs[gi],
                    cond_vecs[gi],
                )
                queues.append(OpQueue(build_ops(idx_chunks[gi], val_chunks[gi], regs)))
            schedule_wave(queues)

        # Write results back to memory
        for ci in range(chunk_count):
            offset = ci * VLEN
            self.emit(
                {
                    "store": [
                        ("vstore", idx_ptrs[ci], idx_buf + offset),
                        ("vstore", val_ptrs[ci], val_buf + offset),
                    ]
                }
            )

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
