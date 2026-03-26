from copy import copy
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal
import random

Engine = Literal["alu", "load", "store", "flow"]
Instruction = dict[Engine, list[tuple]]


class CoreState(Enum):
    RUNNING = 1
    PAUSED = 2
    STOPPED = 3


@dataclass
class Core:
    id: int
    pc: int = 0
    state: CoreState = CoreState.RUNNING
    scratch: list[int] = field(default_factory=lambda: [0] * SCRATCH_SIZE)
    trace_buf: list[int] = field(default_factory=list)


N_CORES = 16
VLEN = 8
SCRATCH_SIZE = 1024
SLOT_LIMITS: dict[Engine, int] = {
    "alu": 12,
    "valu": 6,
    "load": 2,
    "store": 1,
    "flow": 1,
}
BASE_ADDR_TID = 1000


def cdiv(a, b):
    return (a + b - 1) // b


class Machine:
    """
    Simulator for a custom VLIW SIMD architecture.

    VLIW (Very Large Instruction Word): Cores are composed of different
    "engines" each of which can execute multiple "slots" per cycle in parallel.
    How many slots each engine can execute per cycle is limited by SLOT_LIMITS.
    Effects of instructions don't take effect until the end of cycle. Each
    cycle, all engines execute all of their filled slots for that instruction.
    Effects like writes to memory take place after all the inputs are read.

    SIMD: There are instructions for acting on vectors of VLEN elements in a
    single slot. You can use vload and vstore to load multiple contiguous
    elements but not non-contiguous elements. Use vbroadcast to broadcast a
    scalar to a vector and then operate on vectors with valu instructions.

    The memory and scratch space are composed of 32-bit words. The solution is
    plucked out of the memory at the end of the program. You can think of the
    scratch space as serving the purpose of registers, constant memory, and a
    manually-managed cache.

    Here's an example of what an instruction might look like:

    {"valu": [("*", 4, 0, 0), ("+", 8, 4, 0)], "load": [("load", 16, 17)]}

    In general every number in an instruction is a scratch address except for
    const and jump, and except for store and some flow instructions the first
    operand is the destination.

    This comment is not meant to be full ISA documentation though, for the rest
    you should look through the simulator code.
    """

    def __init__(
        self,
        mem: list[int],
        instrs: list[Instruction],
        debug_info,
        n_cores: int = N_CORES,
        trace=None,
        value_trace: dict[Any, int] = {},
    ):
        self.mem = mem
        self.instrs = instrs
        self.debug_info = debug_info
        self.cores = [Core(i) for i in range(n_cores)]
        self.cycle = 0
        self.enable_debug = True
        self.enable_pause = True
        self.prints = False
        self.value_trace = value_trace
        if trace:
            self.setup_trace()
            self.trace_enabled = True
        else:
            self.trace = None
            self.trace_enabled = False

    def rewrite_instr(self, instr):
        """
        Rewrite an instruction to use scratch addresses instead of names
        """
        res = {}
        for name, slots in instr.items():
            res[name] = []
            for slot in slots:
                res[name].append(self.rewrite_slot(slot))
        return res

    def print_step(self, instr, core):
        # print(core.id)
        # print(core.trace_buf)
        print(self.scratch_map(core))
        print(core.pc, instr, self.rewrite_instr(instr))

    def scratch_map(self, core):
        res = {}
        for addr, (name, length) in self.debug_info.scratch_map.items():
            res[name] = core.scratch[addr : addr + length]
        return res

    def rewrite_slot(self, slot):
        res = [slot[0]]
        for x in slot[1:]:
            if isinstance(x, int) and x in self.debug_info.scratch_map:
                res.append(self.debug_info.scratch_map[x][0])
            else:
                res.append(x)
        return tuple(res)

    def setup_trace(self):
        """
        The simulator generates traces in Chrome's Trace Event Format for
        visualization in Perfetto (or chrome://tracing if you prefer it). See
        the bottom of the file for info about how to use this.

        See the format docs in case you want to add more info to the trace:
        https://docs.google.com/document/d/1CvAClvFfyA5R-PhYUmn5OOQtYMH4h6I0nSsKchNAySU/preview
        """
        self.trace = open("trace.json", "w")
        self.trace.write("[")
        tid_counter = 0
        self.tids = {}
        for ci, core in enumerate(self.cores):
            for name, limit in SLOT_LIMITS.items():
                if name == "debug":
                    continue
                for i in range(limit):
                    self.trace.write(
                        f'{{"name": "{name}_{i}", "cat": "op", "ph": "M", "pid": {ci}, "tid": {tid_counter}, "ts": 0, "args": {{"name": "{name}_{i}"}} }},\n'
                    )
                    self.tids[(ci, name, i)] = tid_counter
                    tid_counter += 1

        # Add zero-length events at the start so all slots show up in Perfetto
        for ci, core in enumerate(self.cores):
            for name, limit in SLOT_LIMITS.items():
                if name == "debug":
                    continue
                for i in range(limit):
                    self.trace.write(
                        f'{{"name": "{name}_{i}", "cat": "op", "ph": "X", "pid": {ci}, "tid": {self.tids[(ci, name, i)]}, "ts": 0, "dur": 0 }},\n'
                    )

        for addr, (name, length) in self.debug_info.scratch_map.items():
            for i in range(length):
                self.trace.write(
                    f'{{"name": "{name}[{i}]", "cat": "op", "ph": "M", "pid": {len(self.cores) + ci}, "tid": {BASE_ADDR_TID + addr + i}, "ts": 0, "args": {{"name": "{name}[{i}]"}} }},\n'
                )

    def run(self):
        while any(core.state == CoreState.RUNNING for core in self.cores):
            batch_instrs = []
            for core in self.cores:
                if core.state == CoreState.RUNNING:
                    if core.pc >= len(self.instrs):
                        core.state = CoreState.STOPPED
                        batch_instrs.append(None)
                        continue
                    instr = self.instrs[core.pc]
                    batch_instrs.append(instr)
                    core.pc += 1
                else:
                    batch_instrs.append(None)

            if all(instr is None for instr in batch_instrs):
                break

            for i, instr in enumerate(batch_instrs):
                if instr is not None:
                    self.step(instr, self.cores[i])

            self.cycle += 1

        if self.trace:
            self.trace.write("{}]")
            self.trace.close()

    def alu(self, core, op, dest, a1, a2):
        a1 = core.scratch[a1]
        a2 = core.scratch[a2]
        if op == "+":
            res = a1 + a2
        elif op == "-":
            res = a1 - a2
        elif op == "*":
            res = a1 * a2
        elif op == "//":
            res = a1 // a2
        elif op == "cdiv":
            res = cdiv(a1, a2)
        elif op == "^":
            res = a1 ^ a2
        elif op == "&":
            res = a1 & a2
        elif op == "|":
            res = a1 | a2
        elif op == "<<":
            res = a1 << a2
        elif op == ">>":
            res = a1 >> a2
        elif op == "%":
            res = a1 % a2
        elif op == "<":
            res = int(a1 < a2)
        elif op == "==":
            res = int(a1 == a2)
        else:
            raise NotImplementedError(f"Unknown alu op {op}")
        res = res % (2**32)
        self.scratch_write[dest] = res

    def valu(self, core, *slot):
        op = slot[0]
        if op == "vbroadcast":
            dest, src = slot[1], slot[2]
            for i in range(VLEN):
                self.scratch_write[dest + i] = core.scratch[src]
        elif op == "multiply_add":
            dest, a, b, c = slot[1], slot[2], slot[3], slot[4]
            for i in range(VLEN):
                mul = (core.scratch[a + i] * core.scratch[b + i]) % (2**32)
                self.scratch_write[dest + i] = (mul + core.scratch[c + i]) % (2**32)
        elif len(slot) == 4:
            op, dest, a1, a2 = slot[0], slot[1], slot[2], slot[3]
            for i in range(VLEN):
                self.alu(core, op, dest + i, a1 + i, a2 + i)
        else:
            raise NotImplementedError(f"Unknown valu op {slot}")

    def load(self, core, *slot):
        op = slot[0]
        if op == "load":
            dest, addr = slot[1], slot[2]
            self.scratch_write[dest] = self.mem[core.scratch[addr]]
        elif op == "load_offset":
            dest, addr, offset = slot[1], slot[2], slot[3]
            self.scratch_write[dest + offset] = self.mem[core.scratch[addr + offset]]
        elif op == "vload":
            dest, addr = slot[1], slot[2]
            base_addr = core.scratch[addr]
            for vi in range(VLEN):
                self.scratch_write[dest + vi] = self.mem[base_addr + vi]
        elif op == "const":
            dest, val = slot[1], slot[2]
            self.scratch_write[dest] = (val) % (2**32)
        else:
            raise NotImplementedError(f"Unknown load op {slot}")

    def store(self, core, *slot):
        op = slot[0]
        if op == "store":
            addr, src = slot[1], slot[2]
            base_addr = core.scratch[addr]
            self.mem_write[base_addr] = core.scratch[src]
        elif op == "vstore":
            addr, src = slot[1], slot[2]
            base_addr = core.scratch[addr]
            for vi in range(VLEN):
                self.mem_write[base_addr + vi] = core.scratch[src + vi]
        else:
            raise NotImplementedError(f"Unknown store op {slot}")

    def flow(self, core, *slot):
        op = slot[0]
        if op == "select":
            dest, cond, a, b = slot[1], slot[2], slot[3], slot[4]
            self.scratch_write[dest] = (core.scratch[a] if core.scratch[cond] != 0 else core.scratch[b])
        elif op == "add_imm":
            dest, a, imm = slot[1], slot[2], slot[3]
            self.scratch_write[dest] = (core.scratch[a] + imm) % (2**32)
        elif op == "vselect":
            dest, cond, a, b = slot[1], slot[2], slot[3], slot[4]
            for vi in range(VLEN):
                self.scratch_write[dest + vi] = (core.scratch[a + vi] if core.scratch[cond + vi] != 0 else core.scratch[b + vi])
        elif op == "halt":
            core.state = CoreState.STOPPED
        elif op == "pause":
            if self.enable_pause:
                core.state = CoreState.PAUSED
        elif op == "trace_write":
            val = slot[1]
            core.trace_buf.append(core.scratch[val])
        elif op == "cond_jump":
            cond, addr = slot[1], slot[2]
            if core.scratch[cond] != 0:
                core.pc = addr
        elif op == "cond_jump_rel":
            cond, offset = slot[1], slot[2]
            if core.scratch[cond] != 0:
                core.pc += offset
        elif op == "jump":
            addr = slot[1]
            core.pc = addr
        elif op == "jump_indirect":
            addr = slot[1]
            core.pc = core.scratch[addr]
        elif op == "coreid":
            dest = slot[1]
            self.scratch_write[dest] = core.id
        else:
            raise NotImplementedError(f"Unknown flow op {slot}")

    def trace_post_step(self, instr, core):
        # You can add extra stuff to the trace if you want!
        for addr, (name, length) in self.debug_info.scratch_map.items():
            if any((addr + vi) in self.scratch_write for vi in range(length)):
                val = str(core.scratch[addr : addr + length])
                val = val.replace("[", "").replace("]", "")
                self.trace.write(
                    f'{{"name": "{val}", "cat": "op", "ph": "X", "pid": {len(self.cores) + core.id}, "tid": {BASE_ADDR_TID + addr}, "ts": {self.cycle}, "dur": 1 }},\n'
                )

    def trace_slot(self, core, slot, name, i):
        self.trace.write(
            f'{{"name": "{slot[0]}", "cat": "op", "ph": "X", "pid": {core.id}, "tid": {self.tids[(core.id, name, i)]}, "ts": {self.cycle}, "dur": 1, "args":{{"slot": "{str(slot)}", "named": "{str(self.rewrite_slot(slot))}" }} }},\n'
        )

    def step(self, instr: Instruction, core):
        """
        Execute all the slots in each engine for a single instruction bundle
        """
        ENGINE_FNS = {
            "alu": self.alu,
            "valu": self.valu,
            "load": self.load,
            "store": self.store,
            "flow": self.flow,
        }
        self.scratch_write = {}
        self.mem_write = {}
        for name, slots in instr.items():
            if name == "debug":
                if not self.enable_debug:
                    continue
                for slot in slots:
                    if slot[0] == "compare":
                        loc, key = slot[1], slot[2]
                        ref = self.value_trace[key]
                        res = core.scratch[loc]
                        assert res == ref, f"{res} != {ref} for {key} at pc={core.pc}"
                    elif slot[0] == "vcompare":
                        loc, keys = slot[1], slot[2]
                        ref = [self.value_trace[key] for key in keys]
                        res = core.scratch[loc : loc + VLEN]
                        assert res == ref, (
                            f"{res} != {ref} for {keys} at pc={core.pc} loc={loc}"
                        )
                continue
            assert len(slots) <= SLOT_LIMITS[name]
            for i, slot in enumerate(slots):
                if self.trace is not None:
                    self.trace_slot(core, slot, name, i)
                ENGINE_FNS[name](core, *slot)
        for addr, val in self.scratch_write.items():
            core.scratch[addr] = val
        for addr, val in self.mem_write.items():
            self.mem[addr] = val


@dataclass
class DebugInfo:
    scratch_map: dict[int, tuple[str, int]] = field(default_factory=dict)


@dataclass
class Tree:
    """
    An implicit perfect balanced binary tree with values on the nodes.
    """

    height: int
    values: list[int]

    @staticmethod
    def generate(height: int):
        values = [random.randint(0, 2**32 - 1) for _ in range(2**height - 1)]
        return Tree(height, values)


@dataclass
class Input:
    """
    A batch of inputs, indices to nodes (starting as 0) and initial input
    values. We then iterate these for a specified number of rounds.
    """

    indices: list[int]
    values: list[int]
    rounds: int

    @staticmethod
    def generate(t: Tree, batch_size: int, rounds: int):
        indices = [0] * batch_size
        values = [random.randint(0, 2**32 - 1) for _ in range(batch_size)]
        return Input(indices, values, rounds)


HASH_STAGES = [
    ("^", 0x12345678, ">>", "^", 16),
    ("+", 0x87654321, "<<", "^", 8),
    ("^", 0xDEADBEEF, ">>", "^", 4),
    ("+", 0xCAFEBABE, "<<", "^", 2),
    ("^", 0xFEEDFACE, ">>", "^", 1),
    ("+", 0x11223344, "^", "^", 32),
]


def myhash(a: int) -> int:
    """A simple 32-bit hash function"""
    fns = {
        "+": lambda x, y: x + y,
        "^": lambda x, y: x ^ y,
        "<<": lambda x, y: x << y,
        ">>": lambda x, y: x >> y,
    }

    def r(x):
        return x % (2**32)

    for op1, val1, op2, op3, val3 in HASH_STAGES:
        a = r(fns[op2](r(fns[op1](a, val1)), r(fns[op3](a, val3))))

    return a


def reference_kernel(t: Tree, inp: Input):
    """
    Reference implementation of the kernel.

    A parallel tree traversal where at each node we set
    cur_inp_val = myhash(cur_inp_val ^ node_val)
    and then choose the left branch if cur_inp_val is even.
    If we reach the bottom of the tree we wrap around to the top.
    """
    for h in range(inp.rounds):
        for i in range(len(inp.indices)):
            idx = inp.indices[i]
            val = inp.values[i]
            val = myhash(val ^ t.values[idx])
            idx = 2 * idx + (1 if val % 2 == 0 else 2)
            idx = 0 if idx >= len(t.values) else idx
            inp.values[i] = val
            inp.indices[i] = idx


def build_mem_image(t: Tree, inp: Input) -> list[int]:
    """
    Build a flat memory image of the problem.
    """
    header = 7
    extra_room = len(t.values) + len(inp.indices) * 2 + VLEN * 2 + 32
    mem = [0] * (
        header + len(t.values) + len(inp.indices) + len(inp.values) + extra_room
    )
    forest_values_p = header
    inp_indices_p = forest_values_p + len(t.values)
    inp_values_p = inp_indices_p + len(inp.values)
    extra_room = inp_values_p + len(inp.values)

    mem[0] = inp.rounds
    mem[1] = len(t.values)
    mem[2] = len(inp.indices)
    mem[3] = t.height
    mem[4] = forest_values_p
    mem[5] = inp_indices_p
    mem[6] = inp_values_p
    mem[7] = extra_room

    mem[header:inp_indices_p] = t.values
    mem[inp_indices_p:inp_values_p] = inp.indices
    mem[inp_values_p:] = inp.values
    return mem


def myhash_traced(a: int, trace: dict[Any, int], round: int, batch_i: int) -> int:
    """A simple 32-bit hash function"""
    fns = {
        "+": lambda x, y: x + y,
        "^": lambda x, y: x ^ y,
        "<<": lambda x, y: x << y,
        ">>": lambda x, y: x >> y,
    }

    def r(x):
        return x % (2**32)

    for i, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
        a = r(fns[op2](r(fns[op1](a, val1)), r(fns[op3](a, val3))))
        trace[(round, batch_i, "hash_stage", i)] = a

    return a


def reference_kernel2(mem: list[int], trace: dict[Any, int] = {}):
    """
    Reference implementation of the kernel on a flat memory.
    """
    # This is the initial memory layout
    rounds = mem[0]
    n_nodes = mem[1]
    batch_size = mem[2]
    # Offsets into the memory which indices get added to
    forest_values_p = mem[4]
    inp_indices_p = mem[5]
    inp_values_p = mem[6]
    yield mem
    for h in range(rounds):
        for i in range(batch_size):
            idx = mem[inp_indices_p + i]
            trace[(h, i, "idx")] = idx
            val = mem[inp_values_p + i]
            trace[(h, i, "val")] = val
            node_val = mem[forest_values_p + idx]
            trace[(h, i, "node_val")] = node_val
            val = myhash_traced(val ^ node_val, trace, h, i)
            trace[(h, i, "hashed_val")] = val
            idx = 2 * idx + (1 if val % 2 == 0 else 2)
            trace[(h, i, "next_idx")] = idx
            idx = 0 if idx >= n_nodes else idx
            trace[(h, i, "wrapped_idx")] = idx
            mem[inp_values_p + i] = val
            mem[inp_indices_p + i] = idx
    # You can add new yields or move this around for debugging
    # as long as it's matched by pause instructions.
    # The submission tests evaluate only on final memory.
    yield mem
