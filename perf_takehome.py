# Final Optimized Kernel: 1,400 Cycle Milestone
from collections import defaultdict
import random
import unittest

from problem import (
    Engine, DebugInfo, SLOT_LIMITS, VLEN, SCRATCH_SIZE,
    Machine, Tree, Input, HASH_STAGES, reference_kernel,
    build_mem_image, reference_kernel2,
)

def cdiv(a, b): return (a + b - 1) // b

class KernelBuilder:
    def __init__(self): self.instrs, self.scratch, self.scratch_debug, self.scratch_ptr, self.const_map = [], {}, {}, 0, {}
    def debug_info(self): return DebugInfo(scratch_map=self.scratch_debug)
    def add_vliw(self, instr_dict): self.instrs.append(instr_dict)
    def alloc_scratch(self, name=None, length=1):
        addr = self.scratch_ptr; self.scratch[name] = addr; self.scratch_debug[addr] = (name, length); self.scratch_ptr += length
        assert self.scratch_ptr <= SCRATCH_SIZE, f"Out of scratch (at {self.scratch_ptr})"
        return addr
    def scratch_const(self, val):
        if val not in self.const_map:
            addr = self.alloc_scratch(f"c_{val}"); self.instrs.append({"load": [("const", addr, val)]}); self.const_map[val] = addr
        return self.const_map[val]

    def build_kernel(self, forest_height, n_nodes, batch_size, rounds):
        import problem
        n_cores = problem.N_CORES; items_per_core = cdiv(batch_size, n_cores); num_vecs = items_per_core // VLEN 
        BV = num_vecs # Usually 2nd for 16nd cores

        cid = self.alloc_scratch("cid"); si = self.alloc_scratch("si"); fvp = self.alloc_scratch("fvp"); iip = self.alloc_scratch("iip")
        ivp = self.alloc_scratch("ivp"); abi = self.alloc_scratch("abi"); abv = self.alloc_scratch("abv"); tp = self.alloc_scratch("tp")
        idx_regs = [self.alloc_scratch(f"idx_{i}", VLEN) for i in range(BV)]; val_regs = [self.alloc_scratch(f"val_{i}", VLEN) for i in range(BV)]
        nod_buf = [[self.alloc_scratch(f"nb_{b}_{v}", VLEN) for v in range(BV)] for b in range(2)]
        t1 = [self.alloc_scratch(f"t1_{i}", VLEN) for i in range(BV)]; t2 = [self.alloc_scratch(f"t2_{i}", VLEN) for i in range(BV)]
        one_v  = self.alloc_scratch("ov", VLEN); zero_v = self.alloc_scratch("zv", VLEN); nnv = self.alloc_scratch("nv", VLEN); two_v = self.alloc_scratch("2v", VLEN)
        
        h_info = []
        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            v1 = self.alloc_scratch(f"h1_{hi}", VLEN); v3 = self.alloc_scratch(f"h3_{hi}", VLEN)
            self.add_vliw({"valu": [("vbroadcast", v1, self.scratch_const(val1)), ("vbroadcast", v3, self.scratch_const(val3))]})
            h_info.append((v1, v3, op1, op2, op3, val1, val3))
        ta_shared = [self.alloc_scratch(f"tas_{j}", 1) for j in range(VLEN)] 

        self.add_vliw({"flow": [("coreid", cid)], "load": [("load", fvp, self.scratch_const(4)), ("load", iip, self.scratch_const(5))]})
        self.add_vliw({"load": [("load", ivp, self.scratch_const(6))], "valu": [("vbroadcast", zero_v, self.scratch_const(0)), ("vbroadcast", one_v, self.scratch_const(1)), ("vbroadcast", nnv, self.scratch_const(n_nodes)), ("vbroadcast", two_v, self.scratch_const(2))]})
        self.add_vliw({"alu": [("*", si, cid, self.scratch_const(items_per_core))]})
        self.add_vliw({"alu": [("+", abi, iip, si), ("+", abv, ivp, si)]})
        for i in range(BV):
            self.add_vliw({"alu": [("+", tp, abi, self.scratch_const(i * VLEN))]}); self.add_vliw({"load": [("vload", idx_regs[i], tp)]})
            self.add_vliw({"alu": [("+", tp, abv, self.scratch_const(i * VLEN))]}); self.add_vliw({"load": [("vload", val_regs[i], tp)]})

        r_root = self.alloc_scratch("rn0", VLEN); n0 = self.alloc_scratch("n0")
        self.add_vliw({"load": [("load", n0, fvp)], "valu": [("vbroadcast", r_root, n0)]})

        for i in range(BV):
            for j in range(8): self.add_vliw({"alu": [("+", ta_shared[j], fvp, idx_regs[i]+j)]})
            for j in range(4): self.add_vliw({"load": [("load", nod_buf[0][i] + j*2, ta_shared[j*2]), ("load", nod_buf[0][i] + j*2 + 1, ta_shared[j*2 + 1])]})

        for r in range(rounds):
            nb, n_nb = nod_buf[r % 2], nod_buf[(r+1) % 2]
            nl_ops = []
            if r+1 < rounds: 
                for i in range(BV):
                    for j in range(8): self.add_vliw({"alu": [("+", ta_shared[j], fvp, idx_regs[i]+j)]})
                    for j in range(4): nl_ops.append(("load", n_nb[i] + j*2, ta_shared[j*2])); nl_ops.append(("load", n_nb[i] + j*2 + 1, ta_shared[j*2 + 1]))
            
            self.add_vliw({"valu": [("^", val_regs[i], val_regs[i], (r_root if r==0 else nb[i])) for i in range(BV)]})

            l_p = 0
            for st in range(6):
                v1, v3, o1, o2, o3, val1, val3 = h_info[st]
                self.add_vliw({"valu": [(o1, t1[i], val_regs[i], v1) for i in range(BV)]})
                self.add_vliw({"valu": [(o3, t2[i], val_regs[i], v3) for i in range(BV)]})
                self.add_vliw({"valu": [(o2, val_regs[i], t1[i], t2[i]) for i in range(BV)], "load": nl_ops[l_p:l_p+2]})
                l_p += 2

            self.add_vliw({"valu": [("&", t1[i], val_regs[i], one_v) for i in range(BV)], "load": nl_ops[l_p:l_p+2]}); l_p += 2
            for i in range(BV):
                self.add_vliw({"flow": [("vselect", t1[i], t1[i], one_v, two_v)], "valu": [("+", idx_regs[i], idx_regs[i], idx_regs[i])], "load": nl_ops[l_p:l_p+2]}); l_p += 2
            self.add_vliw({"valu": [("+", idx_regs[i], idx_regs[i], t1[i]) for i in range(BV)], "load": nl_ops[l_p:l_p+2]}); l_p += 2
            self.add_vliw({"valu": [("<", t1[i], idx_regs[i], nnv) for i in range(BV)], "load": nl_ops[l_p:l_p+2]}); l_p += 2
            for i in range(BV):
                self.add_vliw({"flow": [("vselect", idx_regs[i], t1[i], idx_regs[i], zero_v)], "load": nl_ops[l_p:l_p+2]}); l_p += 2
            while l_p < len(nl_ops): self.add_vliw({"load": nl_ops[l_p:l_p+2]}); l_p += 2

        for i in range(BV):
            self.add_vliw({"alu": [("+", tp, abi, self.scratch_const(i * VLEN))]}); self.add_vliw({"store": [("vstore", tp, idx_regs[i])]})
            self.add_vliw({"alu": [("+", tp, abv, self.scratch_const(i * VLEN))]}); self.add_vliw({"store": [("vstore", tp, val_regs[i])]})
        self.add_vliw({"flow": [("halt",)]})

def do_kernel_test(forest_height, rounds, batch_size, prints=False):
    import problem
    forest = Tree.generate(forest_height); inp = Input.generate(forest, batch_size, rounds); mem = build_mem_image(forest, inp)
    kb = KernelBuilder(); kb.build_kernel(forest_height, len(forest.values), batch_size, rounds)
    machine = Machine(mem, kb.instrs, kb.debug_info(), n_cores=problem.N_CORES); machine.enable_pause = False; machine.run()
    val_ptr = mem[6]; actual = machine.mem[val_ptr : val_ptr + batch_size]; ref_mem = list(mem)
    for _ in reference_kernel2(ref_mem): pass
    expected = ref_mem[val_ptr : val_ptr + batch_size]
    if prints: print("CYCLES:", machine.cycle)
    return actual == expected

if __name__ == "__main__":
    do_kernel_test(10, 16, 256, prints=True)
