# Copyright 2025 Can Joshua Lehmann
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from enum import Enum

def ind(indent, width = 2):
    return " " * (indent * width)

class Value:
    def __init__(self, width):
        self.width = width
        self.name = ""

    def format_arg(self):
        return f"%{self.name}"

class Const(Value):
    def __init__(self, width, value):
        super().__init__(width)
        self.value = value

    def format_arg(self):
        return f"{self.width}'d{self.value}"

class Instr(Value):
    def __init__(self, width, args):
        super().__init__(width)
        self.args = args

    def format_instr_kind(self):
        return self.__class__.__name__.lower()

    def format_parameters(self):
        return []

    def format(self, indent):
        res = ind(indent)
        if self.width > 0:
            res += f"%{self.name} = "
        res += self.format_instr_kind()
        if len(self.args) > 0:
            res += " " + ", ".join(
                [arg.format_arg() for arg in self.args] +
                self.format_parameters()
            )
        return res + "\n"

OpKind = Enum("OpKind", [
    "Add", "Sub", "Mul",
    "Shl", "ShrU", "ShrS",
    "And", "Or", "Xor", "Not",
    "Eq", "LtU", "LtS",
    "Concat",
    "Mux"
])

def infer_op_width(kind, args):
    def eq(a, b):
        nonlocal args
        if args[a] is None:
            assert args[b] is not None
            args[a] = args[b]
        elif args[b] is None:
            args[b] = args[a]
        else:
            assert args[a] == args[b]

    match kind:
        case OpKind.Add | OpKind.Sub | OpKind.Mul | OpKind.And | OpKind.Or | OpKind.Xor | OpKind.Shl | OpKind.ShrU | OpKind.ShrS:
            assert len(args) == 2
            eq(0, 1)
            return args[0]
        case OpKind.Not:
            assert args[0] is not None
            return args[0]
        case OpKind.Eq | OpKind.LtU | OpKind.LtS:
            assert len(args) == 2
            eq(0, 1)
            return 1
        case OpKind.Mux:
            assert len(args) == 3
            eq(1, 2)
            if args[0] is None:
                args[0] = 1
            else:
                assert args[0] == 1
            return args[1]
        case OpKind.Concat:
            assert len(args) == 2
            assert args[0] is not None
            assert args[1] is not None
            return args[0] + args[1]
        case _:
            assert False

class Op(Instr):
    def __init__(self, kind, args):
        super().__init__(infer_op_width(kind, [arg.width for arg in args]), args)
        self.kind = kind

    def format_instr_kind(self):
        return self.kind.name.lower()

class Slice(Instr):
    def __init__(self, value, offset, width):
        super().__init__(width, [value])
        self.offset = offset

    def format_parameters(self):
        return [f"offset={self.offset}", f"width={self.width}"]

class Repeat(Instr):
    def __init__(self, value, count):
        super().__init__(value.width * count, [value])
        self.count = count

    def format_parameters(self):
        return [f"count={self.count}"]

class Read(Instr):
    def __init__(self, resource, index, enable):
        super().__init__(resource.width, [index, enable])
        self.resource = resource

    def format_parameters(self):
        return [f"resource={self.resource.name}"]

class BaseWrite(Instr):
    def __init__(self, resource, value, index, enable):
        super().__init__(0, [value, index, enable])
        self.resource = resource

    def format_parameters(self):
        return [f"resource={self.resource.name}"]

class Write(BaseWrite): pass
class Predict(BaseWrite): pass

class Group:
    def __init__(self):
        self.instrs = []
        self.name = ""

    def add(self, instr):
        self.instrs.append(instr)
        return instr

    def format(self, indent):
        res = ind(indent) + f"{self.name}:\n"
        for instr in self.instrs:
            res += instr.format(indent + 1)
        return res

class Resource:
    def __init__(self, name, width):
        self.name = name
        self.width = width
        self.initial = {}

class RegisterResource(Resource):
    def __init__(self, name, width):
        super().__init__(name, width)

class MemoryResource(Resource):
    def __init__(self, name, width, size, read_delay=1, write_delay=1):
        super().__init__(name, width)
        self.size = size
        self.read_delay = read_delay
        self.write_delay = write_delay

class Processor:
    def __init__(self, name):
        self.name = name
        self.resources = []
        self.groups = []

    def add_group(self, group):
        self.groups.append(group)
        return group

    def add_resource(self, resource):
        self.resources.append(resource)
        return resource

    def autoname(self):
        id = 0
        for group in self.groups:
            for instr in group.instrs:
                if len(instr.name) == 0:
                    instr.name = str(id)
                    id += 1

    def format(self, indent):
        self.autoname()
        res = ind(indent) + f"{self.name} {{\n"
        for group in self.groups:
            res += group.format(indent)
        res += ind(indent) + "}"
        return res

# Builder

class Builder:
    def __init__(self, processor):
        self.processor = processor
        self.current_group = None

    def group(self, name):
        group = Group()
        group.name = name
        self.processor.add_group(group)
        return GroupBuilder(self, group)

    def resource(self, resource):
        self.processor.add_resource(resource)
        return ResourceBuilder(self, resource)

    def const(self, width, value):
        return ValueBuilder(self, Const(width, value))

    def emit(self, instr):
        self.current_group.add(instr)
        return instr

    def cond(self, *branches):
        res = branches[-1]
        for branch in branches[-2::-1]:
            res = branch[0].mux(branch[1:], res)
        return res

    def concat(self, *args):
        res = args[0]
        for value in args[1:]:
            res = ValueBuilder.op(OpKind.Concat, res, value)
        return res

class GroupBuilder:
    def __init__(self, builder, group):
        self.builder = builder
        self.group = group

    def __enter__(self):
        assert self.builder.current_group is None
        self.builder.current_group = self.group

    def __exit__(self, exc_type, exc_value, trackback):
        self.builder.current_group = None

class ValueBuilder:
    def __init__(self, builder, value):
        assert isinstance(value, Value)
        self.builder = builder
        self.value = value

    def name(self, name):
        self.value.name = name
        return self

    def op(kind, *args):
        widths = [
            arg.value.width if isinstance(arg, ValueBuilder) else None
            for arg in args
        ]
        infer_op_width(kind, widths)
        op = Op(kind, [
            arg.value if isinstance(arg, ValueBuilder) else Const(width, arg)
            for width, arg in zip(widths, args)
        ])
        builder.emit(op)
        return ValueBuilder(builder, op)

    def __add__(self, other):
        return ValueBuilder.op(OpKind.Add, self, other)

    def __sub__(self, other):
        return ValueBuilder.op(OpKind.Sub, self, other)

    def __and__(self, other):
        return ValueBuilder.op(OpKind.And, self, other)

    def __or__(self, other):
        return ValueBuilder.op(OpKind.Or, self, other)

    def __xor__(self, other):
        return ValueBuilder.op(OpKind.Xor, self, other)

    def __invert__(self):
        return ValueBuilder.op(OpKind.Not, self)

    def __ior__(self, other):
        self.value = (self | other).value
        return self

    def __eq__(self, other):
        return ValueBuilder.op(OpKind.Eq, self, other)

    def __ne__(self, other):
        return ~(self == other)

    def concat(self, other):
        return ValueBuilder.op(OpKind.Concat, self, other)

    def shl(self, other):
        return ValueBuilder.op(OpKind.Shl, self, other)

    def shr_u(self, other):
        return ValueBuilder.op(OpKind.ShrU, self, other)

    def shr_s(self, other):
        return ValueBuilder.op(OpKind.ShrS, self, other)

    def lt_u(self, other):
        return ValueBuilder.op(OpKind.LtU, self, other)

    def lt_s(self, other):
        return ValueBuilder.op(OpKind.LtS, self, other)

    def __getitem__(self, index):
        if isinstance(index, slice):
            instr = Slice(self.value, index.start, index.stop - index.start)
        elif isinstance(index, int):
            instr = Slice(self.value, index, 1)
        else:
            assert False
        builder.emit(instr)
        return ValueBuilder(self.builder, instr)

    def repeat(self, count):
        repeat = Repeat(self.value, count)
        builder.emit(repeat)
        return ValueBuilder(self.builder, repeat)

    def s_ext(self, width):
        assert self.value.width <= width
        if self.value.width == width:
            return ValueBuilder(self.builder, self.value)
        return self[self.value.width - 1].repeat(width - self.value.width).concat(self)

    def mux(self, true_value, false_value):
        if isinstance(true_value, tuple):
            assert isinstance(false_value, tuple)
            assert len(true_value) == len(false_value)
            return tuple(self.mux(a, b) for a, b in zip(true_value, false_value))
        else:
            return ValueBuilder.op(OpKind.Mux, self, true_value, false_value)

class ResourceBuilder:
    def __init__(self, builder, resource):
        self.builder = builder
        self.resource = resource

    def init(self, value, index=0):
        if isinstance(value, ValueBuilder):
            value = value.value
        if isinstance(value, Const):
            value = value.value
        self.resource.initial[index] = value

    def create_index_enable(self, index, enable):
        if index is None:
            index = self.builder.const(1, 0)
        if enable is None:
            enable = self.builder.const(1, 1)
        return index, enable

    def read(self, index=None, enable=None):
        index, enable = self.create_index_enable(index, enable)
        read = Read(self.resource, index.value, enable.value)
        self.builder.emit(read)
        return ValueBuilder(self.builder, read)

    def write(self, value, index=None, enable=None):
        index, enable = self.create_index_enable(index, enable)
        self.builder.emit(Write(self.resource, value.value, index.value, enable.value))

    def predict(self, value, index=None, enable=None):
        index, enable = self.create_index_enable(index, enable)
        self.builder.emit(Predict(self.resource, value.value, index.value, enable.value))

def generate_verilog(processor):
    ports = ["input clock"]
    body = ""

    processor.autoname()

    gen_sym_id = 0
    def gen_sym():
        nonlocal gen_sym_id
        gen_sym_id += 1
        return f"t{gen_sym_id}"

    for resource in processor.resources:
        match resource:
            case RegisterResource(name=name, width=width, initial=initial):
                body += f"reg [{width - 1}:0] {name};\n"
                if 0 in initial:
                    body += f"initial {name} = {width}'d{initial[0]};\n"
            case MemoryResource(name=name, width=width, initial=initial, size=size):
                body += f"reg [{width - 1}:0] {name}[{size}];\n"
                for index, value in initial.items():
                    body += f"initial {name}[{index}] = {width}'d{value};\n"

    for group in processor.groups:
        for instr in group.instrs:
            args = []
            for arg in instr.args:
                if isinstance(arg, Instr):
                    args.append("_" + arg.name)
                elif isinstance(arg, Const):
                    args.append(arg.format_arg())
                else:
                    assert False

            expr = None
            match instr:
                case Read(resource=resource):
                    match resource:
                        case RegisterResource(name=name):
                            expr = name
                        case MemoryResource(name=name, width=width, read_delay=read_delay):
                            expr = f"{name}[{args[0]}]"
                            for it in range(read_delay):
                                name = gen_sym()
                                body += f"reg [{width - 1}:0] {name};\n"
                                body += f"always @(posedge clock) {name} <= {expr};\n"
                                expr = name
                case Predict(): pass
                case Write(resource=resource):
                    match resource:
                        case RegisterResource(name=name):
                            body += f"always @(posedge clock) "
                            body += f"if ({args[2]}) "
                            body += f"{instr.resource.name} <= {args[0]};\n"
                        case MemoryResource(name=name, width=width, write_delay=write_delay):
                            assert write_delay == 1
                            body += f"always @(posedge clock) "
                            body += f"if ({args[2]}) "
                            body += f"{name}[{args[1]}] <= {args[0]};\n"
                case Op(kind=kind):
                    SIMPLE_BINOPS = {
                        OpKind.Add: "+", OpKind.Sub: "-", OpKind.Mul: "*",
                        OpKind.Shl: "<<", OpKind.ShrU: ">>", OpKind.ShrS: ">>>",
                        OpKind.And: "&", OpKind.Or: "|", OpKind.Xor: "^",
                        OpKind.Eq: "=="
                    }

                    if kind in SIMPLE_BINOPS:
                        expr = f"{args[0]} {SIMPLE_BINOPS[kind]} {args[1]}"
                    elif kind == OpKind.Not:
                        expr = f"~{args[0]}"
                    elif kind == OpKind.LtS:
                        expr = f"$signed({args[0]}) < $signed({args[1]})"
                    elif kind == OpKind.LtU:
                        expr = f"$unsigned({args[0]}) < $unsigned({args[1]})"
                    elif kind == OpKind.Concat:
                        expr = f"{{{args[0]}, {args[1]}}}"
                    elif kind == OpKind.Mux:
                        expr = f"{args[0]} ? {args[1]} : {args[2]}"
                    else:
                        print(kind)
                        assert False
                case Slice(width=width, offset=offset):
                    expr = f"{args[0]}[{width + offset - 1}:{offset}]"
                case Repeat(count=count):
                    expr = f"{{{count}{{{args[0]}}}}}"
                case _:
                    print(instr.format(0))
                    assert False

            if expr is not None:
                body += f"wire [{instr.width - 1}:0] _{instr.name} = {expr};\n"

    ports = ", ".join(ports)
    return f"module {processor.name}({ports});\n{body}\nendmodule"

if __name__ == "__main__":
    from parse_opcodes import InstEncoding
    from elftools.elf.elffile import ELFFile
    from elftools.elf.constants import P_FLAGS

    processor = Processor("Processor")
    builder = Builder(processor)

    STATE_RUNNING = builder.const(3, 0)
    STATE_EBREAK = builder.const(3, 1)

    pc = builder.resource(RegisterResource("pc", 32))
    state = builder.resource(RegisterResource("state", 3))
    reg_file = builder.resource(MemoryResource("reg_file", 32, size = 32, read_delay = 0))
    inst_mem = builder.resource(MemoryResource("inst_mem", 32, size = 1 << 24, read_delay = 0))
    data_mem = builder.resource(MemoryResource("data_mem", 32, size = 1 << 24, read_delay = 0))

    reg_file.init(0, index=0)
    state.init(STATE_RUNNING)

    with open("rv32i_test", "rb") as f:
        elf_file = ELFFile(f)
        pc.init(elf_file["e_entry"])
        for segment in elf_file.iter_segments():
            mem = data_mem
            if segment["p_flags"] & P_FLAGS.PF_X:
                mem = inst_mem

            addr = segment["p_vaddr"]
            data = segment.data()
            for it in range(segment["p_memsz"]):
                if it < len(data):
                    byte = int(data[it])
                else:
                    byte = 0
                index = addr // 4
                offset = addr % 4
                if index in mem.resource.initial:
                    word = mem.resource.initial[index]
                else:
                    word = 0
                word |= int(byte) << (8 * offset)
                mem.init(word, index = index)
                addr += 1

    with builder.group("fetch"):
        state_value = state.read()
        is_running = state_value == STATE_RUNNING

        pc_value = pc.read()
        inst = inst_mem.read(pc_value.shr_u(2)).name("inst")

        pc.predict(pc_value + 4)

    with builder.group("decode"):
        is_inst = {}
        is_imm20 = builder.const(1, 0)
        is_jimm20 = builder.const(1, 0)
        is_imm12 = builder.const(1, 0)
        is_bimm12 = builder.const(1, 0)
        is_simm12 = builder.const(1, 0)

        inst_encodings = InstEncoding.load_dir("opcodes")
        for enc in inst_encodings:
            matches = inst & enc.mask == enc.pattern
            matches.name(f"is_{enc.name}")
            is_inst[enc.name] = matches
            if "imm20" in enc.args:
                is_imm20 |= matches
            if "jimm20" in enc.args:
                is_jimm20 |= matches
            if "imm12" in enc.args:
                is_imm12 |= matches
            if "bimm12hi" in enc.args:
                is_bimm12 |= matches
            if "imm12hi" in enc.args:
                is_simm12 |= matches

        rs1 = inst[15:20]
        rs2 = inst[20:25]
        rd = inst[7:12]

        imm20 = builder.concat(inst[12:32], builder.const(12, 0))
        jimm20 = builder.concat(
            inst[31],
            inst[12:20],
            inst[20],
            inst[25:31],
            inst[21:25],
            builder.const(1, 0)
        ).s_ext(32)
        imm12 = inst[20:32].s_ext(32)
        bimm12 = builder.concat(
            inst[31],
            inst[7],
            inst[25:31],
            inst[8:12],
            builder.const(1, 0)
        ).s_ext(32)

        has_imm = is_imm20 | is_jimm20 | is_imm12 | is_bimm12
        imm, = builder.cond(
            (is_imm20,  imm20),
            (is_jimm20, jimm20),
            (is_imm12,  imm12),
            (is_bimm12, bimm12),
            (0,)
        )
        imm.name("imm")

        r1 = reg_file.read(rs1).name("r1")
        r2 = reg_file.read(rs2).name("r2")

    with builder.group("execute"):
        rhs = has_imm.mux(imm, r2)
        alu_valid, alu_res = builder.cond(
            (is_inst["add"]  | is_inst["addi"],  1, r1 + rhs),
            (is_inst["sub"],                     1, r1 - rhs),
            (is_inst["and"]  | is_inst["andi"],  1, r1 & rhs),
            (is_inst["or"]   | is_inst["ori"],   1, r1 | rhs),
            (is_inst["xor"]  | is_inst["xori"],  1, r1 ^ rhs),
            (is_inst["sll"],                     1, r1.shl(rhs)),   # TODO slli
            (is_inst["srl"],                     1, r1.shr_u(rhs)),
            (is_inst["sra"],                     1, r1.shr_s(rhs)),
            (is_inst["slt"]  | is_inst["slti"],  1, r1.lt_s(rhs).mux(builder.const(32, 1), 0)),
            (is_inst["sltu"] | is_inst["sltiu"], 1, r1.lt_u(rhs).mux(builder.const(32, 1), 0)),
            (is_inst["jal"],                     1, pc_value + 4),
            (is_inst["jalr"],                    1, pc_value + 4),
            (is_inst["auipc"],                   1, pc_value + imm),
            (is_inst["lui"],                     1, imm),
            (builder.const(1, 0), 0)
        )

        branch_target, = builder.cond(
            (is_inst["jalr"], r1 + imm),
            (is_inst["ebreak"], pc_value),
            (pc_value + imm,)
        )

        branch, = builder.cond(
            (is_inst["jal"],    1),
            (is_inst["jalr"],   1),
            (is_inst["ebreak"], 1),
            (is_inst["beq"],    r1 == r2),
            (is_inst["bne"],    r1 != r2),
            (is_inst["blt"],    r1.lt_s(r2)),
            (is_inst["bge"],    ~r1.lt_s(r2)),
            (is_inst["bltu"],   r1.lt_u(r2)),
            (is_inst["bgeu"],   ~r1.lt_u(r2)),
            (0,)
        )

        next_state, = builder.cond(
            (is_running & is_inst["ebreak"], STATE_EBREAK),
            (state_value,)
        )

        next_pc = branch.mux(branch_target, pc_value + 4)
        pc.write(next_pc, enable = is_running)

    with builder.group("memory"):
        pass #data_mem.read(, enable)

    with builder.group("writeback"):
        reg_file.write(alu_res, index = rd, enable = is_running & alu_valid & (rd != 0))

        state.write(next_state)

    print(processor.format(0))

    with open(f"{processor.name}.v", "w") as f:
        f.write(generate_verilog(processor))
