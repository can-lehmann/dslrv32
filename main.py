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

    def format(self, indent):
        res = ind(indent)
        if self.width > 0:
            res += f"%{self.name} = "
        res += self.__class__.__name__.lower()
        if len(self.args) > 0:
            res += " " + ", ".join([arg.format_arg() for arg in self.args])
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

class Slice(Instr):
    def __init__(self, value, offset, width):
        super().__init__(width, [value])
        self.offset = offset

class Repeat(Instr):
    def __init__(self, value, count):
        super().__init__(value.width * count, [value])
        self.count = count

class Read(Instr):
    def __init__(self, resource, index, enable):
        super().__init__(resource.width, [index, enable])
        self.resource = resource

class BaseWrite(Instr):
    def __init__(self, resource, value, index, enable):
        super().__init__(0, [value, index, enable])
        self.resource = resource

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

if __name__ == "__main__":
    from parse_opcodes import InstEncoding
    inst_encodings = InstEncoding.load_dir("opcodes")
    print(inst_encodings)

    processor = Processor("rv32")
    builder = Builder(processor)

    pc = builder.resource(Resource("pc", 32))
    reg_file = builder.resource(Resource("reg_file", 32))
    inst_mem = builder.resource(Resource("inst_mem", 32))
    data_mem = builder.resource(Resource("data_mem", 32))

    with builder.group("fetch"):
        pc_value = pc.read()
        inst = inst_mem.read(pc_value)

        pc.predict(pc_value + 4)

    with builder.group("decode"):
        is_inst = {}
        is_imm20 = builder.const(1, 0)
        is_jimm20 = builder.const(1, 0)
        is_imm12 = builder.const(1, 0)
        is_imm12_hi_lo = builder.const(1, 0)
        is_bimm12 = builder.const(1, 0)
        for enc in inst_encodings:
            matches = inst & enc.mask == enc.pattern
            is_inst[enc.name] = matches
            if "imm20" in enc.args:
                is_imm20 |= matches
            if "jimm20" in enc.args:
                is_jimm20 |= matches
            if "imm12" in enc.args:
                is_imm12 |= matches
            if "imm12hi" in enc.args:
                is_imm12_hi_lo |= matches

        rs1 = inst[15:20]
        rs2 = inst[20:25]
        rd = inst[7:12]

        imm20 = inst[12:32].concat(builder.const(12, 0))
        imm12 = inst[20:32].s_ext(32)

        has_imm = is_imm20 | is_imm12
        imm, = builder.cond(
            (is_imm20, imm20),
            (is_imm12, imm12),
            (0,)
        )

        r1 = reg_file.read(rs1)
        r2 = reg_file.read(rs2)

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
            (is_inst["auipc"],                   1, pc_value + imm),
            (is_inst["lui"],                     1, imm),
            (builder.const(1, 0), 0)
        )

        branch_target, = builder.cond(
            (is_inst["jalr"], r1),
            (pc_value + imm,)
        )

        branch, = builder.cond(
            (is_inst["jal"],   1),
            (is_inst["jalr"],  1),
            (is_inst["beq"],   r1 == r2),
            (is_inst["bne"],   r1 != r2),
            (is_inst["blt"],   r1.lt_s(r2)),
            (is_inst["bge"],   ~r1.lt_s(r2)),
            (is_inst["bltu"],  r1.lt_u(r2)),
            (is_inst["bgeu"],  ~r1.lt_u(r2)),
            (0,)
        )

        next_pc = branch.mux(branch_target, pc_value + 4)
        pc.write(next_pc)

    with builder.group("memory"):
        pass #data_mem.read(, enable)

    with builder.group("writeback"):
        reg_file.write(alu_res, index=rd, enable=alu_valid)

    print(processor.format(0))
