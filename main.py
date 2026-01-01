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

import math
from enum import Enum

def ind(indent, width = 2):
    return " " * (indent * width)

class Value:
    def __init__(self, width):
        self.width = width
        self.name = ""

    def is_autonamed(self):
        return len(self.name) == 0 or self.name.isdigit()

    def format_arg(self):
        return f"%{self.name}"

    def is_const(self, value):
        return isinstance(self, Const) and self.value == value

class Const(Value):
    def __init__(self, width, value):
        super().__init__(width)
        self.value = value

    def format_arg(self):
        return f"{self.width}'d{self.value}"

class Arg(Value):
    def __init__(self, name, width):
        super().__init__(width)
        self.name = name

class Wire(Value):
    def __init__(self, value):
        super().__init__(value.width)
        self.value = value

    def get(self):
        return self.value

    def set(self, value):
        assert value.width == self.width
        self.value = value

    def format_arg(self):
        return f"wire({self.value.format_arg()})"

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
    def __init__(self, resource, index=None, enable=None):
        if index is None:
            index = Const(1, 0)
        if enable is None:
            enable = Const(1, 1)
        super().__init__(resource.width, [index, enable])
        self.resource = resource

    def format_parameters(self):
        return [f"resource={self.resource.name}"]

    def get_index(self):
        return self.args[0]

    def get_enable(self):
        return self.args[1]

class BaseWrite(Instr):
    def __init__(self, resource, value, index=None, enable=None):
        if index is None:
            index = Const(1, 0)
        if enable is None:
            enable = Const(1, 1)
        super().__init__(0, [value, index, enable])
        self.resource = resource

    def format_parameters(self):
        return [f"resource={self.resource.name}"]

    def get_enable(self):
        return self.args[2]

    def set_enable(self, enable):
        self.args[2] = enable

    def get_index(self):
        return self.args[1]
    
    def get_value(self):
        return self.args[0]

class Write(BaseWrite): pass
class Predict(BaseWrite): pass
class Forward(BaseWrite): pass

class UnknownGuard(Instr):
    def __init__(self, value):
        super().__init__(0, [value])


class Terminator:
    def __init__(self):
        pass

    def collect_groups(self):
        return set()

    def collect_always(self):
        return set()

class CommitTerminator(Terminator):
    def __init__(self):
        super().__init__()

    def format(self, indent):
        return "commit"

class AlwaysTerminator(Terminator):
    def __init__(self, group, args):
        super().__init__()
        self.group = group
        self.args = args

    def collect_groups(self):
        return {self.group}

    def format(self, indent):
        args = ", ".join([arg.format_arg() for arg in self.args])
        return f"{self.group.name}({args})"

    def collect_always(self):
        return {self}

class BranchTerminator(Terminator):
    def __init__(self, cond, true, false):
        super().__init__()
        self.cond = cond
        self.true = true
        self.false = false

    def collect_groups(self):
        return self.true.collect_groups().union(self.false.collect_groups())

    def format(self, indent):
        text = "branch\n"
        current = self
        while isinstance(current, BranchTerminator):
            text += ind(indent + 1)
            text += current.cond.format_arg()
            text += " -> "
            text += self.true.format(indent + 2)
            text += "\n"
            current = current.false
        text += ind(indent + 1) + "else -> " + current.format(indent + 2)
        return text

    def collect_always(self):
        return self.true.collect_always().union(self.false.collect_always())

class Group:
    def __init__(self):
        self.incoming = set()
        self.args = []
        self.instrs = []
        self.terminator = None
        self.name = ""

    def add(self, instr):
        self.instrs.append(instr)
        return instr

    def add_all(self, *instrs):
        self.instrs += instrs

    def format(self, indent):
        args = ", ".join([
            f"{arg.name}: {arg.width}"
            for arg in self.args
        ])
        res = ind(indent) + f"{self.name}({args}):\n"
        for instr in self.instrs:
            res += instr.format(indent + 1)
        if self.terminator is not None:
            res += ind(indent + 1) + "then " + self.terminator.format(indent + 1) + "\n"
        return res

class Resource:
    def __init__(self, name, width):
        self.name = name
        self.width = width
        self.initial = {}
    
    def __repr__(self):
        return f"Resource({self.name}, {self.width})"

class RegisterResource(Resource):
    def __init__(self, name, width, initial = None):
        super().__init__(name, width)
        if initial is not None:
            self.initial[0] = initial

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
                if instr.is_autonamed():
                    instr.name = str(id)
                    id += 1

    def fix_incoming(self):
        for group in self.groups:
            group.incoming = set()
        for group in self.groups:
            if group.terminator is not None:
                for next in group.terminator.collect_groups():
                    next.incoming.add(group)

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

    def group(self, name, *args):
        group = Group()
        group.name = name
        for (arg_name, arg_width) in args:
            group.args.append(Arg(arg_name, arg_width))
        self.processor.add_group(group)
        return GroupBuilder(self, group)

    def groups(self, *names):
        groups = {}
        for name in names:
            groups[name] = self.group(name)
        return groups

    def then(self, terminator):
        assert isinstance(terminator, Terminator)
        assert self.current_group.terminator is None
        self.current_group.terminator = terminator

    def commit(self):
        return CommitTerminator()

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

    def branch(self, cond, true, false):
        return BranchTerminator(cond.value, true, false)

class GroupBuilder:
    def __init__(self, builder, group):
        self.builder = builder
        self.group = group

    def __enter__(self):
        assert self.builder.current_group is None
        self.builder.current_group = self.group
        return tuple([
            ValueBuilder(self.builder, arg)
            for arg in self.group.args
        ])

    def __exit__(self, exc_type, exc_value, trackback):
        self.builder.current_group = None
    
    def __call__(self, *args):
        arg_values = []
        assert len(args) == len(self.group.args)
        for formal_arg, arg in zip(self.group.args, args):
            arg = ValueBuilder.ensure(arg, formal_arg.width, self.builder)
            arg_values.append(arg.value)
        return AlwaysTerminator(self.group, arg_values)

class ValueBuilder:
    def __init__(self, builder, value):
        assert isinstance(value, Value)
        self.builder = builder
        self.value = value

    def ensure(value, width, builder):
        if isinstance(value, ValueBuilder):
            return value
        else:
            return ValueBuilder(builder, Const(width, value))

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
            ValueBuilder.ensure(arg, width, builder).value
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

    def z_ext(self, width):
        assert self.value.width <= width
        if self.value.width == width:
            return ValueBuilder(self.builder, self.value)
        return self.builder.const(width - self.value.width, 0).concat(self)

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

    def forward(self, value, index=None, enable=None):
        index, enable = self.create_index_enable(index, enable)
        self.builder.emit(Forward(self.resource, value.value, index.value, enable.value))

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
            def value_name(value):
                if isinstance(value, Instr):
                    return "_" + value.name
                elif isinstance(value, Const):
                    return value.format_arg()
                else:
                    assert False

            args = [value_name(arg) for arg in instr.args]
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
                case Forward(): pass
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
                case UnknownGuard():
                    def is_unknown(value):
                        return " | ".join([
                            f"{value_name(value)}[{it}] === 1'bx"
                            for it in range(value.width)
                        ])
                    
                    def gen_reason(instr, indent = "  "):
                        nonlocal body

                        code = instr.format(0).replace("%", "%%").replace("\n", "").replace("%", "\\%")
                        body += f"$display(\"{indent}{code} => \", {value_name(instr)});\n"

                        body += f"if ({is_unknown(instr)}) begin\n"
                        for arg in instr.args:
                            if isinstance(arg, Instr):
                                gen_reason(arg, indent + "  ")
                        body += indent + "end\n"
                    
                    body += f"// Unknown guard for {args[0]}\n"
                    body += f"always @(posedge clock) "
                    body += f"if ({is_unknown(instr.args[0])}) begin\n"
                    body += f"$display(\"Error: {args[0]} is unknown: \", {args[0]});\n"
                    if isinstance(instr.args[0], Instr):
                        gen_reason(instr.args[0])
                    body += "#2;\n"
                    body += f"$finish;\n"
                    body += f"end\n"
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

def group_graph_to_dot(processor, show_args=False):
    processor.autoname()
    res = "digraph {\n"
    for group in processor.groups:
        res += f"{group.name} [label=\"{group.name}\", shape=box];\n"
    for group in processor.groups:
        if group.terminator is not None:
            for terminator in group.terminator.collect_always():
                assert isinstance(terminator, AlwaysTerminator)
                label = ""
                if show_args:
                    label = "\n".join([
                        f"{formal_arg.name} = {arg.format_arg()}"
                        for formal_arg, arg in zip(terminator.group.args, terminator.args)
                    ])
                res += f"{group.name} -> {terminator.group.name} [label=\"{label}\"];\n"
    res += "}\n"
    return res

def analyze_lifetimes(processor):
    """Returns a mapping from groups to the set of live values at the beginning of the group."""
    live = {}
    for group in processor.groups[::-1]:
        current_live = set()
        for next in group.terminator.collect_groups():
            if next in live:
                current_live = current_live.union(live[next])
        for instr in group.instrs[::-1]:
            if instr in current_live:
                current_live.remove(instr)
            for arg in instr.args:
                if isinstance(arg, Instr):
                    current_live.add(arg)
        live[group] = current_live
    assert len(live[processor.groups[0]]) == 0
    return live

def toposort(instrs):
    """Flatten the given instruction graph into a topological order."""

    def strip_wires(value):
        name = None
        while isinstance(value, Wire):
            if not value.is_autonamed():
                name = value.name
            value = value.get()
        if name is not None and value.is_autonamed():
            value.name = name
        return value

    stack = []
    for instr in instrs:
        stack.append((False, instr))
    order = []
    started = set()
    closed = set()
    while len(stack) > 0:
        emit, instr = stack.pop()
        if emit:
            order.append(instr)
            assert instr in started
            assert instr not in closed
            closed.add(instr)
        elif instr not in closed:
            assert instr not in started # Cycle Detection
            started.add(instr)
            stack.append((True, instr))
            instr.args = [strip_wires(arg) for arg in instr.args]
            for arg in instr.args:
                if isinstance(arg, Instr) and instr not in closed:
                    stack.append((False, arg))
    return order

def find_dominators(processor):
    """Returns a mapping from groups to the set of groups that dominate them."""
    dominators = {processor.groups[0]: {processor.groups[0]}}
    for group in processor.groups:
        if group not in dominators:
            continue # Unreachable
        for next in group.terminator.collect_groups():
            doms = set(dominators[group])
            doms.add(next)
            if next in dominators:
                dominators[next] = dominators[next].intersection(doms)
            else:
                dominators[next] = doms
    assert all(
        group in doms and processor.groups[0] in doms
        for group, doms in dominators.items()
    )
    return dominators

def build_stall_tree(terminator, stall): #, counters, regs, id, from_group):
    match terminator:
        case CommitTerminator():
            return Const(1, 0)
        case AlwaysTerminator(group=group):
            needs_stall = stall[group]
            if False and group in counters:
                needs_stall = Op(OpKind.Or, [
                    needs_stall,
                    Op(OpKind.Not, [
                        Op(OpKind.Eq, [
                            counters[group],
                            regs[from_group][id]
                        ])
                    ])
                ])
            return needs_stall
        case BranchTerminator(cond=cond, true=true, false=false):
            return Op(OpKind.Mux, [
                cond,
                build_stall_tree(true, stall),
                build_stall_tree(false, stall)
            ])

def create_id(processor, lifetimes):
    """Each execution is given an identifier for the purpose of ordering."""
    id_reg = processor.add_resource(RegisterResource(
        "id_reg",
        math.ceil(math.log2(len(processor.groups))) * 2, # *2 just to be safe.
        initial = 0
    ))
    id = Read(id_reg)
    id.name = "id"
    next_id = Op(OpKind.Add, [id, Const(id.width, 1)])
    processor.groups[0].add_all(id, next_id, Write(id_reg, next_id))

    # We need to ID to be live everywhere
    for group, live in lifetimes.items():
        if group != processor.groups[0]:
            live.add(id)

    return id

class Access:
    def __init__(self, is_write, resource, index, enable):
        self.is_write = is_write
        self.resource = resource
        self.index = index
        self.enable = enable

    def from_instr(instr):
        if isinstance(instr, Read) or isinstance(instr, Write):
            return Access(
                isinstance(instr, Write),
                instr.resource,
                instr.get_index(),
                instr.get_enable()
            )
        return None

def find_accesses(processor):
    """
    Returns a mapping from groups to a mapping from instructions to their accesses.
    For each group, it includes all accesses which may occur in that group or any
    of its (indirect) successors.
    """
    accesses = {}
    for group in processor.groups[::-1]:
        accesses[group] = {}
        for next in group.terminator.collect_groups():
            for instr, access in accesses[next].items():
                if instr not in accesses[group]:
                    accesses[group][instr] = access
        for instr in group.instrs:
            access = Access.from_instr(instr)
            if access is not None:
                accesses[group][instr] = access
    return accesses

def find_indirect_successors(processor):
    succs = {}
    for group in processor.groups[::-1]:
        succs[group] = {group}
        for next in group.terminator.collect_groups():
            succs[group] = succs[group].union(succs[next])
    return succs

def find_forwards(processor, indirect_successors):
    forwards = {
        group: {
            resource: {from_group: set() for from_group in processor.groups}
            for resource in processor.resources
        }
        for group in processor.groups
    }
    for group in processor.groups:
        for instr in group.instrs:
            if isinstance(instr, Forward):
                for succ in indirect_successors[group]:
                    forwards[succ][instr.resource][group].add(instr)
    return forwards

def stretch_lifetimes(processor, lifetimes, forwards):
    """Forwarded values need to be live in all indirect successors."""
    for group in processor.groups:
        for resource, from_groups in forwards[group].items():
            for from_group, instrs in from_groups.items():
                if from_group != group:
                    for forward in instrs:
                        def add(value):
                            if isinstance(value, Instr):
                                lifetimes[group].add(value)
                        
                        add(forward.get_index())
                        add(forward.get_value())
                        add(forward.get_enable())
    return lifetimes

def create_forward_muxes(processor, forward_paths):
    """
    Maps Read instructions to multiplexers which select the correct forwarded value.
    This results in constructs of the form:
        Mux(cond_0, forward_0, Mux(cond_1, forward_1, ... (forward_n, Read()) ...))
    """
    muxes = {}
    for read, paths in forward_paths.items():
        result = read
        for (cond, value) in paths[::-1]:
            result = Op(OpKind.Mux, [cond, value, result])
        if not read.is_autonamed():
            result.name = read.name
            read.name = ""
        muxes[read] = result
    return muxes

def find_parents(processor):
    parents = {}
    for group in processor.groups:
        for instr in group.instrs:
            parents[instr] = group
    return parents

def lower_pipeline(processor):
    parents = find_parents(processor)
    lifetimes = analyze_lifetimes(processor)
    dominators = find_dominators(processor)
    accesses = find_accesses(processor)

    # Create ID
    id = create_id(processor, lifetimes)

    # Forwards
    indirect_successors = find_indirect_successors(processor)
    forwards = find_forwards(processor, indirect_successors)
    lifetimes = stretch_lifetimes(processor, lifetimes, forwards)

    processor.autoname()

    regs = {}
    for group, live in lifetimes.items():
        regs[group] = {}
        for instr in live:
            assert isinstance(instr, Instr)
            assert len(instr.name) > 0
            regs[group][instr] = processor.add_resource(RegisterResource(
                group.name + "_" + instr.name,
                instr.width
            ))

    valid = {}
    stall = {}
    for group in processor.groups:
        valid[group] = processor.add_resource(RegisterResource(
            group.name + "_valid", 1, initial=0
        ))
        stall[group] = Wire(Const(1, 0))
        stall[group].name = group.name + "_stall"

    # Pipeline entry is always valid
    valid[processor.groups[0]].initial[0] = 1

    for group in processor.groups:
        stall[group].set(Op(OpKind.And, [
            build_stall_tree(group.terminator, stall),
            Read(valid[group])
        ]))

    side_effects = set()

    for group in processor.groups:
        for instr in group.instrs:
            if instr.width == 0:
                side_effects.add(instr)

    def value_in_group(value, group):
        assert not isinstance(value, Wire)
        if isinstance(value, Const):
            return value
        elif value in regs[group]:
            return Read(regs[group][value])
        elif isinstance(value, Instr) and parents[value] == group:
            return value
        elif isinstance(value, Op) and value.width == 1:
            args = []
            for arg in value.args:
                arg = value_in_group(arg, group)
                if arg is None:
                    return None
                args.append(arg)
            return Op(value.kind, args)
        else:
            return None

    def access_in_group(access, group):
        return Access(
            access.is_write,
            access.resource,
            value_in_group(access.index, group),
            value_in_group(access.enable, group)
        )

    # Maps Read instructions to a list of possible forwarding path. Each path is a tuple
    # of the form (condition, value). This is filled out during conflict analysis.
    forward_paths = {}

    for group in processor.groups:
        for other in processor.groups:
            if group == other:
                continue
            # can_conflict := group.valid & other.valid & other.id <_commit group.id
            # So: not(can_conflict)
            #     <- other.id >=_commit group.id
            #     <- other dominates group
            if other in dominators[group]:
                continue
            can_conflict = Op(OpKind.And, [
                Read(valid[group]),
                Read(valid[other])
            ])
            # If group dominates other -> other.id <_commit group.id
            if group not in dominators[other]:
                assert False # Not implemented
            
            any_conflict = Const(1, 0)

            # Writes from groups with an earlier id may conflict
            for instr in group.instrs:
                any_conflict_for_instr = Const(1, 0)

                for other_instr, other_access in accesses[other].items():
                    access = Access.from_instr(instr)
                    if access is not None and \
                       (access.is_write or other_access.is_write) and \
                       access.resource == other_access.resource:
                        other_access_known = access_in_group(other_access, other)
                        conflict = access.enable
                        if other_access_known.enable is not None:
                            conflict = Op(OpKind.And, [
                                conflict,
                                other_access_known.enable
                            ])
                        if other_access_known.index is not None:
                            conflict = Op(OpKind.And, [
                                conflict,
                                Op(OpKind.Eq, [
                                    other_access_known.index,
                                    access.index
                                ])
                            ])
                        any_conflict_for_instr = Op(OpKind.Or, [any_conflict_for_instr, conflict])
            
                if isinstance(instr, Read):
                    # However, if there is a forward for the exact index, there is no conflict
                    # TODO: It has to be the latest forward for this index
                    for from_group, instrs in forwards[other][instr.resource].items():
                        for forward in instrs:
                            index = value_in_group(forward.get_index(), other)
                            enable = value_in_group(forward.get_enable(), other)
                            value = value_in_group(forward.get_value(), other)

                            if index is None or enable is None or value is None:
                                continue

                            can_forward = Op(OpKind.And, [
                                can_conflict, # can_forward is also used for the forwarding muxes, so we need to include can_conflict here
                                Op(OpKind.And, [
                                    Op(OpKind.Eq, [index, instr.get_index()]),
                                    enable
                                ])
                            ])
                            can_forward.name = f"can_forward_{instr.name}_for_{instr.resource.name}_from_{other.name}_to_{group.name}"
                            any_conflict_for_instr = Op(OpKind.And, [
                                any_conflict_for_instr,
                                Op(OpKind.Not, [can_forward])
                            ])

                            if instr not in forward_paths:
                                forward_paths[instr] = []
                            forward_paths[instr].append((can_forward, value))

                any_conflict_for_instr.name = f"any_conflict_for_instr_{instr.name}_from_{other.name}_to_{group.name}"

                if not any_conflict_for_instr.is_const(0): # Some basic simplification in order to prevent generating too many instructions
                    any_conflict = Op(OpKind.Or, [any_conflict, any_conflict_for_instr])
            
            can_conflict = Op(OpKind.And, [can_conflict, any_conflict])
            
            stall[group].set(Op(OpKind.Or, [
                can_conflict,
                stall[group].get()
            ]))

    forward_muxes = create_forward_muxes(processor, forward_paths)
    print(forward_muxes)

    for group in processor.groups:
        substs = {instr: Read(resource) for instr, resource in regs[group].items()}

        for instr in group.instrs:
            instr.args = [substs[arg] if arg in substs else arg for arg in instr.args]
            if instr in forward_muxes:
                substs[instr] = forward_muxes[instr]
        
        for next in group.terminator.collect_groups(): # TODO
            side_effects.add(Write(
                valid[next],
                Op(OpKind.And, [Read(valid[group]), Op(OpKind.Not, [stall[group]])]),
                enable=Op(OpKind.Not, [stall[next]])
            ))
            for instr, resource in regs[next].items():
                if instr in substs:
                    instr = substs[instr]
                side_effects.add(Write(
                    resource,
                    instr,
                    enable=Op(OpKind.Not, [stall[next]])
                ))

    for group in processor.groups:
        enable = Op(OpKind.And, [
            Read(valid[group]),
            Op(OpKind.Not, [stall[group]])
        ])
        for instr in group.instrs:
            # TODO: Read
            if isinstance(instr, BaseWrite):
                instr.set_enable(Op(OpKind.And, [enable, instr.get_enable()]))

    for group, wire in stall.items():
        side_effects.add(UnknownGuard(wire.get()))

    group = Group()
    group.name = "pipeline"
    group.instrs = toposort(side_effects)
    group.terminator = CommitTerminator()
    processor.groups = [group]

if __name__ == "__main__":
    from parse_opcodes import InstEncoding
    from elftools.elf.elffile import ELFFile
    from elftools.elf.constants import P_FLAGS

    processor = Processor("Processor")
    builder = Builder(processor)

    STATE_RUNNING = builder.const(3, 0)
    STATE_EBREAK = builder.const(3, 1)

    pc = builder.resource(RegisterResource("pc", 32))
    cycle = builder.resource(RegisterResource("cycle", 32))
    state = builder.resource(RegisterResource("state", 3))
    reg_file = builder.resource(MemoryResource("reg_file", 32, size = 32, read_delay = 0))
    inst_mem = builder.resource(MemoryResource("inst_mem", 8, size = 1 << 24, read_delay = 0))
    data_mem = builder.resource(MemoryResource("data_mem", 8, size = 1 << 24, read_delay = 0))

    reg_file.init(0, index=0)
    cycle.init(0)
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
                mem.init(byte, index = addr)
                addr += 1

    fetch_group = builder.group("fetch")
    decode_group = builder.group("decode")
    execute_group = builder.group("execute")
    divide_group = builder.group("divide", ("count", 32), ("remainder", 64), ("dividend", 64), ("quotient", 32))
    memory_group = builder.group("memory")
    writeback_group = builder.group("writeback", ("rd_valid", 1), ("rd_res", 32))

    with fetch_group:
        state_value = state.read().name("state_value")
        is_running = (state_value == STATE_RUNNING).name("is_running")

        pc_value = pc.read().name("pc_value")
        inst = builder.concat(
            inst_mem.read(pc_value + 3),
            inst_mem.read(pc_value + 2),
            inst_mem.read(pc_value + 1),
            inst_mem.read(pc_value + 0)
        ).name("inst")

        pc.predict(pc_value + 4)

        cycle.write(cycle.read() + 1)

        builder.then(decode_group())

    with decode_group:
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

        rs1 = inst[15:20].name("rs1")
        rs2 = inst[20:25].name("rs2")
        rd = inst[7:12].name("rd")

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

        is_branch = is_inst["jal"] | \
                    is_inst["jalr"] | \
                    is_inst["ebreak"] | \
                    is_inst["beq"] | \
                    is_inst["bne"] | \
                    is_inst["blt"] | \
                    is_inst["bge"] | \
                    is_inst["bltu"] | \
                    is_inst["bgeu"]

        pc.forward(pc_value + 4, enable = is_running & ~is_branch)

        next_state, = builder.cond(
            (is_running & is_inst["ebreak"], STATE_EBREAK),
            (state_value,)
        )
        next_state.name("next_state")
        state_changed = (next_state != state_value).name("state_changed")
        state.forward(state_value, enable = ~state_changed)

        is_divide = is_inst["div"] | is_inst["divu"] | is_inst["rem"] | is_inst["remu"]
        is_divide.name("is_divide")

        builder.then(
            builder.branch(
                is_divide,
                divide_group(
                    32,
                    r1.z_ext(64),
                    r2.z_ext(64).shl(32),
                    builder.const(32, 0)
                ),
                execute_group()
            )
        )

    with execute_group:
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
        alu_valid.name("alu_valid")
        alu_res.name("alu_res")

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

        reg_file.forward(alu_res, index = rd, enable = is_running & alu_valid & (rd != 0))

        next_pc = branch.mux(branch_target, pc_value + 4)
        pc.write(next_pc, enable = is_running)

        builder.then(memory_group())

    with divide_group as (count, remainder, dividend, quotient):
        is_less = dividend.lt_u(remainder)
        next_dividend = is_less.mux(remainder - dividend, remainder)
        next_quotient = quotient.shl(1) | is_less.mux(1, builder.const(32, 0))

        result, = builder.cond(
            (is_inst["div"],   quotient),
            (is_inst["divu"],  quotient),
            (is_inst["rem"],   remainder[0:32]),
            (is_inst["remu"],  remainder[0:32]),
            (builder.const(32, 0), )
        )

        builder.then(builder.branch(
            count == 0,
            writeback_group(1, result),
            divide_group(count - 1, remainder.shr_u(1), next_dividend, quotient)
        ))

    with memory_group:
        addr = r1 + imm

        # For now we just assume that we have unlimited read/write ports, this is
        # not a problem which I am interested in solving right now.
        mem_valid, mem_res = builder.cond(
            (is_inst["lb"],  1, data_mem.read(addr).s_ext(32)),
            (is_inst["lh"],  1, builder.concat(
                data_mem.read(addr + 1),
                data_mem.read(addr + 0)
            ).s_ext(32)),
            (is_inst["lw"],  1, builder.concat(
                data_mem.read(addr + 3),
                data_mem.read(addr + 2),
                data_mem.read(addr + 1),
                data_mem.read(addr + 0)
            )),
            (is_inst["lbu"], 1, data_mem.read(addr).z_ext(32)),
            (is_inst["lh"],  1, builder.concat(
                data_mem.read(addr + 1),
                data_mem.read(addr + 0)
            ).z_ext(32)),
            (builder.const(1, 0), 0)
        )
        mem_valid.name("mem_valid")
        mem_res.name("mem_res")

        write_mask, = builder.cond(
            (is_inst["sb"], 1),
            (is_inst["sh"], 3),
            (is_inst["sw"], 15),
            (builder.const(4, 0),)
        )
        write_mask.name("write_mask")

        data_mem.write(r2[24:32], index = addr, enable = is_running & write_mask[3])
        data_mem.write(r2[16:24], index = addr, enable = is_running & write_mask[2])
        data_mem.write(r2[8:16], index = addr, enable = is_running & write_mask[1])
        data_mem.write(r2[0:8], index = addr, enable = is_running & write_mask[0])

        (rd_valid, rd_res) = builder.cond(
            (alu_valid, 1, alu_res),
            (mem_valid, 1, mem_res),
            (builder.const(1, 0), 0)
        )

        builder.then(writeback_group(rd_valid, rd_res))

    with writeback_group as (rd_valid, rd_res):
        reg_file.write(rd_res, index = rd, enable = is_running & rd_valid & (rd != 0))

        state.write(next_state, enable=state_changed)

        builder.then(builder.commit())

    with open(f"{processor.name}.gv", "w") as f:
        f.write(group_graph_to_dot(processor))

    print(processor.format(0))
    print({group.name: {instr.name for instr in live} for group, live in analyze_lifetimes(processor).items()})

    lower_pipeline(processor)
    print(processor.format(0))

    with open(f"{processor.name}.v", "w") as f:
        f.write(generate_verilog(processor))

