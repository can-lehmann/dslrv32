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
    "Mux"
])

def infer_op_width(kind, args):
    match kind:
        case OpKind.Add | OpKind.Sub | OpKind.Mul | OpKind.And | OpKind.Or | OpKind.Xor | OpKind.Shl | OpKind.ShrU | OpKind.ShrS:
            assert len(args) == 2
            assert args[0] == args[1]
            return args[0]
        case OpKind.Not:
            return args[0]
        case OpKind.Eq | OpKind.LtU | OpKind.LtS:
            assert len(args) == 2
            assert args[0] == args[1]
            return 1
        case OpKind.Mux:
            assert len(args) == 3
            assert args[1] == args[2]
            assert args[0] == 1
            return args[1]
        case _:
            assert False

class Op(Instr):
    def __init__(self, kind, args):
        super().__init__(infer_op_width(kind, [arg.width for arg in args]), args)

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

    def op(self, kind, *args):
        op = Op(kind, [self.value, *(arg.value for arg in args)])
        self.builder.emit(op)
        return ValueBuilder(self.builder, op)

    def __add__(self, other):
        return self.op(OpKind.Add, other)

    def __sub__(self, other):
        return self.op(OpKind.Sub, other)

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
    processor = Processor("rv32")
    builder = Builder(processor)

    pc = builder.resource(Resource("pc", 32))
    reg_file = builder.resource(Resource("reg_file", 32))
    inst_mem = builder.resource(Resource("inst_mem", 32))
    data_mem = builder.resource(Resource("data_mem", 32))

    with builder.group("fetch"):
        pc_value = pc.read()
        inst = inst_mem.read(pc_value)

        pc.predict(pc_value + builder.const(32, 4))

    with builder.group("decode"):
        pass

    with builder.group("execute"):
        pass
        # pc.write()

    with builder.group("memory"):
        pass

    with builder.group("writeback"):
        pass
        #reg_file.write()

    print(processor.format(0))
