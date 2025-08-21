# MIT License
# 
# Copyright (c) 2023 Can Joshua Lehmann
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# Script for generating the instruction decoder

from dataclasses import dataclass
import os, os.path

def read_file(path):
    with open(path, "r") as f:
        return f.read()

@dataclass
class InstEncoding:
    name: str
    mask: int
    pattern: int
    args: list[str]
    
    def load(path):
        insts = []
        for line in read_file(path).split("\n"):
            if len(line) == 0 or line.startswith("#") or line.startswith("$"):
                continue
            
            parts = [part for part in line.split(" ") if len(part) > 0]
            
            name = parts[0]
            pattern = 0
            mask = 0
            args = []
            for part in parts[1:]:
                if "=" in part:
                    lhs, rhs = part.split("=")
                    
                    if ".." in lhs:
                        msb, lsb = lhs.split("..")
                        offset = int(lsb)
                        size = int(msb) - offset + 1
                    else:
                        offset = int(lhs)
                        size = 1
                    
                    local_mask = 2 ** size - 1
                    mask |= local_mask << offset
                    
                    if rhs.startswith("0x"):
                        val = int(rhs[2:], 16)
                    elif rhs.startswith("0b"):
                        val = int(rhs[2:], 2)
                    else:
                        val = int(rhs)
                    
                    assert (val & local_mask) == val
                    pattern |= val << offset
                else:
                    args.append(part)
            
            insts.append(InstEncoding(
                name=name,
                mask=mask,
                pattern=pattern,
                args=args
            ))
        return insts
    
    def load_dir(path, prefix="rv"):
        insts = []
        for base, dirs, files in os.walk(path):
            for file_name in files:
                if file_name.startswith(prefix):
                    insts += InstEncoding.load(os.path.join(base, file_name))
        return insts
