__asm__(
  ".globl _start\n"
  "_start:\n"
  "nop\n"
  "nop\n"
  "nop\n"
  "call main\n"
  "ebreak"
);

volatile int* out = (void*)0x100;

void main() {
  for (int i = 0; i < 10; i++) {
    *out = '0' + i;
  }
}
