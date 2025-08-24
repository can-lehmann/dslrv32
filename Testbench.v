module Testbench;
  reg clock = 0;
  always #1 clock <= !clock;

  Processor processor(
    .clock(clock)
  );

  initial
    begin
      $dumpfile("waveform.fst");
      $dumpvars(0, Testbench);
      #100
      $finish;
    end
endmodule
