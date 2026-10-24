#!/usr/bin/env bash
# Dispatch the arm-B T4 run and WAIT for it to finish. gpush's `published bytes`
# gate refuses when the working tree moves between the manifest and the upload,
# so nothing may be edited while a push is in flight -- hence this is a script
# that blocks, not a fire-and-forget background call.
cd /Users/huyan00/mycode/mc-cuda-tinygrad
./minecu_tg_knobs/kaggle/run.sh --session 35 --queue-wait 40 --poll 20 \
  > out/minecu_tg_knobs/armB/t4_runN.log 2>&1
echo "EXIT=$?" >> out/minecu_tg_knobs/armB/t4_runN.log
