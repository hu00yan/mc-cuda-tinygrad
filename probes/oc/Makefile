# probe 1: the semantic camera. See ../../notes/01-obs-camera.md
SEEDS ?= 0 3 12345
CORE  := ../../netherite/blaze/core
PY    := ../../.venv/bin/python

ref_oc: ref_oc.c
	clang -O2 -ffp-contract=off -I$(CORE) ref_oc.c -o ref_oc -lm

cells: ref_oc
	@for s in $(SEEDS) 1 2 4 5 6 7; do ./ref_oc --dump-cells cells_$$s.bin --seeds $$s --quiet; done

ref.bin: ref_oc
	./ref_oc --out ref.bin --seeds 0,3,12345 --poses 6

report: ref.bin cells
	$(PY) oc_report.py

check: ref.bin cells
	$(PY) run_oc.py --seeds 0,3,12345 --check-steps

clean:
	rm -f ref_oc ref.bin cells_*.bin
	rm -rf __pycache__
