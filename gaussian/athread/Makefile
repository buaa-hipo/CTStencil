CC = sw5cc
CFLAGS = -O2
OUTPUT = *.out vn* .tune_err
LINK_SPC = -Wl,--whole-archive,-wrap,athread_init,-wrap,__expt_handler,-wrap,__real_athread_spawn /home/export/online3/swmore/release/lib/libspc.a -Wl,--no-whole-archive
LINK_PERF = -lswperf

SPE_OBJS = $(patsubst %.c, %.o, $(wildcard athread_spe_?.c))
EXCUTABALES = $(patsubst athread_spe_%.c, athread_%, $(wildcard athread_spe_?.c))

.PHONY: $(EXCUTABALES)
$(EXCUTABALES):athread_%:athread_ppe.o athread_spe_%.o
	$(CC) -hybrid $(CFLAGS) $^ -o $@ $(LINK_SPC) $(LINK_PERF)

.PHONY: athread_ppe.o
athread_ppe.o: athread_ppe.c
	$(CC) -host -c $^ -o $@

.PHONY: $(SPE_OBJS)
$(SPE_OBJS):%.o:%.c
	$(CC) -slave -c $^ -msimd -o $@

clean:
	rm -f $(OUTPUT)
