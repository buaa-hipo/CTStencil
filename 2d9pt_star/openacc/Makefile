CC = swacc
CCFLAGS = -O2
OUTPUT = *.out vn* .tune_err

.PHONY: openacc
openacc:
	$(CC) -o openacc $(CCFLAGS) openacc.c -lm

clean:
	rm -f $(OUTPUT)
