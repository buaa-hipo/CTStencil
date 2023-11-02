CC = sw5cc
CCFLAGS = -O2
OUTPUT = *.out vn* .tune_err

.PHONY: mpe
mpe:
	$(CC) -host -o mpe $(CCFLAGS) mpe.c -lm

clean:
	rm -f $(OUTPUT)
