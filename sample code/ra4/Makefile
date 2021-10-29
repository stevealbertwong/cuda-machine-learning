CXX=nvcc
CXXFLAGS=--gpu-architecture=compute_75 --gpu-code=sm_75
SRC=RA4.cu
EXE=RA4

% : %.cu
	$(CXX) $(CXXFLAGS) -o $@ $<

.PHONY: run clean

run: $(EXE)
	srun --partition=CME --gres=gpu:1 $(EXE)

clean:
	$(RM) $(EXE)
