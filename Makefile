CXX = nvcc
CFLAGS = -O2 -I.
LDFLAGS = -lcudart -lcublas -lcublas

# Source files
SRCS = main.cu layernorm.cu gelu.cu softmax.cu attention.cu transformer.cu embedding.cu linear.cu

# Object files
OBJS = $(SRCS:.cu=.o)

# Executable name
EXEC = inference_engine

all: $(EXEC)

$(EXEC): $(OBJS)
	$(CXX) $(CFLAGS) -o $(EXEC) $(OBJS) $(LDFLAGS)

%.o: %.cu
	$(CXX) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(EXEC)
