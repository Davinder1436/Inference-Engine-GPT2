CXX = nvcc
CFLAGS = -O2 -I.
LDFLAGS = -lcudart -lcublas

# Source files for main program
SRCS = main.cu layernorm.cu gelu.cu softmax.cu attention.cu transformer.cu embedding.cu linear.cu attention_softmax.cu
OBJS = $(SRCS:.cu=.o)
EXEC = inference_engine

# Source files for test program  
TEST_SRCS = test_main.cu layernorm.cu embedding.cu
TEST_OBJS = $(TEST_SRCS:.cu=.o)
TEST_EXEC = test_inference

all: $(EXEC)

test: $(TEST_EXEC)

$(EXEC): $(OBJS)
	$(CXX) $(CFLAGS) -o $(EXEC) $(OBJS) $(LDFLAGS)

$(TEST_EXEC): $(TEST_OBJS)
	$(CXX) $(CFLAGS) -o $(TEST_EXEC) $(TEST_OBJS) $(LDFLAGS)

%.o: %.cu
	$(CXX) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(TEST_OBJS) $(EXEC) $(TEST_EXEC)

.PHONY: all test clean
