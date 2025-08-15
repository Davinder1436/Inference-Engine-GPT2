#ifndef EMBEDDING_H
#define EMBEDDING_H

#include "common.h"

void embedding_forward(float* out, int* tokens, float* wte, float* wpe, int B, int T, int C);

#endif // EMBEDDING_H
