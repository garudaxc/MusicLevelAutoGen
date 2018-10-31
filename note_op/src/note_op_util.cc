#include "note_op_util.h"

int note_op_util::ArgMax(const double *data, int count) {
    // todo 这个能用cuda实现？
    double maxValue = -100000.0;
    int maxIdx = 0;
    for (int i = 0; i < count; ++i) {
        if (maxValue < data[i]) {
            maxValue = data[i];
            maxIdx = i;
        }
    }
    return maxIdx;
}
