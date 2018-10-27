#ifndef HMM_VITERBI_OP_H
#define HMM_VITERBI_OP_H

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"

class NoteHMMViterbiOp : public OpKernel {
public:
    explicit NoteHMMViterbiOp(OpKernelConstruction *context);
    void Compute(OpKernelContext *context) override;
};

#endif
