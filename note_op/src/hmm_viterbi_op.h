#ifndef HMM_VITERBI_OP_H
#define HMM_VITERBI_OP_H

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"

class NoteHMMViterbiOp : public tensorflow::OpKernel {
public:
    explicit NoteHMMViterbiOp(tensorflow::OpKernelConstruction *context);
    void Compute(tensorflow::OpKernelContext *context) override;
};

#endif
