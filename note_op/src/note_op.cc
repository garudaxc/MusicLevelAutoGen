#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "hmm_viterbi_op.h"

REGISTER_OP("NoteHMMViterbi")
    .Input("stateCount: int32")
    .Input("tmStates: uint32")
    .Input("tmPointers: uint32")
    .Input("tmProbabilities: float64")
    .Input("omDensities: float64")
    .Input("omPointers: uint32")
    .Input("initialDistribution: float64")
    .Output("path: uint32")
    .Output("logProbability: float64")
    .Doc(R"doc(native implement of madmom hmm viterbi, support cuda only now.)doc");
REGISTER_KERNEL_BUILDER(Name("NoteHMMViterbi").Device(tensorflow::DEVICE_GPU), NoteHMMViterbiOp);