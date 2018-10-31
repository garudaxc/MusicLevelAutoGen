#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "hmm_viterbi_op.h"

REGISTER_OP("NoteHMMViterbi")
    .Input("state_count: int32")
    .Input("tm_states: uint32")
    .Input("tm_pointers: uint32")
    .Input("tm_probabilities: double")
    .Input("om_densities: double")
    .Input("om_pointers: uint32")
    .Input("initial_distribution: double")
    .Output("path: uint32")
    .Output("log_probability: double")
    .Doc(R"doc(native implement of madmom hmm viterbi, support cuda only now.)doc");
REGISTER_KERNEL_BUILDER(Name("NoteHMMViterbi")
    .Device(tensorflow::DEVICE_GPU)
    .HostMemory("state_count")
    , NoteHMMViterbiOp);