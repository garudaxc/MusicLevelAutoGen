#include "hmm_viterbi_op.h"
#include "note_op_util.h"

void HMMViterbiStep(int frameIdx, int stateCount, 
    const unsigned int *tmStates, const unsigned int *tmPointers, const double *tmProbabilities, 
    const double *omDensities, int omDensitiesSliceSize, const unsigned int *omPointers, 
    const double *preViterbi, double *curViterbi, unsigned int *btPointers);

NoteHMMViterbiOp::NoteHMMViterbiOp(OpKernelConstruction *context): OpKernel(context) {

}

void NoteHMMViterbiOp::Compute(OpKernelContext *context) {

    // todo 支持多个batch

    const Tensor &stateCountTensor = context->input(0);
    int stateCount = stateCountTensor.flat<int32>().data()[0];

    const Tensor &tmStatesTensor = context->input(1);
    auto tmStates = tmStatesTensor.flat<uint32>().data();

    const Tensor &tmPointersTensor = context->input(2);
    auto tmPointers = tmPointersTensor.flat<uint32>().data();

    const Tensor &tmProbabilitiesTensor = context->input(3);
    auto tmProbabilities = tmProbabilitiesTensor.flat<double>().data();

    const Tensor &omDensitiesTensor = context->input(4);
    auto omDensities = omDensitiesTensor.flat<double>().data();
    int frameCount = omDensities.shape()[0];

    const Tensor &omPointersTensor = context->input(5);
    auto omPointers = omPointersTensor.flat<uint32>().data();

    const Tensor &initialDistributionTensor = context->input(6);
    auto initialDistribution = initialDistributionTensor.flat<double>().data();


    // todo allocate and copy from initialDistribution
    double *preViterbi = nullptr;
    double *curViterbi = nullptr;
    unsigned int *btPointers = nullptr;

    // todo output shape
    Tensor *pathTensor = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(), &pathTensor));
    Tensor *logProbabilityTensor = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(1, input_tensor.shape(), &logProbabilityTensor));
    
    int omDensitiesSliceSize = omDensities.shape()[1];
    for (int i = 0; i < frameCount; ++i) {
        HMMViterbiStep(i, stateCount, tmStates, tmPointers, tmProbabilities, 
            omDensities, omDensitiesSliceSize, omPointers, preViterbi, curViterbi, btPointers);
    }

    // template flat<int32>(); ??
    auto path = pathTensor->template flat<int32>().data();
    auto logProbability = logProbabilityTensor->template flat<double>().data();
    
    int state = note_op_util::ArgMax(curViterbi, stateCount);
    *logProbability = curViterbi[state];
    for (int i = frameCount - 1; i >= 0; --i) {
        path[i] = state;
        state = btPointers[i * stateCount + state];
    }
}