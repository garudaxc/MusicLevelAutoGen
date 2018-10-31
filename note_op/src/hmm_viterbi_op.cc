#include "hmm_viterbi_op.h"
#include "note_op_util.h"

using namespace tensorflow;

void HMMViterbi(int frameCount, int stateCount, double minProb, 
    const unsigned int *tmStates, const unsigned int *tmPointers, const double *tmProbabilities, 
    const double *omDensities, int omDensitiesSliceSize, const unsigned int *omPointers, 
    double *preViterbi, const double *initialDistribution, double *curViterbi, unsigned int *btPointers, 
    unsigned int *path, double *logProbability);

NoteHMMViterbiOp::NoteHMMViterbiOp(OpKernelConstruction *context): OpKernel(context) {

}

void NoteHMMViterbiOp::Compute(OpKernelContext *context) {
    const Tensor &stateCountTensor = context->input(0);
    int stateCount = stateCountTensor.scalar<int32>()();

    const Tensor &tmStatesTensor = context->input(1);
    auto tmStates = tmStatesTensor.flat<uint32>().data();

    const Tensor &tmPointersTensor = context->input(2);
    auto tmPointers = tmPointersTensor.flat<uint32>().data();

    const Tensor &tmProbabilitiesTensor = context->input(3);
    auto tmProbabilities = tmProbabilitiesTensor.flat<double>().data();

    const Tensor &omDensitiesTensor = context->input(4);
    auto omDensities = omDensitiesTensor.flat<double>().data();
    const TensorShape &omDensitiesTensorShape = omDensitiesTensor.shape();
    int frameCount = omDensitiesTensorShape.dim_size(0);

    const Tensor &omPointersTensor = context->input(5);
    auto omPointers = omPointersTensor.flat<uint32>().data();

    const Tensor &initialDistributionTensor = context->input(6);
    auto initialDistribution = initialDistributionTensor.flat<double>().data();

    Tensor preViterbiTensor = Tensor();
    OP_REQUIRES_OK(context, context->allocate_temp(DT_DOUBLE, TensorShape({stateCount}), &preViterbiTensor));
    double *preViterbi = preViterbiTensor.template flat<double>().data();

    Tensor curViterbiTensor = Tensor();
    OP_REQUIRES_OK(context, context->allocate_temp(DT_DOUBLE, TensorShape({stateCount}), &curViterbiTensor));
    double *curViterbi = curViterbiTensor.template flat<double>().data();

    Tensor btPointersTensor = Tensor();
    OP_REQUIRES_OK(context, context->allocate_temp(DT_UINT32, TensorShape({frameCount, stateCount}), &btPointersTensor));
    unsigned int *btPointers = btPointersTensor.template flat<uint32>().data();

    Tensor *pathTensor = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape({frameCount}), &pathTensor));

    Tensor *logProbabilityTensor = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(1, TensorShape({}), &logProbabilityTensor));
    
    int omDensitiesSliceSize = omDensitiesTensorShape.dim_size(1);
    auto path = pathTensor->template flat<uint32>().data();
    auto logProbability = logProbabilityTensor->template flat<double>().data();
    double minProb = std::numeric_limits<double>::min();
    HMMViterbi(frameCount, stateCount, minProb, tmStates, tmPointers, tmProbabilities, omDensities, omDensitiesSliceSize, omPointers, preViterbi, initialDistribution, curViterbi, btPointers, path, logProbability);
}