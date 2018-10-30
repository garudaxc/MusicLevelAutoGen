__global__ void HMMViterbiStateKernel(int frameIdx, int stateCount, 
    const unsigned int *tmStates, const unsigned int *tmPointers, const double *tmProbabilities, 
    const double *omDensities, int omDensitiesSliceSize, const unsigned int *omPointers, 
    const double *preViterbi, double *curViterbi, unsigned int *btPointers) {
    // omDensities， btPointers 这些每次都整块传进来？
    int state = blockIdx.x * blockDim.x + threadIdx.x;
    if (state < stateCount) {
        int startIdx = tmPointers[state];
        int endIdx = tmPointers[state + 1];
        unsigned int preState = 0;
        unsigned int maxTransitionProbPreState = 0;
        double transitionProb = 0;
        // todo send by c++ std
        double maxTransitionProb = -1000000.0;
        for (int i = startIdx; i < endIdx; ++i) {
            preState = tmStates[i];
            transitionProb = preViterbi[preState] + tmProbabilities[i];
            if (transitionProb > maxTransitionProb) {
                maxTransitionProb = transitionProb;
                maxTransitionProbPreState = preState;
            }
        }
        double density = omDensities[frameIdx * omDensitiesSliceSize + omPointers[state]];
        curViterbi[state] = maxTransitionProb + density;
        btPointers[frameIdx * stateCount + state] = maxTransitionProbPreState;
    }
}

void HMMViterbiStep(int frameIdx, int stateCount, 
    const unsigned int *tmStates, const unsigned int *tmPointers, const double *tmProbabilities, 
    const double *omDensities, int omDensitiesSliceSize, const unsigned int *omPointers, 
    const double *preViterbi, double *curViterbi, unsigned int *btPointers) {

    dim3 blockSize(128);
    // 如果正好整除，多一个block有多少影响？
    int blockCount = stateCount / blockSize.x + 1;
    dim3 gridSize(blockCount);
    HMMViterbiStateKernel<<<gridSize, blockCount>>>(frameIdx, stateCount, tmStates, tmPointers, tmProbabilities, 
                omDensities, omDensitiesSliceSize, omPointers, preViterbi, curViterbi, btPointers);
    cudaDeviceSynchronize();
}
