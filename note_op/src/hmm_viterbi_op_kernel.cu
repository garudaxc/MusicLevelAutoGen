__global__ void HMMViterbiStateKernel(int frameIdx, int stateCount, double minProb, 
    const unsigned int *tmStates, const unsigned int *tmPointers, const double *tmProbabilities, 
    const double *omDensities, int omDensitiesSliceSize, const unsigned int *omPointers, 
    const double *preViterbi, double *curViterbi, unsigned int *btPointers) {
    
    int state = blockIdx.x * blockDim.x + threadIdx.x;
    if (state < stateCount) {
        int startIdx = tmPointers[state];
        int endIdx = tmPointers[state + 1];
        unsigned int preState = 0;
        unsigned int maxTransitionProbPreState = 0;
        double transitionProb = 0;
        double maxTransitionProb = minProb;
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

__global__ void HMMCopyKernel(double *dst, const double * src, int count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        dst[idx] = src[idx];
    }
}

__global__ void HMMViterbiKernel(int frameCount, int stateCount, double minProb, 
    const unsigned int *tmStates, const unsigned int *tmPointers, const double *tmProbabilities, 
    const double *omDensities, int omDensitiesSliceSize, const unsigned int *omPointers, 
    double *preViterbi, const double *initialDistribution, double *curViterbi, unsigned int *btPointers, 
    unsigned int *path, double *logProbability) {

    dim3 blockSize(128);
    // 如果正好整除，多一个block有多少影响？
    dim3 gridSize(static_cast<int>(stateCount / blockSize.x + 1));
    HMMCopyKernel<<<gridSize, blockSize>>>(preViterbi, initialDistribution, stateCount);
    for (int i = 0; i < frameCount; ++i) {
        HMMViterbiStateKernel<<<gridSize, blockSize>>>(i, stateCount, minProb, tmStates, tmPointers, tmProbabilities, omDensities, omDensitiesSliceSize, omPointers, preViterbi, curViterbi, btPointers);
        HMMCopyKernel<<<gridSize, blockSize>>>(preViterbi, curViterbi, stateCount);
    }
    cudaDeviceSynchronize();

    double maxValue = -100000.0;
    int maxIdx = 0;
    for (int i = 0; i < stateCount; ++i) {
        if (maxValue < curViterbi[i]) {
            maxValue = curViterbi[i];
            maxIdx = i;
        }
    }
    int state = maxIdx;
    *logProbability = curViterbi[state];
    for (int i = frameCount - 1; i >= 0; --i) {
        path[i] = state;
        state = btPointers[i * stateCount + state];
    }
}

void HMMViterbi(int frameCount, int stateCount, double minProb, 
    const unsigned int *tmStates, const unsigned int *tmPointers, const double *tmProbabilities, 
    const double *omDensities, int omDensitiesSliceSize, const unsigned int *omPointers, 
    double *preViterbi, const double *initialDistribution, double *curViterbi, unsigned int *btPointers, 
    unsigned int *path, double *logProbability) {

    dim3 blockSize(1);
    dim3 gridSize(1);

    HMMViterbiKernel<<<gridSize, blockSize>>>(frameCount, stateCount, minProb, tmStates, tmPointers, tmProbabilities, omDensities, omDensitiesSliceSize, omPointers, preViterbi, initialDistribution, curViterbi, btPointers, path, logProbability);
}