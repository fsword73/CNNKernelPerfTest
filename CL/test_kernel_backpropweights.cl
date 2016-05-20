// Copyright Hugh Perkins 2014,2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

// expected defines:
// BIASED (or not)

// globalId: [outPlane][inputPlane][filterRow][filterCol]
// per-thread iteration: [n][outputRow][outputCol]

//DeepCL / src / conv / BackpropWeightsScratch.cpp 
//27     int workgroupsize = std::max(32, square(dim.filterSize) ); // no point in wasting cores... 
//28     int numWorkgroups = dim.inputPlanes * dim.numFilters; 
//29     int globalSize = workgroupsize * numWorkgroups; 
//30     globalSize = (( globalSize + workgroupsize - 1) / workgroupsize) * workgroupsize; 


#define 	gNumFilters 	8
#define 	gInputPlanes	4
#define   gFilterSize   3 
#define   gFilterSizeSquared (gFilterSize*gFilterSize)
//#define   gOutputSize   128 
#define   gInputSize    128 
#define   gMargin        0
#if 1 
void __kernel test_kernel(				
__global const float* filter,
						__global const float* gradOutput,
						__global float* images,	
						__global const float* dataBuf3,
						__global const float* dataBuf4,
        #ifdef BIASED
             __global float *gradBiasWeights,
				#else 
					 __global const float* dataBuf5, 
        #endif						
						__global float* gradWeights,
						const int batchSize,  
						const int gOutputSize,  
						const int const3,  
						const int const4,  
						const int const5,  
						const int const6
 ) 
#else
void kernel backprop_floats(const float learningRateMultiplier,
        const int batchSize, 
         global const float *gradOutput, global const float *images, 
        global float *gradWeights,
				const int gOutputSize
        #ifdef BIASED
            , global float *gradBiasWeights
        #endif
				
 ) 
 #endif
 {
	 	
	 
    int globalId = get_global_id(0);
    if (globalId >= gNumFilters * gInputPlanes * gFilterSize * gFilterSize) {
        return;
    }
		
		const float learningRateMultiplier = 0.0001f;

    int IntraFilterOffset = globalId % gFilterSizeSquared;
    int filterRow = IntraFilterOffset / gFilterSize;
    int filterCol = IntraFilterOffset % gFilterSize;

    int filter2Id = globalId / gFilterSizeSquared;
    int outPlane = filter2Id / gInputPlanes;
    int upstreamPlane = filter2Id % gInputPlanes;

    float thiswchange = 0;
		
	
    // gradWeights:     [outPlane][upstreamPlane][filterRow][filterCol]
    //       aggregate over:  [outRow][outCol][n]
#ifdef BIASED
    float thisbiaschange = 0;
#endif
    for (int n = 0; n < batchSize; n++) {
        for (int outRow = 0; outRow < gOutputSize; outRow++) {
            int upstreamRow = outRow - gMargin + filterRow;
            for (int outCol = 0; outCol < gOutputSize; outCol++) {
                int upstreamCol = outCol - gMargin + filterCol;
                bool proceed = upstreamRow >= 0 && upstreamCol >= 0 && upstreamRow < gInputSize
                    && upstreamCol < gInputSize;
                if (proceed) {
                    int resultIndex = (( n * gNumFilters 
                              + outPlane) * gOutputSize
                              + outRow) * gOutputSize
                              + outCol;
                    float error = gradOutput[resultIndex];
                    int upstreamDataIndex = (( n * gInputPlanes 
                                     + upstreamPlane) * gInputSize
                                     + upstreamRow) * gInputSize
                                     + upstreamCol;
                    float upstreamResult = images[upstreamDataIndex];
                    float thisimagethiswchange = upstreamResult * error;
                    thiswchange += thisimagethiswchange;
    #ifdef BIASED
                    thisbiaschange += error;
    #endif
                }
            }
        }
    }
    // gradWeights:     [outPlane][upstreamPlane][filterRow][filterCol]
    //       aggregate over:  [outRow][outCol][n]
    gradWeights[ globalId ] = learningRateMultiplier * thiswchange;
#ifdef BIASED
    bool writeBias = upstreamPlane == 0 && filterRow == gMargin && filterCol == gMargin;
    if (writeBias) {
        gradBiasWeights[outPlane] = learningRateMultiplier * thisbiaschange;
    }
#endif
}

// JIAN DOT YANG AT AMD DOT COM
//CNNBench command line 
//-dim 1 -lx 64 -ly 1 -gx 288 -gy 1 -f 3 -c1 128 -c2 128 -i 1 -x 4096 -y 4096
	//    Input 	128x128, 4 Planes, BatchSize 128 
  //    output  128x128, 8 planes,  BatchSize 128    
  //    Filter Size = 3x3 
  // globalId: [outPlane][inputPlane][filterRow][filterCol]   8*4*3*3
  // per-thread iteration: [n][outputRow][outputCol]
	//    	    Input 	= 128x128x4x128 batch = 4096 * 2048
  //			    output  = 128x128x8x128 batch = 4096 * 4096  
	//          force localthread_x =64,  globalthread_y =1
	//          FilterSize =3,  
	//         -constant1 = batchSize 128
	//         -constant2 = 3     filterSize;

