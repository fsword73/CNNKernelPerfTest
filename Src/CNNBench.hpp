#ifndef CNN_BENCH_H_
#define CNN_BENCH_H_


#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>


#include "CLUtil.hpp"

using namespace appsdk;

#define SAMPLE_VERSION "AMD-APP-SDK-v2.9-1.599.2"


#define WIDTH 2048
#define HEIGHT 2048

class CNNBench {
	float*		pdata1;
	float*		pdata2;
	float*		pdata3;
	float*		pdata4;
	float*		pdata5;
	float*		pdata6;
	cl_int      filterSize;
	float*     filter;
	cl_int      const1;
	cl_int      const2;
	cl_int      const3;
	cl_int      const4;
	cl_int      const5;
	cl_int      const6;
	

	cl_int width;               /**< width of the input */
	cl_int height;              /**< height of the input */
	cl_int iterations;			/**< iterations to run the kernel */

	cl_double setupTime;        /**< time taken to setup OpenCL resources and building kernel */
	cl_double kernelTime;       /**< time taken to run kernel and read result back */

	cl_ulong totalLocalMemory;      /**< Max local memory allowed */
	cl_ulong usedLocalMemory;       /**< Used local memory by kernel */

	cl_context context;             /**< CL context */
	cl_device_id *devices;          /**< CL device list */

	cl_command_queue commandQueue;  /**< CL command queue */
	cl_program program;             /**< CL program  */
	cl_kernel  kernel;              /**< CL kernel   */

	SDKDeviceInfo         deviceInfo;            /**< Structure to store device information*/
	KernelWorkGroupInfo   kernelInfo;      		 /**< Structure to store kernel related info */

	cl_mem dataBuf1;
	cl_mem dataBuf2;
	cl_mem dataBuf3;
	cl_mem dataBuf4;
	cl_mem dataBuf5;
	cl_mem dataBuf6;
	cl_mem filterBuf;

	size_t globalThreads[3];
	size_t localThreads[3];
	cl_uint dimension;

	SDKTimer    *sampleTimer;      /**< SDKTimer object */

public:

	CLCommandArgs   *sampleArgs;   /**< CLCommand argument class */
								   /**
								   * Constructor
								   * Initialize member variables
								   * @param name name of sample (string)
								   */
	CNNBench()
		:
		setupTime(0),
		kernelTime(0),
		devices(NULL),
		pdata1(NULL),
		pdata2(NULL),
		pdata3(NULL),
		pdata4(NULL),
		pdata5(NULL),
		pdata6(NULL),
		filter(NULL)
	{
		/* Set default values for width and height */
		width = WIDTH;
		height = HEIGHT;
		iterations = 200;
		sampleArgs = new CLCommandArgs();
		sampleTimer = new SDKTimer();
		sampleArgs->sampleVerStr = SAMPLE_VERSION;

		dimension = 2;
		globalThreads[0] = width;
		globalThreads[1] = height;
		globalThreads[2] = 1;

		localThreads[0] = 8;
		localThreads[1] = 8;
		localThreads[2] = 1;

		const1 = 0;
		const2 = 0;
		const3 = 0;
		const4 = 0;
		const5 = 0;
		const6 = 0;

	}



	/**
	* Override from SDKSample. Initialize
	* command line parser, add custom options
	* @return SDK_SUCCESS on success and SDK_FAILURE on failure
	*/
	int initialize();

	/**
	* Override from SDKSample, adjust width and height
	* of execution domain, perform all sample setup
	* @return SDK_SUCCESS on success and SDK_FAILURE on failure
	*/
	int setup();

	/**
	* Override from SDKSample
	* Run OpenCL Black-Scholes
	* @return SDK_SUCCESS on success and SDK_FAILURE on failure
	*/
	int run();

	/**
	* Override from SDKSample
	* Cleanup memory allocations
	* @return SDK_SUCCESS on success and SDK_FAILURE on failure
	*/
	int cleanup();
private:
	int SavePNG(const char* filename);
	int setupCL();
	int SetupKernels();
	int SetupBuffers();	
	int SetupFilter(int filterSize);
	int InitChessBoard();

	/**
	* clEnqueueMapBuffer
	* @return SDK_SUCCESS on success and SDK_FAILURE on failure
	*/
	template<typename T>
	int mapBuffer(cl_mem deviceBuffer, T* &hostPointer, size_t sizeInBytes,
		cl_map_flags flags = CL_MAP_READ);

	/**
	* clEnqueueUnmapMemObject
	* @return SDK_SUCCESS on success and SDK_FAILURE on failure
	*/
	int unmapBuffer(cl_mem deviceBuffer, void* hostPointer);

};



#endif