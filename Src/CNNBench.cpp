#include <windows.h>
#include <gdiplus.h>
#include <stdio.h>
using namespace Gdiplus;

#include "CNNBench.hpp"
#include <math.h>

#pragma warning(disable:4244)  //Warning	C4244	'=': conversion from 'int' to 'float', possible loss of data
#pragma comment(lib, "Gdiplus.lib")

template<typename T>
int CNNBench::mapBuffer(cl_mem deviceBuffer, T* &hostPointer,
	size_t sizeInBytes, cl_map_flags flags)
{
	cl_int status;
	hostPointer = (T*)clEnqueueMapBuffer(commandQueue,
		deviceBuffer,
		CL_TRUE,
		flags,
		0,
		sizeInBytes,
		0,
		NULL,
		NULL,
		&status);
	CHECK_OPENCL_ERROR(status, "clEnqueueMapBuffer failed");

	return SDK_SUCCESS;
}

int
CNNBench::unmapBuffer(cl_mem deviceBuffer, void* hostPointer)
{
	cl_int status;
	status = clEnqueueUnmapMemObject(commandQueue,
		deviceBuffer,
		hostPointer,
		0,
		NULL,
		NULL);
	CHECK_OPENCL_ERROR(status, "clEnqueueUnmapMemObject failed");

	return SDK_SUCCESS;
}


int CNNBench::initialize()
{
	// Call base class Initialize to get default configuration
	if (sampleArgs->initialize() != SDK_SUCCESS)
	{
		return SDK_FAILURE;
	}

	Option* width_option = new Option;
	CHECK_ALLOCATION(width_option, "Memory allocation error.\n");

	width_option->_sVersion = "x";
	width_option->_lVersion = "width";
	width_option->_description = "Width of the input";
	width_option->_type = CA_ARG_INT;
	width_option->_value = &width;

	sampleArgs->AddOption(width_option);
	delete width_option;

	Option* height_option = new Option;
	CHECK_ALLOCATION(height_option, "Memory allocation error.\n");

	height_option->_sVersion = "y";
	height_option->_lVersion = "height";
	height_option->_description = "Height of the input";
	height_option->_type = CA_ARG_INT;
	height_option->_value = &height;

	sampleArgs->AddOption(height_option);
	delete height_option;

	Option* filtersize_option = new Option;
	CHECK_ALLOCATION(filtersize_option, "Memory allocation error.\n");

	height_option->_sVersion = "f";
	height_option->_lVersion = "filterSize";
	height_option->_description = "filterSize of the input";
	height_option->_type = CA_ARG_INT;
	height_option->_value = &filterSize;

	sampleArgs->AddOption(filtersize_option);
	delete filtersize_option;

	Option* iteration_option = new Option;
	CHECK_ALLOCATION(iteration_option, "Memory allocation error.\n");

	iteration_option->_sVersion = "i";
	iteration_option->_lVersion = "iterations";
	iteration_option->_description = "Number of iterations to execute kernel";
	iteration_option->_type = CA_ARG_INT;
	iteration_option->_value = &iterations;

	sampleArgs->AddOption(iteration_option);
	delete iteration_option;

	Option* globalthread_x_option = new Option;
	CHECK_ALLOCATION(globalthread_x_option, "Memory allocation error.\n");

	globalthread_x_option->_sVersion = "gx";
	globalthread_x_option->_lVersion = "globalthread_x";
	globalthread_x_option->_description = "Number of globalthread_x ";
	globalthread_x_option->_type = CA_ARG_INT;
	globalthread_x_option->_value = &globalThreads[0];
	
	sampleArgs->AddOption(globalthread_x_option);
	delete globalthread_x_option;

	Option* globalthread_y_option = new Option;
	CHECK_ALLOCATION(globalthread_y_option, "Memory allocation error.\n");

	globalthread_y_option->_sVersion = "gy";
	globalthread_y_option->_lVersion = "globalthread_y";
	globalthread_y_option->_description = "Number of globalthread_y ";
	globalthread_y_option->_type = CA_ARG_INT;
	globalthread_y_option->_value = &globalThreads[1];

	sampleArgs->AddOption(globalthread_y_option);
	delete globalthread_y_option;


	Option* globalthread_z_option = new Option;
	CHECK_ALLOCATION(globalthread_z_option, "Memory allocation error.\n");

	globalthread_z_option->_sVersion = "gz";
	globalthread_z_option->_lVersion = "globalthread_z";
	globalthread_z_option->_description = "Number of globalthread_z ";
	globalthread_z_option->_type = CA_ARG_INT;
	globalthread_z_option->_value = &globalThreads[2];

	sampleArgs->AddOption(globalthread_z_option);
	delete globalthread_z_option;


	Option* localthread_x_option = new Option;
	CHECK_ALLOCATION(localthread_x_option, "Memory allocation error.\n");

	localthread_x_option->_sVersion = "lx";
	localthread_x_option->_lVersion = "localthread_x";
	localthread_x_option->_description = "Number of localthread_x ";
	localthread_x_option->_type = CA_ARG_INT;
	localthread_x_option->_value = &localThreads[0];

	sampleArgs->AddOption(localthread_x_option);
	delete localthread_x_option;

	Option* localthread_y_option = new Option;
	CHECK_ALLOCATION(localthread_y_option, "Memory allocation error.\n");

	localthread_y_option->_sVersion = "ly";
	localthread_y_option->_lVersion = "localthread_y";
	localthread_y_option->_description = "Number of localthread_y ";
	localthread_y_option->_type = CA_ARG_INT;
	localthread_y_option->_value = &localThreads[1];

	sampleArgs->AddOption(localthread_y_option);
	delete localthread_y_option;

	Option* localthread_z_option = new Option;
	CHECK_ALLOCATION(localthread_z_option, "Memory allocation error.\n");

	localthread_z_option->_sVersion = "lz";
	localthread_z_option->_lVersion = "localthread_z";
	localthread_z_option->_description = "Number of localthread_z";
	localthread_z_option->_type = CA_ARG_INT;
	localthread_z_option->_value = &localThreads[2];

	sampleArgs->AddOption(localthread_z_option);
	delete localthread_z_option;

	Option* dimension_option = new Option;
	CHECK_ALLOCATION(dimension_option, "Memory allocation error.\n");

	dimension_option->_sVersion = "dim";
	dimension_option->_lVersion = "dimension";
	dimension_option->_description = "Number of dimension";
	dimension_option->_type = CA_ARG_INT;
	dimension_option->_value = &dimension;

	sampleArgs->AddOption(dimension_option);
	delete dimension_option;

	Option* option_constant1 = new Option;
	CHECK_ALLOCATION(option_constant1, "Memory allocation error.\n");

	option_constant1->_sVersion = "c1";
	option_constant1->_lVersion = "costant1";
	option_constant1->_description = "Value of Constant1";
	option_constant1->_type = CA_ARG_INT;
	option_constant1->_value = &const1;

	sampleArgs->AddOption(option_constant1);
	delete option_constant1;

	Option* option_constant2 = new Option;
	CHECK_ALLOCATION(option_constant2, "Memory allocation error.\n");

	option_constant2->_sVersion = "c2";
	option_constant2->_lVersion = "costant2";
	option_constant2->_description = "Value of Constant2";
	option_constant2->_type = CA_ARG_INT;
	option_constant2->_value = &const2;

	sampleArgs->AddOption(option_constant2);
	delete option_constant2;

	Option* option_constant3 = new Option;
	CHECK_ALLOCATION(option_constant3, "Memory allocation error.\n");

	option_constant3->_sVersion = "c3";
	option_constant3->_lVersion = "costant3";
	option_constant3->_description = "Value of Constant3";
	option_constant3->_type = CA_ARG_INT;
	option_constant3->_value = &const3;

	sampleArgs->AddOption(option_constant3);
	delete option_constant3;


	Option* option_constant4 = new Option;
	CHECK_ALLOCATION(option_constant4, "Memory allocation error.\n");

	option_constant4->_sVersion = "c4";
	option_constant4->_lVersion = "costant4";
	option_constant4->_description = "Value of Constant4";
	option_constant4->_value = &const4;

	sampleArgs->AddOption(option_constant4);
	delete option_constant4;

	Option* option_constant5 = new Option;
	CHECK_ALLOCATION(option_constant5, "Memory allocation error.\n");

	option_constant5->_sVersion = "c5";
	option_constant5->_lVersion = "costant5";
	option_constant5->_description = "Value of Constant5";
	option_constant5->_type = CA_ARG_INT;
	option_constant5->_value = &const5;

	sampleArgs->AddOption(option_constant5);
	delete option_constant5;


	Option* option_constant6= new Option;
	CHECK_ALLOCATION(option_constant6, "Memory allocation error.\n");

	option_constant6->_sVersion = "c6";
	option_constant6->_lVersion = "costant6";
	option_constant6->_description = "Value of Constant6";
	option_constant6->_type = CA_ARG_INT;
	option_constant6->_value = &const6;

	sampleArgs->AddOption(option_constant6);
	delete option_constant6;


	return SDK_SUCCESS;
}

int CNNBench::InitChessBoard()
{
	int status = mapBuffer(dataBuf1, pdata1, sizeof(cl_uint) * width * height,
		CL_MAP_WRITE_INVALIDATE_REGION);
	CHECK_ERROR(status, SDK_SUCCESS, "Failed to map device buffer.(dataBuf)");

	//chess board
	int blockSize = 64;
	for (int j = 0; j < height; j++)
	{
		for (int i = 0; i < width; i++)
		{
			float v = 0;
			if (((j / blockSize) % 2) == 0)  //odd row
			{
				v = 1 - (i / blockSize) % 2;
			}
			else  //even row
			{
				v = (i / blockSize) % 2;
			}
			pdata1[j * width + i] = v;
		}
	}

	status = unmapBuffer(dataBuf1, pdata1);
	CHECK_ERROR(status, SDK_SUCCESS, "Failed to unmap device buffer.(dataBuf1)");

	status = mapBuffer(dataBuf2, pdata2, sizeof(cl_uint) * width * height,
		CL_MAP_WRITE_INVALIDATE_REGION);
	CHECK_ERROR(status, SDK_SUCCESS, "Failed to map device buffer.(dataBuf2)");

	//chess board
	blockSize = 64;
	for (int j = 0; j < height; j++)
	{
		for (int i = 0; i < width; i++)
		{
			float v = 0;
			if (((j / blockSize) % 2) == 0)  //odd row
			{
				v = 1 - (i / blockSize) % 2;
			}
			else  //even row
			{
				v = (i / blockSize) % 2;
			}
			pdata2[j * width + i] = v*.25;
		}
	}

	status = unmapBuffer(dataBuf2, pdata2);
	CHECK_ERROR(status, SDK_SUCCESS, "Failed to unmap device buffer.(dataBuf2)");

	
	status = mapBuffer(dataBuf3, pdata3, sizeof(cl_uint) * width * height,
		CL_MAP_WRITE_INVALIDATE_REGION);
	CHECK_ERROR(status, SDK_SUCCESS, "Failed to map device buffer.(dataBuf3)");

	//chess board
	blockSize = 64;
	for (int j = 0; j < height; j++)
	{
		for (int i = 0; i < width; i++)
		{
			float v = 0;
			if (((j / blockSize) % 2) == 0)  //odd row
			{
				v = 1 - (i / blockSize) % 2;
			}
			else  //even row
			{
				v = (i / blockSize) % 2;
			}
			pdata3[j * width + i] = v*.5;
		}
	}

	status = unmapBuffer(dataBuf3, pdata3);
	CHECK_ERROR(status, SDK_SUCCESS, "Failed to unmap device buffer.(dataBuf3)");


	status = mapBuffer(dataBuf4, pdata4, sizeof(cl_uint) * width * height,
		CL_MAP_WRITE_INVALIDATE_REGION);
	CHECK_ERROR(status, SDK_SUCCESS, "Failed to map device buffer.(dataBuf4)");
	//chess board
	 blockSize = 64;
	for (int j = 0; j < height; j++)
	{
		for (int i = 0; i < width; i++)
		{
			float v = 0;
			if (((j / blockSize) % 2) == 0)  //odd row
			{
				v = 1 - (i / blockSize) % 2;
			}
			else  //even row
			{
				v = (i / blockSize) % 2;
			}
			pdata4[j * width + i] = v*.75;
		}
	}

	status = unmapBuffer(dataBuf4, pdata4);
	CHECK_ERROR(status, SDK_SUCCESS, "Failed to unmap device buffer.(dataBuf4)");


	return SDK_SUCCESS;
}

int GetEncoderClsid(const WCHAR* format, CLSID* pClsid)
{
	UINT  num = 0;          // number of image encoders
	UINT  size = 0;         // size of the image encoder array in bytes

	ImageCodecInfo* pImageCodecInfo = NULL;

	GetImageEncodersSize(&num, &size);
	if (size == 0)
		return -1;  // Failure

	pImageCodecInfo = (ImageCodecInfo*)(malloc(size));
	if (pImageCodecInfo == NULL)
		return -1;  // Failure

	GetImageEncoders(num, size, pImageCodecInfo);

	for (UINT j = 0; j < num; ++j)
	{
		if (wcscmp(pImageCodecInfo[j].MimeType, format) == 0)
		{
			*pClsid = pImageCodecInfo[j].Clsid;
			free(pImageCodecInfo);
			return j;  // Success
		}
	}

	free(pImageCodecInfo);
	return -1;  // Failure
}

int CNNBench::SavePNG(const char* filename)
{//CreateFile from GDI
	// Create four image objects.
	GdiplusStartupInput gdiplusStartupInput;
	ULONG_PTR gdiplusToken;
	GdiplusStartup(&gdiplusToken, &gdiplusStartupInput, NULL);


	Bitmap* bmp = new Bitmap(width, height, PixelFormat32bppARGB);



	//Mapp
	int status = mapBuffer(dataBuf6, pdata6, sizeof(cl_float) * width * height,
		CL_MAP_READ);
	CHECK_ERROR(status, SDK_SUCCESS, "Failed to map device buffer.(resultBuf)");

	//chess board
	for (int i = 0; i < width; i++)
	{
		for (int j = 0; j < height; j++)
		{
			bmp->SetPixel(i, j, Color(255, pdata6[j * width + i] * 255.0f, pdata6[j * width + i] * 255.0f, pdata6[j * width + i] * 255.0f));
		}
	}

	status = unmapBuffer(dataBuf6, pdata6);
	CHECK_ERROR(status, SDK_SUCCESS, "Failed to unmap device buffer.(resultBuf)");

	CLSID  encoderClsid;
	INT    result2;

	result2 = GetEncoderClsid(L"image/png", &encoderClsid);

	wchar_t fn[512];

	mbstowcs(fn, filename, 256);

	if (result2 > 0)
	{

		result2 = bmp->Save(fn, &encoderClsid, NULL);
	}
	delete bmp;
	GdiplusShutdown(gdiplusToken);

	return SDK_SUCCESS;
}

int  CNNBench::SetupKernels()
{

	// create a CL program using the kernel source
	buildProgramData buildData;
	buildData.kernelName = std::string("test_kernel.cl");
	buildData.devices = devices;
	buildData.deviceId = sampleArgs->deviceId;
	buildData.flagsStr = std::string("");
	if (sampleArgs->isLoadBinaryEnabled())
	{
		buildData.binaryName = std::string(sampleArgs->loadBinary.c_str());
	}

	if (sampleArgs->isComplierFlagsSpecified())
	{
		buildData.flagsFileName = std::string(sampleArgs->flags.c_str());
	}

	int retValue = buildOpenCLProgram(program, context, buildData);
	CHECK_ERROR(retValue, 0, "buildOpenCLProgram() failed");

	
	cl_int status;
	// get a kernel object handle for a kernel with the given name
	kernel = clCreateKernel(program, "test_kernel", &status);
	char error_str[256];
	sprintf(error_str, "clCreateKernel %s failed.", "test_kernel");

	CHECK_OPENCL_ERROR(status, error_str);


	return SDK_SUCCESS;
}

int CNNBench::setupCL()
{
	cl_int status = 0;
	cl_device_type dType;

	if (sampleArgs->deviceType.compare("cpu") == 0)
	{
		dType = CL_DEVICE_TYPE_CPU;
	}
	else //sampleArgs->deviceType = "gpu"
	{
		dType = CL_DEVICE_TYPE_GPU;
		if (sampleArgs->isThereGPU() == false)
		{
			std::cout << "GPU not found. Falling back to CPU device" << std::endl;
			dType = CL_DEVICE_TYPE_CPU;
		}
	}

	/*
	* Have a look at the available platforms and pick either
	* the AMD one if available or a reasonable default.
	*/
	cl_platform_id platform = NULL;
	int retValue = getPlatform(platform, sampleArgs->platformId,
		sampleArgs->isPlatformEnabled());
	CHECK_ERROR(retValue, SDK_SUCCESS, "getPlatform() failed");

	// Display available devices.
	retValue = displayDevices(platform, dType);
	CHECK_ERROR(retValue, SDK_SUCCESS, "displayDevices() failed");


	/*
	* If we could find our platform, use it. Otherwise use just available platform.
	*/
	cl_context_properties cps[3] =
	{
		CL_CONTEXT_PLATFORM,
		(cl_context_properties)platform,
		0
	};
	context = clCreateContextFromType(
		cps,
		dType,
		NULL,
		NULL,
		&status);
	CHECK_OPENCL_ERROR(status, "clCreateContextFromType failed.");

	// getting device on which to run the sample
	status = getDevices(context, &devices, sampleArgs->deviceId,
		sampleArgs->isDeviceIdEnabled());
	CHECK_ERROR(status, SDK_SUCCESS, "getDevices() failed");

	// Create command queue
	commandQueue = clCreateCommandQueue(context,
		devices[sampleArgs->deviceId],
		0,
		&status);
	CHECK_OPENCL_ERROR(status, "clCreateCommandQueue failed.");

	//Set device info of given cl_device_id
	retValue = deviceInfo.setDeviceInfo(devices[sampleArgs->deviceId]);
	CHECK_ERROR(retValue, SDK_SUCCESS, "SDKDeviceInfo::setDeviceInfo() failed");

	SetupBuffers();

	SetupKernels();

	return SDK_SUCCESS;
}

int CNNBench::SetupBuffers()
{
	cl_int status;

	dataBuf1 = clCreateBuffer(
		context,
		CL_MEM_READ_WRITE,
		sizeof(cl_float) * width  * height,
		NULL,
		&status);

	CHECK_OPENCL_ERROR(status, "clCreateBuffer failed. (databuf1)");

	dataBuf2 = clCreateBuffer(
		context,
		CL_MEM_READ_WRITE,
		sizeof(cl_float) * width  * height,
		NULL,
		&status);
	CHECK_OPENCL_ERROR(status, "clCreateBuffer failed. (databuf2)");

	dataBuf3 = clCreateBuffer(
		context,
		CL_MEM_READ_WRITE,
		sizeof(cl_float) * width  * height,
		NULL,
		&status);
	CHECK_OPENCL_ERROR(status, "clCreateBuffer failed. (databuf3)");

	dataBuf4 = clCreateBuffer(
		context,
		CL_MEM_READ_WRITE,
		sizeof(cl_float) * width  * height,
		NULL,
		&status);
	CHECK_OPENCL_ERROR(status, "clCreateBuffer failed. (databuf4)");

	dataBuf5 = clCreateBuffer(
		context,
		CL_MEM_READ_WRITE,
		sizeof(cl_float) * width  * height,
		NULL,
		&status);
	CHECK_OPENCL_ERROR(status, "clCreateBuffer failed. (databuf5)");

	dataBuf6 = clCreateBuffer(
		context,
		CL_MEM_READ_WRITE,
		sizeof(cl_float) * width  * height,
		NULL,
		&status);
	CHECK_OPENCL_ERROR(status, "clCreateBuffer failed. (databuf6)");



	filterBuf = clCreateBuffer(
		context,
		CL_MEM_READ_ONLY,
		sizeof(cl_float) * 256 * 64,
		NULL,
		&status);
	CHECK_OPENCL_ERROR(status, "clCreateBuffer failed. (filterBuf)");

	return SDK_SUCCESS;
}

int InitlizeChessBoard()
{
	return SDK_SUCCESS;
}

int CNNBench::SetupFilter(int filterSize)
{
	//Normalized Box Filter

	int status = mapBuffer(filterBuf, filter, sizeof(cl_float) * 256 * 64,
		CL_MAP_WRITE_INVALIDATE_REGION);
	CHECK_ERROR(status, SDK_SUCCESS, "Failed to map device buffer.(filterBuf)");

	//chess board
	for (int i = 0; i < filterSize; i++)
	{
		for (int j = 0; j < filterSize; j++)
		{
			filter[i * filterSize + j] = 1.0 / (filterSize*filterSize);
		}
	}

	status = unmapBuffer(filterBuf, filter);
	CHECK_ERROR(status, SDK_SUCCESS, "Failed to unmap device buffer.(filterBuf)");

	return SDK_SUCCESS;
}



/**
* Override from SDKSample, adjust width and height
* of execution domain, perform all sample setup
* @return SDK_SUCCESS on success and SDK_FAILURE on failure
*/
int CNNBench::setup()
{
	if (iterations < 1)
	{
		std::cout << "Error, iterations cannot be 0 or negative. Exiting..\n";
		exit(0);
	}
	int status = 0;

	int timer = sampleTimer->createTimer();
	sampleTimer->resetTimer(timer);
	sampleTimer->startTimer(timer);

	status = setupCL();
	if (status != SDK_SUCCESS)
	{
		return status;
	}

	status = InitChessBoard();
	CHECK_ERROR(status, SDK_SUCCESS, "Sample Resource Setup Failed");

	sampleTimer->stopTimer(timer);

	// Compute setup time
	setupTime = (double)(sampleTimer->readTimer(timer));

	return SDK_SUCCESS;
}

/**
* Override from SDKSample
* Run OpenCL Black-Scholes
* @return SDK_SUCCESS on success and SDK_FAILURE on failure
*/
int CNNBench::run()
{
	//Setup Filter 
	SetupFilter(filterSize);

	//Setup Workthreads 

	//setup args
	// whether sort is to be in increasing order. CL_TRUE implies increasing
	cl_int status;
	cl_int eventStatus = CL_QUEUED;


	status = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&filterBuf);
	CHECK_OPENCL_ERROR(status, "clSetKernelArg failed. (filterBuf)");

	status = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&dataBuf1);
	CHECK_OPENCL_ERROR(status, "clSetKernelArg failed. (dataBuf1)");

	status = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&dataBuf2);
	CHECK_OPENCL_ERROR(status, "clSetKernelArg failed. (dataBuf2)");

	status = clSetKernelArg(kernel, 3, sizeof(cl_mem), (void*)&dataBuf3);
	CHECK_OPENCL_ERROR(status, "clSetKernelArg failed. (dataBuf3)");

	status = clSetKernelArg(kernel, 4, sizeof(cl_mem), (void*)&dataBuf4);
	CHECK_OPENCL_ERROR(status, "clSetKernelArg failed. (dataBuf4)");

	status = clSetKernelArg(kernel, 5, sizeof(cl_mem), (void*)&dataBuf5);
	CHECK_OPENCL_ERROR(status, "clSetKernelArg failed. (dataBuf5)");

	status = clSetKernelArg(kernel, 6, sizeof(cl_mem), (void*)&dataBuf6);
	CHECK_OPENCL_ERROR(status, "clSetKernelArg failed. (dataBuf6)");

	status = clSetKernelArg(kernel, 7, sizeof(cl_uint), (void*)&const1);
	CHECK_OPENCL_ERROR(status, "clSetKernelArg failed. (const1)");

	status = clSetKernelArg(kernel, 8, sizeof(cl_uint), (void*)&const2);
	CHECK_OPENCL_ERROR(status, "clSetKernelArg failed. (const2)");

	status = clSetKernelArg(kernel, 9, sizeof(cl_uint), (void*)&const3);
	CHECK_OPENCL_ERROR(status, "clSetKernelArg failed. (const3)");

	status = clSetKernelArg(kernel, 10, sizeof(cl_uint), (void*)&const4);
	CHECK_OPENCL_ERROR(status, "clSetKernelArg failed. (const4)");

	status = clSetKernelArg(kernel, 11, sizeof(cl_uint), (void*)&const5);
	CHECK_OPENCL_ERROR(status, "clSetKernelArg failed. (const5)");

	status = clSetKernelArg(kernel, 12, sizeof(cl_uint), (void*)&const6);
	CHECK_OPENCL_ERROR(status, "clSetKernelArg failed. (const6)");


	// Enqueue a kernel run call to Warm-up.	
	cl_event endrEvt;
	status = clEnqueueNDRangeKernel(
		commandQueue,
		kernel,
		dimension,
		NULL,
		globalThreads,
		localThreads,
		0,
		NULL,
		&endrEvt);
	CHECK_OPENCL_ERROR(status, "clEnqueueNDRangeKernel failed.");

	status = clFlush(commandQueue);
	CHECK_OPENCL_ERROR(status, "clFlush failed.");

	status = waitForEventAndRelease(&endrEvt);
	CHECK_ERROR(status, SDK_SUCCESS, "WaitForEventAndRelease(ndrEvt1) Failed");


	//run 100 times 
	int timer = sampleTimer->createTimer();
	sampleTimer->resetTimer(timer);
	sampleTimer->startTimer(timer);
	for (int j = 0; j < iterations; j++)
	{
		status = clEnqueueNDRangeKernel(
			commandQueue,
			kernel,
			dimension,
			NULL,
			globalThreads,
			localThreads,
			0,
			NULL,
			&endrEvt);
		CHECK_OPENCL_ERROR(status, "clEnqueueNDRangeKernel failed.");

		status = clFlush(commandQueue);
		CHECK_OPENCL_ERROR(status, "clFlush failed.");

		status = waitForEventAndRelease(&endrEvt);
		CHECK_ERROR(status, SDK_SUCCESS, "WaitForEventAndRelease(ndrEvt1) Failed");
	}

	sampleTimer->stopTimer(timer);

	// Compute average kernel time
	kernelTime = (double)(sampleTimer->readTimer(timer));

	//print out result
	//ouputImage;
	char filename[256];
	sprintf(filename, "%s.png", "result");
	SavePNG(filename);

	//Print Result
	std::string strArray[6] =
	{
		"Kernel",
		"Width",
		"Height",
		"Filter Size",
		"kernel Time(sec)",
		"Elements/sec"
	};
	std::string stats[6];

	stats[0] = "test_kernel";
	stats[1] = toString(width, std::dec);
	stats[2] = toString(height, std::dec);
	stats[3] = toString(filterSize, std::dec);
	stats[4] = toString(kernelTime / iterations, std::dec);
	stats[5] = toString(width*height / (kernelTime / iterations), std::dec);

	printStatistics(strArray, stats, 6);

	return SDK_SUCCESS;
}
/**
* Override from SDKSample
* Cleanup memory allocations
* @return SDK_SUCCESS on success and SDK_FAILURE on failure
*/
int CNNBench::cleanup()
{
	// Releases OpenCL resources (Context, Memory etc.)


	cl_int status;

	status = clReleaseMemObject(dataBuf1);
	status = clReleaseMemObject(dataBuf2);
	status = clReleaseMemObject(dataBuf3);
	status = clReleaseMemObject(dataBuf4);
	status = clReleaseMemObject(dataBuf5);
	status = clReleaseMemObject(dataBuf6);

	status = clReleaseMemObject(filterBuf);

	CHECK_OPENCL_ERROR(status, "clReleaseMemObject failed.(dataBuf)");


	status = clReleaseKernel(kernel);
	CHECK_OPENCL_ERROR(status, "clReleaseKernel failed.(kernel)");

	status = clReleaseProgram(program);
	CHECK_OPENCL_ERROR(status, "clReleaseProgram failed.(program)");

	status = clReleaseCommandQueue(commandQueue);
	CHECK_OPENCL_ERROR(status, "clReleaseCommandQueue failed.(commandQueue)");

	status = clReleaseContext(context);
	CHECK_OPENCL_ERROR(status, "clReleaseContext failed.(context)");

	// Release program resources (input memory etc.)
	FREE(devices);

	return SDK_SUCCESS;
}

int
main(int argc, char * argv[]) {

	// Example command lineL: -dim 1 -lx 64 -ly 1 -gx 4194304 -gy 1 -f 3 -c1 2048 -c2 3 
	// 2048x2048 image, 1D , 64 threads, 
	//          2048x2048 = 4194304
	//          force localthread_y =1,  globalthread_y =1
	//          FilterSize =3,  
	//         -constant1 = 2048 image Size
	//         -constant2 = 3     filterSize;

	int status = 0;
	// Create MonteCalroAsian object
	CNNBench cnnbench;

	// Initialization
	if (cnnbench.initialize() != SDK_SUCCESS)
	{
		return SDK_FAILURE;
	}


	// Parse command line options
	if (cnnbench.sampleArgs->parseCommandLine(argc, argv) != SDK_SUCCESS)
	{
		return SDK_FAILURE;
	}

	if (cnnbench.setup() != SDK_SUCCESS)
	{
		return SDK_FAILURE;
	}

	if (cnnbench.run() != SDK_SUCCESS)
	{
		return SDK_FAILURE;
	}

	if (cnnbench.cleanup() != SDK_SUCCESS)
	{
		return SDK_FAILURE;
	}

	return SDK_SUCCESS;
}


#if 0
ssh - keygen - t rsa - C "your_email@youremail.com"
ssh - T git@github.com
$ git config --global user.name "your name"
02.$ git config --global user.email "your_email@youremail.com"

echo "# CNNKernelPerfTest" >> README.md
git init
git add README.md
git commit -m "first commit"
git remote add origin https://github.com/fsword73/CNNKernelPerfTest.git
git push -u origin master

#endif 