
CNNKernelPerfTest

Aimed to promote performance of OpenCL kernel of  Convolutional Neural Network.  The simple APP is defined as 
 * 4 buffers with initiazed with Chess Board  
 * 6 Input Buffer and 6 constant integers
 * the 6th buffer will be saved as PNG 
 * The kernel is defined as standard name and format 
 
void kernel test_kernel(__constant float* filter,
						__global float* dataBuf1,
						__global float* dataBuf2,	
						__global float* dataBuf3,
						__global float* dataBuf4,
						__global float* dataBuf5,
						__global float* dataBuf6,
						const int const1,  
						const int const2,  
						const int const3,  
						const int const4,  
						const int const5,  
						const int const6)  
{
	int id = get_global_id(0);
	
	int imageSize = const1;
	int filterSize = const2;
	int imageOffset = id / (imageSize * imageSize) * (imageSize * imageSize);
	int localid = id % (imageSize * imageSize);
	int row = localid / imageSize;
	int col = localid % imageSize;
	int halfFilterSize = filterSize >> 1;
	float sum = 0;
	int minm = max(-halfFilterSize, -row);
	int maxm = min(halfFilterSize, imageSize - 1 - row);
	int minn = max(-halfFilterSize, -col);
	int maxn = min(halfFilterSize, imageSize - 1 - col);
	int m = minm;	
		
	while(m <= maxm) {
		int x = (row + m);
		int ximage = imageOffset + x * imageSize;
		int filterrowoffset = (m+halfFilterSize) * filterSize + halfFilterSize;
		int n = minn;
		while(n <= maxn) {
			int y = col + n;
			sum += dataBuf1[ ximage + y] * filter[ filterrowoffset + n ];
			n++;
		}
		m++;
	}
	dataBuf6[id]  = sum;
}

The example command line will be 
CnnForward.exe -dim 1 -gx 4194304 -gy 1 -lx 64 -ly 1  -f 3 -c1 2048 -c2 3 

-dim 1        :  1D buffer
-gx  4194304  :   globalthreads_x 2048x2048 image
-gy  1        :   globalthreads_y 
-lx  64       :   localthreads_x 
-ly  1        :   localthreads_y
-f  3         :   filterSize; 
-c1  2048     :    constant1,  iamgeSize
-c2  3        :    constant2,  fitlerSize


