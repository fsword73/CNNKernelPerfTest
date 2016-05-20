//Example command line: -dim 1 -lx 64 -ly 1 -gx 4194304 -gy 1 -f 3 -c1 2048 -c2 3 
//2048x2048 3x3 =0.0007 ms


void kernel test_kernel(__global const float* filter,
						__global const float* dataBuf1,
						__global float* dataBuf2,	
						__global const float* dataBuf3,
						__global const float* dataBuf4,
						__global const float* dataBuf5,
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
	
//#define imageSize 2048
//#define filterSize 3 
	int imageOffset = id / (imageSize * imageSize) * (imageSize * imageSize);
	int localid = id     % (imageSize * imageSize);
	int   row = localid / imageSize;
	int col = localid % imageSize;
	int   halfFilterSize = filterSize >> 1;
	float sum = 0;
	/*int minm = max(-halfFilterSize, -row);
	int maxm = min(halfFilterSize, imageSize - 1 - row);
	int minn = max(-halfFilterSize, -col);
	int maxn = min(halfFilterSize, imageSize - 1 - col);
	int m = minm;	*/
	
	for(int i=0; i < filterSize; i++)
	{
		int rowoffset = row - halfFilterSize + i;
		//float ff[24]; 
		//float pp[24];
		int ximage = imageOffset + rowoffset * imageSize + col - halfFilterSize ;
		/*for(int j =0; j < filterSize; j+=4)
		{
			ff[j] = filter[i * filterSize + j];
			float4 *p ;
			p = &pp[j];
			*p = vload4(ximage + j, dataBuf1);
		}*/
		for(int j =0; j < filterSize; j++)
		{
			  
			    int colOffset = col - halfFilterSize + j;
			  
			  if( rowoffset >= 0  && rowoffset < imageSize  &&
					  colOffset >= 0  && colOffset < imageSize)
				{
												
						float p = dataBuf1[ximage+colOffset]  ;
						float  f =	filter[i * filterSize + j] ;
						//sum += ff[j] * pp[j];
						sum += f*p;
				}	
		}
	}
	/*
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
	}*/
	dataBuf6[id]  = sum;//id/(2048*2048.0f);
	//DEBUG usage
	//dataBuf6[id] = dataBuf1[id]*0.5f;	
	//dataBuf6[id] = filterSize/3.0f;
	//sum = 0;
	//for( int i = 0; i < 9; i++)
	//  sum += filter[i];
	////dataBuf6[id] = filter[0] * 9.0f;
}

	
#undef imageSize 
#undef filterSize


void kernel convolve_floats(global int *p_imageSize, 
   global  int *p_filterSize,
   global const float *image,  
   global const float *filter, 
   global float *result) {
	int id = get_global_id(0);
	
	int imageSize = p_imageSize[0];
	int filterSize = p_filterSize[0];
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
			sum += image[ ximage + y] * filter[ filterrowoffset + n ];
			n++;
		}
		m++;
	}
	result[id] = sum;
}