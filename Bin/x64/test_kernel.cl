//Example command line: -dim 1 -lx 64 -ly 1 -gx 4194304 -gy 1 -f 3 -c1 2048 -c2 3 


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
	//DEBUG usage
	//dataBuf6[id] = dataBuf1[id]*0.5f;	
	//dataBuf6[id] = filterSize/3.0f;
	//sum = 0;
	//for( int i = 0; i < 9; i++)
	//  sum += filter[i];
	////dataBuf6[id] = filter[0] * 9.0f;
}

