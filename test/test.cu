//test.cpp
// Tothe measure the performace of R2D for large number of events
#include <iostream>
#include <fstream>
#include <chrono>
#include <cuda.h>
#include <cuda_runtime.h>
#include "../interface/CudaError.h"
#include "../interface/R2Dlimits.h"

using namespace std;
using namespace std::chrono;

void RawToDigi_kernel_wrapper(uint *eventIndex_h, uint *fedIndex_h, uint *word_h);
/*
  This functions checks for cuda error
  Input: debug message
  Output: returns cuda error message
*/
void checkCUDAError(const char *msg) {
  cudaError_t err = cudaGetLastError();
  if( cudaSuccess != err) {
    fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString( err) );
    exit(-1);
  }
}
// to read the inputs from the ascii file
void readInput(uint *fedCount_h,uint *eventIndex_h,uint *fedIndex_h,uint *word_h ) {
	// contains FED Count and size for each event (500 Events)
	//ifstream fedCount_EventFile("../data/fedCount_EventFile.dat");
  ifstream fedCount_EventFile("/afs/cern.ch/work/s/sdubey/public/GPU_HLT_phase1/data_920_PU50/fedCount_EventFile.dat");
	// contains fedId and its index; fedId:[0-150), index: [150-300)
	ifstream fedIndexFile("/afs/cern.ch/work/s/sdubey/public/GPU_HLT_phase1/data_920_PU50/fedIndexFile.dat");
	// contains 32 bit unsigned word(hit) for all the feds and events
	ifstream wordDataFile("/afs/cern.ch/work/s/sdubey/public/GPU_HLT_phase1/data_920_PU50/wordDataFile.dat");

	int i=0;
  uint eventCount = 0;
  eventIndex_h[0] = 0;
  while(!fedCount_EventFile.eof()) {
    fedCount_EventFile>>fedCount_h[i]>>eventCount;
    eventIndex_h[i+1] = eventIndex_h[i]+eventCount;
    //cout<<eventCount<<"  "<<eventIndex_h[i]<<"  "<<eventIndex_h[i+1]<<endl;
    i++;
    //if(i==4) break;
  }
  fedCount_EventFile.close();
  i=0;
  while(!fedIndexFile.eof()) {
  	fedIndexFile>>fedIndex_h[i++];
    //if(i==601) break;
  }
  fedIndexFile.close();
  i=0;
  while(!wordDataFile.eof()) {
  	wordDataFile>>word_h[i++];
    //if(i==260000) break;
  }
  wordDataFile.close();

}
int main() {
	
  //currently testing for 4 events with N=2
  const int N = N_EVENT* N_STREAM*sizeof(uint);
  const int SIZE = MAX_FED*MAX_WORD*N;
  const int M = 2*150*N;

  uint *fedCount_h,*eventIndex_h, *fedIndex_h, *word_h; // host copy
  //uint *fedCount_d,*eventIndex_d, *fedIndex_d, *word_d; // device copy
  // host pinned
  cudaMallocHost((void**)&fedCount_h, N);
  cudaMallocHost((void**)&eventIndex_h, N+2*sizeof(uint));
  cudaMallocHost((void**)&fedIndex_h, M+2*sizeof(uint));
  cudaMallocHost((void**)&word_h,SIZE);
  checkCUDAError("cudaMallocHost failed!");

  high_resolution_clock::time_point t1 = high_resolution_clock::now();
  readInput(fedCount_h, eventIndex_h, fedIndex_h, word_h);
  high_resolution_clock::time_point t2 = high_resolution_clock::now();
  double ms = duration_cast<milliseconds> (t2-t1).count();
  cout<<"Time(ms) to read from ascii file: "<<ms<<endl;

  //for(int i=0;i<=N_EVENT*2;i++) cout<<eventIndex_h[i]<<endl;
  RawToDigi_kernel_wrapper(eventIndex_h, fedIndex_h, word_h); 

  cudaFreeHost(fedCount_h);
  cudaFreeHost(eventIndex_h);
  cudaFreeHost(word_h);
  checkCUDAError("after memory free");
	
}










