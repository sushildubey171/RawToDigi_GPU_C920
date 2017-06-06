/**2017-03-02  Sushil Dubey  <sdubey@felk40.cern.ch>
 *
 * File Name: RawToDigiGPU.cu
 * Description: It converts Raw data into Digi data using GPU
 * then it applies the adc threshold to drop the dead pixels
 * The Output of RawToDigi data is given to pixelClusterizer
 *
**/ 
// System includes
#include <stdio.h>
#include <malloc.h>
#include <stdlib.h>
#include <assert.h>
#include <iomanip>
#include <string>
#include <iostream>
#include <fstream>
#include <chrono>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/unique.h>
#include <thrust/execution_policy.h>
// CUDA runtime
#include <cuda.h>
#include <cuda_runtime.h>
#include "../interface/RawToDigiGPU.h"
#include "../interface/R2Dlimits.h"
 #include <cuda_runtime.h>
#include "../interface/CudaError.h"
using namespace std;

// forward declaration to be moved in header file
void PixelCluster_Wrapper(uint *xx_adc, uint *yy_adc, uint *adc_d,const uint wordCounter, 
                          const int *mIndexStart, const int *mIndexEnd, uint *xx, uint *yy);
/*
void checkCUDAError(const char *msg) {
  cudaError_t err = cudaGetLastError();
  if( cudaSuccess != err) {
    fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString( err) );
    exit(-1);
  }
}
*/
/*
void initCablingMap() {

  ifstream mapFile;
  mapFile.open("RawId_ModuleId_CablingMap_ArrayFile.txt");
  string str;
  getline(mapFile, str);
  uint rawId, moduleId, rocInDU;
  int i =1;  // cabling map index starts at 1
  while(!mapFile.eof()) {
    mapFile >> rawId >> rocInDU >> moduleId;
    Map->RawId[i]    = rawId;
    Map->rocInDet[i] = rocInDU;
    Map->moduleId[i] = moduleId;
    i++;
  }
  mapFile.close();
}
*/
// New cabling Map
void initCablingMap() {

  ifstream mapFile;
  mapFile.open("../data/Pixel_Phase1_Raw2Digi_GPU_Cabling_Map.dat");
  string str;
  getline(mapFile, str);
  uint Index, FedId, Link, idinLNK, B_F, RawID, idinDU, ModuleID;
  int i =1;  // cabling map index starts at 1
  while(!mapFile.eof()) {
    mapFile >> Index>>FedId>>Link>>idinLNK>>B_F>>RawID>>idinDU>>ModuleID;
    Map->RawId[i] = RawID;
    Map->rocInDet[i] = idinDU;
    Map->moduleId[i] = ModuleID;
    i++;
  }
  mapFile.close();
  cout<<"Cabling Map uploaded successfully!"<<endl;
}

void initDeviceMemory() {
  int sizeByte = 150 * MAX_LINK * MAX_ROC * sizeof(uint)+sizeof(uint);
  // Unified memory for cabling map
  cudaMallocManaged((void**)&Map,  sizeof(CablingMap));
  cudaMallocManaged((void**)&Map->RawId,    sizeByte);
  cudaMallocManaged((void**)&Map->rocInDet, sizeByte);
  cudaMallocManaged((void**)&Map->moduleId, sizeByte);
    // Number of words for all the feds 
  //const uint MAX_WORD_SIZE = MAX_FED*MAX_WORD*sizeof(uint); 
  
  int mSize = totalModule*sizeof(int);
  mIndexStart = (int*)malloc(mSize); 
  mIndexEnd = (int*)malloc(mSize);

  const int N = N_EVENT* N_STREAM*sizeof(uint);
  const int SIZE = MAX_FED*MAX_WORD*N;
  const int M = 2*150*N;
  // device memory
  //cudaMallocHost((void**)&fedCount_d,N);
  cudaMalloc((void**)&eventIndex_d, N+2*sizeof(uint));
  cudaMalloc((void**)&fedIndex_d, M+2*sizeof(uint));
  cudaMalloc((void**)&word_d,SIZE);
  checkCUDAError("cudaMalloc failed!");
 
  //cudaMalloc((void**)&word_d,       MAX_WORD_SIZE);
  //cudaMalloc((void**)&fedIndex_d,   2*(MAX_FED+1)*sizeof(uint));
  
  cudaMalloc((void**)&xx_d,         SIZE); // to store the x and y coordinate
  cudaMalloc((void**)&yy_d,         SIZE);
  cudaMalloc((void**)&adc_d,        SIZE);

  //cudaMalloc((void**)&xx_adc,       MAX_WORD_SIZE); // to store the x and y coordinate
  //cudaMalloc((void**)&yy_adc,       MAX_WORD_SIZE);
  //cudaMalloc((void**)&layer_d ,     MAX_WORD_SIZE);
  //cudaMalloc((void**)&RawId_d,      MAX_WORD_SIZE);
  //cudaMalloc((void**)&moduleId_d,   MAX_WORD_SIZE);
  cudaMalloc((void**)&mIndexStart_d, mSize);
  cudaMalloc((void**)&mIndexEnd_d, mSize);
  
  cout<<"Memory Allocated successfully !\n";
  // Upload the cabling Map
  initCablingMap();
  
}

void freeMemory() {

  //GPU specific
  // memory used for testing purpose will be released during
  // deployment.
  free(mIndexStart);
  free(mIndexEnd);
  cudaFree(word_d);
  cudaFree(fedIndex_d);
  cudaFree(eventIndex_d);
  cudaFree(xx_d);
  cudaFree(yy_d);
  cudaFree(adc_d);
  //cudaFree(layer_d);
  
  //cudaFree(xx_adc);
  //cudaFree(yy_adc);
  //cudaFree(RawId_d);
  //cudaFree(moduleId_d);
  cudaFree(mIndexStart_d);
  cudaFree(mIndexEnd_d);
  cudaFree(Map->RawId);
  cudaFree(Map->rocInDet); 
  cudaFree(Map->moduleId);
  cudaFree(Map);
  cout<<"Memory Released !\n";

}

__device__ uint getLink(uint ww)  {
  //printf("Link_shift: %d  LINK_mask: %d\n", LINK_shift, LINK_mask);
  return ((ww >> LINK_shift) & LINK_mask);
}

__device__ uint getRoc(uint ww) {
  return ((ww >> ROC_shift ) & ROC_mask);
}
__device__ uint getADC(uint ww) {
  return ((ww >> ADC_shift) & ADC_mask);
}

__device__ bool isBarrel(uint rawId) {
  return (1==((rawId>>25)&0x7));
}
//__device__ uint FED_START = 1200;

__device__ DetIdGPU getRawId(const CablingMap *Map, uint fed, uint link, uint roc) {
  uint index = fed * MAX_LINK* MAX_ROC + (link-1)* MAX_ROC + roc;
  DetIdGPU detId = {Map->RawId[index], Map->rocInDet[index], Map->moduleId[index]};
  return detId;  
}

//reference http://cmsdoxygen.web.cern.ch/cmsdoxygen/CMSSW_9_2_0/doc/html/dd/d31/FrameConversion_8cc_source.html
//http://cmslxr.fnal.gov/source/CondFormats/SiPixelObjects/src/PixelROC.cc?v=CMSSW_9_2_0#0071
// Convert local pixel to global pixel
__device__ Pixel frameConversion(bool bpix, int side, uint layer,uint rocIdInDetUnit, Pixel local) {
  
  int slopeRow  = 0,  slopeCol = 0;
  int rowOffset = 0, colOffset = 0;

  if(bpix) {
    
    if(side==-1 && layer!=1) { // -Z side: 4 non-flipped modules oriented like 'dddd', except Layer 1
      if (rocIdInDetUnit <8) {
        slopeRow = 1;     
        slopeCol = -1;
        rowOffset = 0;
        colOffset = (8-rocIdInDetUnit)*numColsInRoc-1;
      }
      else {
        slopeRow  = -1;
        slopeCol  = 1;
        rowOffset = 2*numRowsInRoc-1;
        colOffset = (rocIdInDetUnit-8)*numColsInRoc;
      } // if roc
    }
    else { // +Z side: 4 non-flipped modules oriented like 'pppp', but all 8 in layer1
      if(rocIdInDetUnit <8) {
        slopeRow  = -1;
        slopeCol  =  1;
        rowOffset = 2*numRowsInRoc-1;
        colOffset = rocIdInDetUnit * numColsInRoc;
      }
      else {
        slopeRow  = 1;
        slopeCol  = -1;
        rowOffset = 0;
        colOffset = (16-rocIdInDetUnit)*numColsInRoc-1;
      }
    }

  }
  else { // fpix
    if(side==-1) { // pannel 1
      if (rocIdInDetUnit < 8) {
        slopeRow = 1;
        slopeCol = -1;
        rowOffset = 0;
        colOffset = (8-rocIdInDetUnit)*numColsInRoc-1;
      }
      else {
        slopeRow = -1;
        slopeCol = 1;
        rowOffset = 2*numRowsInRoc-1;
        colOffset = (rocIdInDetUnit-8)*numColsInRoc;
      }
    }
    else { // pannel 2
      if (rocIdInDetUnit < 8) {
        slopeRow = 1;
        slopeCol = -1;
        rowOffset = 0;
        colOffset = (8-rocIdInDetUnit)*numColsInRoc-1;
      }
      else {
        slopeRow = -1;
        slopeCol = 1;
        rowOffset = 2*numRowsInRoc-1;
        colOffset = (rocIdInDetUnit-8)*numColsInRoc;
      }

    } // side

  }

  uint gRow = rowOffset+slopeRow*local.row;
  uint gCol = colOffset+slopeCol*local.col;
  //printf("Inside frameConversion gRow: %u  gCol: %u\n",gRow, gCol);
  Pixel global = {gRow, gCol};
  return global;
}


/*----------
* Name: applyADCthreshold_kernel()
* Desc: converts adc count to electrons and then applies the 
* threshold on each channel. 
* make pixel to 0 if it is below the threshold
* Input: xx_d[], yy_d[], layer_d[], wordCounter, adc[], ADCThreshold
*-----------
* Output: xx_adc[], yy_adc[] with pixel threshold applied 
*/
// Before giving input to the cluster we convert adc into electron
// kernel to apply adc threshold on the channels	
/* //comment for measuring RAwTODigi timings
__global__ void applyADCthreshold_kernel
(const uint *xx_d, const uint *yy_d, const uint *layer_d, uint *adc, const uint wordCounter,
 const ADCThreshold adcThreshold, uint *xx_adc, uint *yy_adc ) {
  int tid = threadIdx.x;
  int gIndex = blockDim.x*blockIdx.x+tid;
  if(gIndex<wordCounter) {
    //int i=0;
    //for(DigiIterator di = begin; di != end; ++di) {
      uint adcOld = adc[gIndex];
      const float gain = adcThreshold.theElectronPerADCGain_; // default: 1 ADC = 135 electrons
      const float pedestal = 0; //
      int adcNew = int(adcOld*gain+pedestal);
      // rare chance of entering into the if()
      if (layer_d[gIndex]>=adcThreshold.theFirstStack_) {
        if (adcThreshold.theStackADC_==1 && adcOld==1) {
          adcNew = int(255*135); // Arbitrarily use overflow value.
        }
        if (adcThreshold.theStackADC_ >1 && adcThreshold.theStackADC_!=255 && adcOld>=1){
          adcNew = int((adcOld-1) * gain * 255/float(adcThreshold.theStackADC_-1));
        }
      }
  
    if(adcNew >adcThreshold.thePixelThreshold ) {
      xx_adc[gIndex]=xx_d[gIndex];
      yy_adc[gIndex]=yy_d[gIndex];
    }
    else {
      xx_adc[gIndex]=0; // 0: dead pixel
      yy_adc[gIndex]=0;
    }
    adc[gIndex] = adcNew;
  }
}  
*/

// Kernel to perform Raw to Digi conversion
__global__ void RawToDigi_kernel(const CablingMap *Map,const uint *Word,const uint *fedIndex, 
                                 uint *eventIndex, uint stream, uint *XX, uint *YY,uint *ADC,int *mIndexStart,int *mIndexEnd)
 
{
  //printf("Inside GPU: \n");
  uint blockId = blockIdx.x;
  uint eventno = blockIdx.y;

  uint event_offset = eventIndex[blockDim.y*stream +eventno];
  uint fed_offset = 2*150*eventno; 
   
  uint fedId    = fedIndex[fed_offset+blockId];
  if(blockIdx.x==gridDim.x-1) {
    //printf("blockId: %u fedId: %u\n",blockId, fedId);
    return;}
  uint threadId = threadIdx.x;
  uint begin  = event_offset+ fedIndex[fed_offset+150+blockId];
  uint end    = event_offset+ fedIndex[fed_offset+150+blockId+1];
  //if(threadIdx.x==0)
  //printf("blockId: %u eventno: %u event_offset: %u  fed_offset: %u fedId: %u begin: %u  end: %u\n",blockId,eventno, event_offset,fed_offset, fedId, begin, end);
  int no_itr = (end - begin)/ blockDim.x + 1; // to deal with number of hits greater than blockDim.x 
  #pragma unroll
  for(uint i =0; i<no_itr; i++) { // use a static number to optimize this loop
    uint gIndex = begin + threadId + i*blockDim.x;  // *optimize this
    //printf("g: %d\n",gIndex);
    if(gIndex <end) {
      uint ww    = Word[gIndex]; // Array containing 32 bit raw data
      if(ww == 0 ) {
        //noise and dead channels are ignored
        XX[gIndex] = 0;  // 0 is an indicator of a noise/dead channel
        YY[gIndex]  = 0; // skip these pixels during clusterization
        //RawId[gIndex] = 0; 
        ADC[gIndex]   = 0; 
        //moduleId[gIndex] = 9999; //9999 is the indication of bad module, taken care later  
        //layerArr[gIndex] = 0;
        //fedIdArr[gIndex] = fedId; // used for testing
        continue ;         // 0: bad word, 
      } 
      uint link  = getLink(ww);            // Extract link
      uint roc   = getRoc(ww);             // Extract Roc in link
      DetIdGPU detId = getRawId(Map, fedId, link, roc);
      uint rawId  = detId.RawId;
      uint rocIdInDetUnit = detId.rocInDet;
     
      bool barrel = isBarrel(rawId);
  
      //printf("ww: %u    link:  %u  roc: %u   rawId: %u\n", ww, link, roc, rawId);
      //printf("from CablingMap  rocInDU: %u  moduleId: %u", rocIdInDetUnit, detId.moduleId);
      //printf("barrel: %d\n", barrel);
      uint layer =0;//, ladder =0;
      int side =0, panel =0,  module=0;//blade =0,disk =0,
    
      if(barrel) {
        layer  = (rawId >> layerStartBit_)  & layerMask_;
        //ladder = (rawId >> ladderStartBit_) & ladderMask_;
        module = (rawId >> moduleStartBit_) & moduleMask_;
        side   = (module<5)? -1:1;
     
      }
      else {
        // endcap ids
        layer = 0;
        panel = (rawId >> panelStartBit_) & panelMask_;
        //disk  = (rawId >> diskStartBit_)  & diskMask_ ;
        side  = (panel==1)? -1:1;
        //blade = (rawId>>bladeStartBit_) & bladeMask_;
      }
      // ***special case of layer to 1 be handled here
      Pixel localPix;
      if(layer==1) {
        uint col = (ww >> COL_shift) & COL_mask;
        uint row = (ww >> ROW_shift) & ROW_mask;
        localPix.row = row;
        localPix.col = col;
        //if(event==0 && fedId==0)
         //printf("col: %u  row: %u\n",col, row);
      }
      else {
        // ***conversion rules for dcol and pxid
        uint dcol = (ww >> DCOL_shift) & DCOL_mask;
        uint pxid = (ww >> PXID_shift) & PXID_mask;
        uint row  = numRowsInRoc - pxid/2;
        uint col  = dcol*2 + pxid%2;
        localPix.row = row;
        localPix.col = col;
      }
      //if(fedId==48)
        //printf("%14u%6d%6d%6d\n",ww,localPix.row,localPix.col, getADC(ww));
      Pixel globalPix = frameConversion(barrel, side, layer,rocIdInDetUnit, localPix);
      XX[gIndex]    = globalPix.row +1 ; // origin shifting by 1 0-159
      YY[gIndex]    = globalPix.col +1 ; // origin shifting by 1 0-415
      ADC[gIndex]   = getADC(ww);
      //RawId[gIndex] = detId.RawId; // only for testing
      //moduleId[gIndex] = detId.moduleId;
    } // end of if(gIndex < end)
  } // end of for(int i =0;i<no_itr...)
  
// Further code is used to prepare the input for clustering

/* As we pass the output of RawToDigi on clustering 
 we store the ModuleStartIndex and moduleEndIndex
 Since one pixel of a module reappear after another module
 We do all this to apply the correction
*/
// comment the below section for measuring the RawToDigi performance
/* 
  __syncthreads();
  // three cases possible
  // case 1: 21 21 21 22 21 22 22
  // pos   : 0  1  2  3  4  5  6
  // solution swap 21 with 22 : 21 21 21 21 22 22 22
  // atomicExch(address, value), set the variable at address to value.
  // do the swapping for above case and replace the 9999 with 
  // valid moduleId
  for(int i =0; i<no_itr; i++) { 
    int gIndex = begin + threadId + i*blockDim.x;  
    if(gIndex <end) {
      //rare condition 
      if(moduleId[gIndex]==moduleId[gIndex+2] && moduleId[gIndex]<moduleId[gIndex+1]) {
        atomicExch(&moduleId[gIndex+2], atomicExch(&moduleId[gIndex+1], moduleId[gIndex+2]));
        //*swap all the digi id
        atomicExch(&XX[gIndex+2], atomicExch(&XX[gIndex+1], XX[gIndex+2]));
        atomicExch(&YY[gIndex+2], atomicExch(&YY[gIndex+1], YY[gIndex+2]));
        atomicExch(&RawId[gIndex+2], atomicExch(&RawId[gIndex+1], RawId[gIndex+2])); 
        //atomicExch(&fedIdArr[gIndex+2], atomicExch(&fedIdArr[gIndex+1], fedIdArr[gIndex+2]));
        atomicExch(&ADC[gIndex+2], atomicExch(&ADC[gIndex+1], ADC[gIndex+2]));
        atomicExch(&layerArr[gIndex+2], atomicExch(&layerArr[gIndex+1], layerArr[gIndex+2]));
      }
      __syncthreads();
      //rarest condition
      // above condition fails at 361 361 361 363 362 363 363
      // here we need to swap 362 with previous 363
      if(moduleId[gIndex]==moduleId[gIndex+2] && moduleId[gIndex]>moduleId[gIndex+1]) {
        atomicExch(&moduleId[gIndex+1], atomicExch(&moduleId[gIndex], moduleId[gIndex+1]));
        //*swap all the digi id
        atomicExch(&XX[gIndex+1], atomicExch(&XX[gIndex], XX[gIndex+1]));
        atomicExch(&YY[gIndex+1], atomicExch(&YY[gIndex], YY[gIndex+1]));
        atomicExch(&RawId[gIndex+1], atomicExch(&RawId[gIndex], RawId[gIndex+1])); 
        //atomicExch(&fedIdArr[gIndex+2], atomicExch(&fedIdArr[gIndex+1], fedIdArr[gIndex+2]));
        atomicExch(&ADC[gIndex+1], atomicExch(&ADC[gIndex], ADC[gIndex+1]));
        atomicExch(&layerArr[gIndex+1], atomicExch(&layerArr[gIndex], layerArr[gIndex+1]));
      }
      // moduleId== 9999 then pixel is bad with x=y=layer=adc=0
      // this bad pixel will not affect the cluster, since for cluster
      // the origin is shifted at (1,1) so x=y=0 will be ignored
      // assign the previous valid moduleId to this pixel to remove 9999
      // so that we can get the start & end index of module easily.
      __syncthreads(); // let the swapping finish first
      if(moduleId[gIndex]==9999) {
        int m=gIndex;
        while(moduleId[--m]==9999) {} //skip till you get the valid module
        moduleId[gIndex]=moduleId[m];
      } 
    } // end of if(gIndex<end)
  } //  end of for(int i=0;i<no_itr;...)
  __syncthreads();

  // mIndexStart stores staring index of module 
  // mIndexEnd stores end index of module 
  // both indexes are inclusive 
  // check consecutive module numbers
  // for start of fed
  for(int i =0; i<no_itr; i++) { 
    int gIndex = begin + threadId + i*blockDim.x;  
    if(gIndex <end) {
      if(gIndex == begin) {
        mIndexStart[moduleId[gIndex]] = gIndex;
      }
      // for end of the fed
      if(gIndex == (end-1)) {  
        mIndexEnd[moduleId[gIndex]] = gIndex;
      }   
      // point to the gIndex where two consecutive moduleId varies
      if(gIndex!= begin && (gIndex<(end-1)) && moduleId[gIndex]!=9999) {
        if(moduleId[gIndex]<moduleId[gIndex+1] ) {
          mIndexEnd[moduleId[gIndex]] = gIndex;
        }
        if(moduleId[gIndex] > moduleId[gIndex-1] ) {
          mIndexStart[moduleId[gIndex]] = gIndex;
        } 
      } //end of if(gIndex!= begin && (gIndex<(end-1)) ...  
    } //end of if(gIndex <end) 
  }
*/  
} // end of Raw to Digi kernel

void RawToDigi_kernel_wrapper(uint *eventIndex_h, uint *fedIndex_h, uint *word_h) {
  initDeviceMemory();
  cudaStream_t stream[N_STREAM];
  for(int i=0;i<N_STREAM;i++)
    cudaStreamCreate(&stream[i]);
  
  uint word_offset, fed_offset;
  uint W_size = 0, F_size = 0;
  dim3 gridsize(MAX_FED,N_EVENT);
  dim3 blocksize = 512;
  //for(uint j=0;j<20;j++)
  for(uint i=0;i<N_STREAM;i++) {
    word_offset = W_size;
    fed_offset  = F_size;

    W_size  =  eventIndex_h[N_EVENT+N_EVENT*i]-eventIndex_h[N_EVENT*i];
    F_size  = 2*150*N_EVENT;
    cout<<"word_offset: "<<word_offset<<"  fed_offset: "<<fed_offset<<endl;
    cout<<"W_size: "<<W_size<<" F_size:"<<F_size<<endl;
    cudaMemcpyAsync(&eventIndex_d[N_EVENT*i], &eventIndex_h[N_EVENT*i], N_EVENT*sizeof(uint), cudaMemcpyHostToDevice, stream[i] );
    checkCUDAError("Error in copying eventIndex");
    cudaMemcpyAsync(&word_d[word_offset], &word_h[word_offset], W_size*sizeof(uint), cudaMemcpyHostToDevice,stream[i]);
    checkCUDAError("Error in copying word");
    cudaMemcpyAsync(&fedIndex_d[fed_offset], &fedIndex_h[fed_offset], F_size*sizeof(uint), cudaMemcpyHostToDevice,stream[i]);
    checkCUDAError("Error in copying fedIndex");
    for(int j=0;j<20;j++) {
    RawToDigi_kernel<<<gridsize,blocksize,0,stream[i]>>>(Map,word_d,fedIndex_d,eventIndex_d,i,xx_d,yy_d,adc_d,mIndexStart_d,mIndexEnd_d);
    //cudaDeviceSynchronize();
    }
  }
  cudaDeviceSynchronize();
  checkCUDAError("Error after kernel call");
  freeMemory();
}

/* old implementation with 1 events
// kernel wrapper called from runRawToDigi_kernel
void RawToDigi_kernel_wrapper(const uint wordCounter,uint *word,const uint fedCounter, uint *fedIndex) { 

  cout<<"Inside RawToDigi , total words: "<<wordCounter<<endl;
  int nBlocks = fedCounter; // = MAX_FED
  int threads = 512; //
  fedIndex[MAX_FED+nBlocks] = wordCounter;
  // for debugging 
  uint *fedId;
  int mSize = totalModule*sizeof(int);
  { 
    uint eventSize = wordCounter*sizeof(uint);
	  // initialize moduleStart & moduleEnd with some constant(-1)
	  // number just to check if it updated in kernel or not
    cudaMemset(mIndexStart_d, -1, mSize);
    cudaMemset(mIndexEnd_d, -1, mSize);
    cudaMemcpy(word_d, word, eventSize, cudaMemcpyHostToDevice);
    cudaMemcpy(fedIndex_d, fedIndex, 2*(MAX_FED+1)*sizeof(uint), cudaMemcpyHostToDevice); 
    // for debugging 
    cudaMallocManaged((void**)&fedId, eventSize);
    // Launch rawToDigi kernel
    RawToDigi_kernel<<<nBlocks,threads>>>(Map,word_d, fedIndex_d, xx_d, yy_d, RawId_d,
                                          moduleId_d, mIndexStart_d, mIndexEnd_d, adc_d, layer_d,fedId);
    cudaDeviceSynchronize();
    
    //---- Prepare the input for clustering----------
     
     // This correction can be done during clustering also

     // apply the correction to the moduleStart & moduleEnd
     // if module contains only one pixel then either moduleStart 
     // or moduleEnd is not updated(remains 9999) in RawToDigi kernel
     // ex. moduleStart[1170] =9999 & moduleEnd[1170] = 34700
     // because of 1 pixel moduleStart[1170] didn't update
     // as per the if condition
   
     // before finding the cluster 
    cudaMemcpy(mIndexStart, mIndexStart_d, mSize, cudaMemcpyDeviceToHost);
    cudaMemcpy(mIndexEnd,   mIndexEnd_d,   mSize, cudaMemcpyDeviceToHost);
    for(int i=0;i<totalModule;i++) {
      // if module is empty then index are not updated in kernel
      if(mIndexStart[i]==-1 && mIndexEnd[i]==-1) {
        mIndexStart[i]=0;
        mIndexEnd[i]=0;
      }
      else if(mIndexStart[i]==-1) {
        mIndexStart[i] = mIndexEnd[i];
      }
      else if(mIndexEnd[i]==-1) {
        mIndexEnd[i] = mIndexStart[i];
      }
    }
    //copy te data back to the device memory
    cudaMemcpy(mIndexStart_d, mIndexStart, mSize, cudaMemcpyHostToDevice);
    cudaMemcpy(mIndexEnd_d,   mIndexEnd,   mSize, cudaMemcpyHostToDevice);
    
  }

  cudaFree(fedId);
  //cout<<"RawToDigi Kernel executed successfully!\n";
  
  ///*ofstream outFile; 
  outFile.open("InputForCluster.txt");
  outFile<<"FedId   "<<"  wwIndex   "<<"  moduleId  "<<" RawId  "<<endl;
  for(uint i=0; i<wordCounter;i++) {
   // if(RawId[i]!=0)
    outFile <<setw(10)<<xx[i]<<"\t\t"<<setw(10)<<yy[i]<<endl;
    //outFile<<setw(4)<<xx[i]<<setw(12)<<yy[i]<<setw(12)<<moduleId[i]<<setw(16)<<RawId[i]<<endl;
    //cout<<"ww: "<<setw(10)<<RawId[i]<<"  xx: "<<setw(3)<<xx[i]<<"  yy: "<<setw(3)<<yy[i]<<endl;
  }
  //*
  //ofstream mse("ModuleStartEndIndex.txt");
  //for(int i=0;i<totalModule;i++) {
   // mse<<mIndexStart[i]<<"\t\t"<<mIndexEnd[i]<<endl;
    //cout<<mIndexStart[i]<<"\t\t"<<mIndexEnd[i]<<endl;
  //}
  //mse.close();
  //outFile.close();                                           
  //++eventno;
   
  //cout<<"Calling pixel cluster"<<endl;
  // kernel to apply adc threashold on the channel
  ADCThreshold adcThreshold;
  uint numThreads = 512;
  uint numBlocks = wordCounter/512 +1;
  applyADCthreshold_kernel<<<numBlocks, numThreads>>>(xx_d, yy_d,layer_d,adc_d,wordCounter,adcThreshold, xx_adc, yy_adc);
  cudaDeviceSynchronize();
  
  // call to pixelClusterizer kernel from here
  //PixelCluster_Wrapper(xx_adc , yy_adc, adc_d,wordCounter, mIndexStart_d, mIndexEnd_d,xx,yy);
}

*/