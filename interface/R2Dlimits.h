#ifndef R2Dlimits_H
#define R2Dlimits_H

typedef unsigned int  uint;
// Maximum fed for phase1 is 150 but not all of them are filled
// only 108 feds are filled,found after debugging 920/PU50 for 500 events
// change only MAX_FED dependning upon actual no of feds present 
const uint MAX_FED  = 108; 
const uint N_EVENT  = 500; // number of events to process on one stream
const uint N_STREAM = 2;
const uint MAX_LINK = 48; //maximum links/channels for phase1 
const uint MAX_ROC  = 8;  
const uint MAX_WORD = 4096; 

#endif

