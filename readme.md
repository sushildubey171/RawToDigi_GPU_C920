Standalone code for RawToDigi GPU aligned with CMSSW_9_2_0

We have used 2 streams. On each stream, 500 events are transferred to 
the device memory asyncronously and rawToDigi kernel executes 500 events
at a time. The memcpy and kernel calls are overlapped.


