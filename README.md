# Compressor-zfp
This is ZFP compressor project example for visual studio<br/>
Downloaded example from intel<br/>
https://www.intel.com/content/dam/develop/external/us/en/documents/intel_ipp_zfp_parallel.c

## Image compression support
Supported 4 channel image only. <br/>
use "#define ChangeToImage 1" to change 4 Channel Image compression<br/>
### Pseudocode<br/>
1. Open image
2. format data into 1d array to fit in Intel ZFP algorithm
3. Compress data
4. save compressed data into client.data
5. decompress data
6. calculate and shows data result
7. show decompressed image 
8. save decompressed image
