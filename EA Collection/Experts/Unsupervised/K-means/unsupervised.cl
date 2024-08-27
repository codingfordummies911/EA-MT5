/// \file
/// \brief unsupervised.cl
/// Library consist OpenCL kernels
/// \author <A HREF="https://www.mql5.com/en/users/dng"> DNG </A>
/// \copyright Copyright 2022, DNG
//---
//--- by default some GPU doesn't support floats
//--- cl_khr_fp64 directive is used to enable work with floats
#pragma OPENCL EXTENSION __opencl_c_fp64 /*cl_khr_fp64*/ : enable
//+------------------------------------------------------------------+
///\ingroup k-means K means clustering
/// Describes the process of distance calculation for the class CKmeans (#CKmeans).
///\details Detailed description on <A HREF="https://www.mql5.com/ru/articles/10947#distance">the link.</A>
//+------------------------------------------------------------------+
__kernel void KmeansCulcDistance(__global float *data,///<[in] Inputs data matrix m*n, where m - number of patterns and n - size of vector to describe 1 pattern
                                 __global float *means,///<[in] Means tensor k*n, where k - number of clasters and n - size of vector to describe 1 pattern
                                 __global float *distance,///<[out] distance tensor m*k, where m - number of patterns and k - number of clasters
                                 int vector_size///< Size of vector
                                )
  {
   int m = get_global_id(0);
   int k = get_global_id(1);
   int total_k = get_global_size(1);
   float sum = 0.0;
   int shift_m = m * vector_size;
   int shift_k = k * vector_size;
   for(int i = 0; i < vector_size; i++)
      sum += pow(data[shift_m + i] - means[shift_k + i], 2);
   distance[m * total_k + k] = sum;
  }
//+------------------------------------------------------------------+
///\ingroup k-means K means clustering
/// Describes the process of clustering for the class CKmeans (#CKmeans).
///\details Detailed description on <A HREF="https://www.mql5.com/ru/articles/10947#clustering">the link.</A>
//+------------------------------------------------------------------+
__kernel void KmeansClustering(__global float *distance,///<[in] distance tensor m*k, where m - number of patterns and k - number of clasters
                               __global float *clusters,///<[out] Numbers of cluster tensor m-size
                               __global float *flags,///[out] Flags of changes
                               int total_k///< Number of clusters
                              )
  {
   int i = get_global_id(0);
   int shift = i * total_k;
   float value = distance[shift];
   int result = 0;
   for(int k = 1; k < total_k; k++)
     {
      if(value <= distance[shift + k])
         continue;
      value =  distance[shift + k];
      result = k;
     }
   flags[i] = (float)(clusters[i] != (float)result);
   clusters[i] = (float)result;
  }
//+------------------------------------------------------------------+
///\ingroup k-means K means clustering
/// Describes the process of updates means vectors
///\details Detailed description on <A HREF="https://www.mql5.com/ru/articles/10947#updates">the link.</A>
//+------------------------------------------------------------------+
__kernel void KmeansUpdating(__global float *data,///<[in] Inputs data matrix m*n, where m - number of patterns and n - size of vector to describe 1 pattern
                             __global float *clusters,///<[in] Numbers of cluster tensor m-size
                             __global float *means,///<[out] Means tensor k*n, where k - number of clasters and n - size of vector to describe 1 pattern
                             int total_m///< number of patterns
                            )
  {
   int i = get_global_id(0);
   int vector_size = get_global_size(0);
   int k = get_global_id(1);
   float sum = 0;
   int count = 0;
   for(int m = 0; m < total_m; m++)
     {
      if(clusters[m] != k)
         continue;
      sum += data[m * vector_size + i];
      count++;
     }
   if(count > 0)
      means[k * vector_size + i] = sum / count;
  }
//+------------------------------------------------------------------+
///\ingroup k-means K means clustering
/// Describes the process of loss function
///\details Detailed description on <A HREF="https://www.mql5.com/ru/articles/10947#loss">the link.</A>
//+------------------------------------------------------------------+
__kernel void KmeansLoss(__global float *data,///<[in] Inputs data matrix m*n, where m - number of patterns and n - size of vector to describe 1 pattern
                         __global float *clusters,///<[in] Numbers of cluster tensor m-size
                         __global float *means,///<[in] Means tensor k*n, where k - number of clasters and n - size of vector to describe 1 pattern
                         __global float *loss,///<[out] Loss tensor m-size
                         int vector_size///< Size of vector
                        )
  {
   int m = get_global_id(0);
   int c = clusters[m];
   int shift_c = c * vector_size;
   int shift_m = m * vector_size;
   float sum = 0;
   for(int i = 0; i < vector_size; i++)
      sum += pow(data[shift_m + i] - means[shift_c + i], 2);
   loss[m] = sum;
  }
//+------------------------------------------------------------------+
///\ingroup k-means K means clustering
/// Describes the calculation of probability
///\details Detailed description on <A HREF="https://www.mql5.com/ru/articles/10947#statistic">the link.</A>
//+------------------------------------------------------------------+
__kernel void KmeansStatistic(__global float *clusters,///<[in] Numbers of cluster tensor m-size
                              __global float *target,///<[in] Targets tensor 3*m, where m - number of patterns
                              __global float *probability,///<[out] Probability tensor 3*k, where k - number of clasters
                              int total_m///< number of patterns
                             )
  {
   int c = get_global_id(0);
   int shift_c = c * 3;
   float buy = 0;
   float sell = 0;
   float skip = 0;
   for(int i = 0; i < total_m; i++)
     {
      if(clusters[i] != c)
         continue;
      int shift = i * 3;
      buy += target[shift];
      sell += target[shift + 1];
      skip += target[shift + 2];
     }
//---
   int total = buy + sell + skip;
   if(total < 10)
     {
      probability[shift_c] = 0;
      probability[shift_c + 1] = 0;
      probability[shift_c + 2] = 0;
     }
   else
     {
      probability[shift_c] = buy / total;
      probability[shift_c + 1] = sell / total;
      probability[shift_c + 2] = skip / total;
     }
  }
//+------------------------------------------------------------------+
///\ingroup k-means K means clustering
/// Describes the process of softmax normalization distance for the class CKmeans (#CKmeans).
///\details Detailed description on <A HREF="https://www.mql5.com/ru/articles/10943#para4">the link.</A>
//+------------------------------------------------------------------+
__kernel void KmeansSoftMax(__global float *distance,///<[in] distance tensor m*k, where m - number of patterns and k - number of clasters
                            __global float *softmax,///<[out] softmax tensor m*k, where m - number of patterns and k - number of clasters
                            int total_k///< Number of clusters
                           )
  {
   int i = get_global_id(0);
   int shift = i * total_k;
   float m=distance[shift];
   for(int k = 1; k < total_k; k++)
      m =  max(distance[shift + k],m);
   float sum = 0;
   for(int k = 0; k < total_k; k++)
     {
      float value =  exp(1-distance[shift + k]/m);
      sum += value;
      softmax[shift + k] = value;
     }
   for(int k = 0; k < total_k; k++)
     {
      softmax[shift + k] /= sum;
     }
  }
//+------------------------------------------------------------------+
