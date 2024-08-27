/// \file
/// \brief NeuroNet.cl
/// Library consist OpenCL kernels
/// \author <A HREF="https://www.mql5.com/en/users/dng"> DNG </A>
/// \copyright Copyright 2019, DNG
//---
//--- by default some GPU doesn't support floats
//--- cl_khr_fp64 directive is used to enable work with floats
//#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#define l1 1.0e-4f
#define l2 1.0e-1f
#define MAX_WEIGHT 1.0e2f
#define LOCAL_ARRAY_SIZE   64
//+------------------------------------------------------------------+
///\ingroup neuron_base_ff Feed forward process kernel
/// Describes the forward path process for the Neuron Base (#CNeuronBaseOCL).
///\details Detailed description on <A HREF="https://www.mql5.com/ru/articles/8435#para41">the link.</A>
//+------------------------------------------------------------------+
__kernel void FeedForward(__global float *matrix_w,///<[in] Weights matrix (m+1)*n, where m - number of neurons in layer and n - number of outputs (neurons in next layer)
                          __global float *matrix_i,///<[in] Inputs tesor
                          __global float *matrix_o,///<[out] Output tensor
                          int inputs,///< Number of inputs
                          int activation///< Activation type (#ENUM_ACTIVATION)
                         )
  {
   int i = get_global_id(0);
   float sum = 0;
   float4 inp, weight;
   int shift = (inputs + 1) * i;
   for(int k = 0; k <= inputs; k = k + 4)
     {
      switch(inputs - k)
        {
         case 0:
            inp = (float4)(1, 0, 0, 0);
            weight = (float4)(matrix_w[shift + k], 0, 0, 0);
            break;
         case 1:
            inp = (float4)(matrix_i[k], 1, 0, 0);
            weight = (float4)(matrix_w[shift + k], matrix_w[shift + k + 1], 0, 0);
            break;
         case 2:
            inp = (float4)(matrix_i[k], matrix_i[k + 1], 1, 0);
            weight = (float4)(matrix_w[shift + k], matrix_w[shift + k + 1], matrix_w[shift + k + 2], 0);
            break;
         case 3:
            inp = (float4)(matrix_i[k], matrix_i[k + 1], matrix_i[k + 2], 1);
            weight = (float4)(matrix_w[shift + k], matrix_w[shift + k + 1], matrix_w[shift + k + 2], matrix_w[shift + k + 3]);
            break;
         default:
            inp = (float4)(matrix_i[k], matrix_i[k + 1], matrix_i[k + 2], matrix_i[k + 3]);
            weight = (float4)(matrix_w[shift + k], matrix_w[shift + k + 1], matrix_w[shift + k + 2], matrix_w[shift + k + 3]);
            break;
        }
      float d = dot(inp, weight);
      if(isnan(sum + d))
         continue;
      sum += d;
     }
   if(isnan(sum))
      sum = 0;
   switch(activation)
     {
      case 0:
         sum = tanh(sum);
         break;
      case 1:
         sum = 1 / (1 + exp(-sum));
         break;
      case 2:
         if(sum < 0)
            sum *= 0.01f;
         break;
      default:
         break;
     }
   matrix_o[i] = sum;
  }
//+------------------------------------------------------------------+
///\ingroup neuron_base_gr  Neuron Base Output Gradients Calculation kernel
/// Describes the process of output gradients calculation for the Neuron Base (#CNeuronBaseOCL).
///\details Detailed description on <A HREF="https://www.mql5.com/ru/articles/8435#para42">the link.</A>
//+------------------------------------------------------------------+
__kernel void CalcOutputGradient(__global float *matrix_t,///<[in] Target tensor
                                 __global float *matrix_o,///<[in] Output tensor
                                 __global float *matrix_ig,///<[out] Tensor of gradients
                                 int activation,///< Activation type (#ENUM_ACTIVATION)
                                 float error
                                )
  {
   int i = get_global_id(0);
   float out = matrix_o[i];
   float temp = 0;
   switch(activation)
     {
      case 0:
         //temp=clamp(matrix_t[i],-1.0,1.0)-out;
         temp = 2.0f * (matrix_t[i] - out);
         break;
      case 1:
         //temp=clamp(matrix_t[i],0.0,1.0)-out;
         temp = 2 * (matrix_t[i] - out) * error;
         temp = temp * out * (1 - out);
         break;
      case 2:
         //temp=(matrix_t[i]-out)*(out>=0 ? 1.0 : 0.01);
         temp = (2 * (matrix_t[i] - out) * error) * (out >= 0 ? 1.0f : 0.01f);
         break;
      default:
         temp = 2 * (matrix_t[i] - out) * error;
         break;
     }
   matrix_ig[i] = temp;
  }
//+------------------------------------------------------------------+
///\ingroup neuron_base_gr  Neuron Base Hidden Gradients Calculation kernel
/// Describes the process of hidden gradients calculation for the Neuron Base (#CNeuronBaseOCL).
///\details Detailed description on <A HREF="https://www.mql5.com/ru/articles/8435#para42">the link.</A>
//+------------------------------------------------------------------+
__kernel void CalcHiddenGradient(__global float *matrix_w,///<[in] Weights matrix (m+1)*n, where m - number of neurons in previous layer and n - number of neurons in current layer
                                 __global float *matrix_g,///<[in] Tensor of gradients at current layer
                                 __global float *matrix_o,///<[in] Previous layer Output tensor
                                 __global float *matrix_ig,///<[out] Tensor of gradients at previous layer
                                 int outputs,///< Number of outputs
                                 int activation///< Activation type (#ENUM_ACTIVATION)
                                )
  {
   int i = get_global_id(0);
   int inputs = get_global_size(0);
   float sum = 0;
   float out = matrix_o[i];
   float4 grad, weight;
   for(int k = 0; k < outputs; k += 4)
     {
      switch(outputs - k)
        {
         case 1:
            weight = (float4)(matrix_w[k * (inputs + 1) + i], 0, 0, 0);
            grad = (float4)(matrix_g[k], 0, 0, 0);
            break;
         case 2:
            grad = (float4)(matrix_g[k], matrix_g[k + 1], 0, 0);
            weight = (float4)(matrix_w[k * (inputs + 1) + i], matrix_w[(k + 1) * (inputs + 1) + i], 0, 0);
            break;
         case 3:
            grad = (float4)(matrix_g[k], matrix_g[k + 1], matrix_g[k + 2], 0);
            weight = (float4)(matrix_w[k * (inputs + 1) + i], matrix_w[(k + 1) * (inputs + 1) + i], matrix_w[(k + 2) * (inputs + 1) + i], 0);
            break;
         default:
            grad = (float4)(matrix_g[k], matrix_g[k + 1], matrix_g[k + 2], matrix_g[k + 3]);
            weight = (float4)(matrix_w[k * (inputs + 1) + i], matrix_w[(k + 1) * (inputs + 1) + i], matrix_w[(k + 2) * (inputs + 1) + i], matrix_w[(k + 3) * (inputs + 1) + i]);
            break;
        }
      sum += dot(grad, weight);
     }
   if(isnan(sum))
      sum = 0;
   switch(activation)
     {
      case 0:
         out = clamp(out, -1.0f, 1.0f);
         sum = clamp(sum + out, -1.0f, 1.0f) - out;
         sum = sum * max(1 - pow(out, 2), 1.0e-4f);
         break;
      case 1:
         out = clamp(out, 0.0f, 1.0f);
         sum = clamp(sum + out, 0.0f, 1.0f) - out;
         sum = sum * max(out * (1 - out), 1.0e-4f);
         break;
      case 2:
         if(out < 0)
            sum *= 0.01f;
         break;
      default:
         break;
     }
   matrix_ig[i] = sum;
  }
//+------------------------------------------------------------------+
///\ingroup neuron_base_opt  Neuron Base SGD Updating Weights Calculation kernel
/// Describes the process of SGD optimization weights for the Neuron Base (#CNeuronBaseOCL).
///\details Detailed description on <A HREF="https://www.mql5.com/ru/articles/8435#para43">the link.</A>
//+------------------------------------------------------------------+
__kernel void UpdateWeightsMomentum(__global float *matrix_w, ///<[in,out] Weights matrix (m+1)*n, where m - number of neurons in previous layer and n - number of neurons in current layer
                                    __global float *matrix_g,         ///<[in] Tensor of gradients at current layer
                                    __global float *matrix_i,         ///<[in] Inputs tesor
                                    __global float *matrix_dw,        ///<[in,out] Matrix of delta weights in last correction
                                    int inputs,                        ///< Number of inputs
                                    float learning_rates,             ///< Learning rates
                                    float momentum                    ///< Momentum multiplier
                                   )
  {
   int i = get_global_id(0);
   int j = get_global_id(1);
   int wi = i * (inputs + 1) + j;
   float delta = learning_rates * matrix_g[i] * (j < inputs ? matrix_i[j] : 1) + momentum * matrix_dw[wi];
   if(!isnan(delta))
     {
      matrix_dw[wi] = delta;
      if((delta * matrix_g[i]) > 0)
         matrix_w[wi] = clamp(matrix_w[wi] + delta, -MAX_WEIGHT, MAX_WEIGHT);
     }
  }
//+------------------------------------------------------------------+
///\ingroup neuron_base_opt  Neuron Base Adam Updating Weights Calculation kernel
/// Describes the process of Adam optimization weights for the Neuron Base (#CNeuronBaseOCL).
///\details Detailed description on <A HREF="https://www.mql5.com/ru/articles/8598#para31">the link.</A>
//+------------------------------------------------------------------+
__kernel void UpdateWeightsAdam(__global float *matrix_w,        ///<[in,out] Weights matrix (m+1)*n, where m - number of neurons in previous layer and n - number of neurons in current layer
                                __global const float *matrix_g,  ///<[in] Tensor of gradients at current layer
                                __global const float *matrix_i,  ///<[in] Inputs tesor
                                __global float *matrix_m,        ///<[in,out] Matrix of first momentum
                                __global float *matrix_v,        ///<[in,out] Matrix of seconfd momentum
                                const int inputs,                 ///< Number of inputs
                                const float l,                   ///< Learning rates
                                const float b1,                  ///< First momentum multiplier
                                const float b2                   ///< Second momentum multiplier
                               )
  {
   const int i = get_global_id(0);
   const int j = get_global_id(1);
   const int wi = i * (inputs + 1) + j * 4;
   float4 m, v, weight, inp;
   switch(inputs + 1 - j * 4)
     {
      case 0:
         inp = (float4)(1, 0, 0, 0);
         weight = (float4)(matrix_w[wi], 0, 0, 0);
         m = (float4)(matrix_m[wi], 0, 0, 0);
         v = (float4)(matrix_v[wi], 0, 0, 0);
         break;
      case 1:
         inp = (float4)(matrix_i[j * 4], 1, 0, 0);
         weight = (float4)(matrix_w[wi], matrix_w[wi + 1], 0, 0);
         m = (float4)(matrix_m[wi], matrix_m[wi + 1], 0, 0);
         v = (float4)(matrix_v[wi], matrix_v[wi + 1], 0, 0);
         break;
      case 2:
         inp = (float4)(matrix_i[j * 4], matrix_i[j * 4 + 1], 1, 0);
         weight = (float4)(matrix_w[wi], matrix_w[wi + 1], matrix_w[wi + 2], 0);
         m = (float4)(matrix_m[wi], matrix_m[wi + 1], matrix_m[wi + 2], 0);
         v = (float4)(matrix_v[wi], matrix_v[wi + 1], matrix_v[wi + 2], 0);
         break;
      case 3:
         inp = (float4)(matrix_i[j * 4], matrix_i[j * 4 + 1], matrix_i[j * 4 + 2], 1);
         weight = (float4)(matrix_w[wi], matrix_w[wi + 1], matrix_w[wi + 2], matrix_w[wi + 3]);
         m = (float4)(matrix_m[wi], matrix_m[wi + 1], matrix_m[wi + 2], matrix_m[wi + 3]);
         v = (float4)(matrix_v[wi], matrix_v[wi + 1], matrix_v[wi + 2], matrix_v[wi + 3]);
         break;
      default:
         inp = (float4)(matrix_i[j * 4], matrix_i[j * 4 + 1], matrix_i[j * 4 + 2], matrix_i[j * 4 + 3]);
         weight = (float4)(matrix_w[wi], matrix_w[wi + 1], matrix_w[wi + 2], matrix_w[wi + 3]);
         m = (float4)(matrix_m[wi], matrix_m[wi + 1], matrix_m[wi + 2], matrix_m[wi + 3]);
         v = (float4)(matrix_v[wi], matrix_v[wi + 1], matrix_v[wi + 2], matrix_v[wi + 3]);
         break;
     }
   float4 g = (float4)(matrix_g[i]) * inp;
   float4 mt = b1 * m + (1 - b1) * g;
   float4 vt = b2 * v + (1 - b2) * pow(g, 2);
   float4 delta = l * (mt / (sqrt(vt) + 1.0e-37f) - (l1 * sign(weight) + l2 * weight));
   switch(min(inputs + 1 - j * 4, 3))
     {
      case 3:
         if(delta.s3 * g.s3 > 0)
            matrix_w[wi + 3] = clamp(matrix_w[wi + 3] + delta.s3, -MAX_WEIGHT, MAX_WEIGHT);
         matrix_m[wi + 3] = mt.s3;
         matrix_v[wi + 3] = vt.s3;
      case 2:
         if(delta.s2 * g.s2 > 0)
            matrix_w[wi + 2] = clamp(matrix_w[wi + 2] + delta.s2, -MAX_WEIGHT, MAX_WEIGHT);
         matrix_m[wi + 2] = mt.s2;
         matrix_v[wi + 2] = vt.s2;
      case 1:
         if(delta.s1 * g.s1 > 0)
            matrix_w[wi + 1] = clamp(matrix_w[wi + 1] + delta.s1, -MAX_WEIGHT, MAX_WEIGHT);
         matrix_m[wi + 1] = mt.s1;
         matrix_v[wi + 1] = vt.s1;
      case 0:
         if(delta.s0 * g.s0 > 0)
            matrix_w[wi] = clamp(matrix_w[wi] + delta.s0, -MAX_WEIGHT, MAX_WEIGHT);
         matrix_m[wi] = mt.s0;
         matrix_v[wi] = vt.s0;
         break;
     }
  }
//+------------------------------------------------------------------+
///\ingroup neuron_base_opt  Neuron Base Least Squares Updating Weights Calculation kernel
/// Describes the process of Least Squares optimization weights for the Neuron Base (#CNeuronBaseOCL).
//\details Detailed description on <A HREF="https://www.mql5.com/ru/articles/8598#para31">the link.</A>
//+------------------------------------------------------------------+
__kernel void UpdateWeightsLS(__global float *matrix_w,        ///<[in,out] Weights matrix (m+1)*n, where m - number of neurons in previous layer and n - number of neurons in current layer
                              __global const float *matrix_g,  ///<[in] Tensor of gradients at current layer
                              __global const float *matrix_i,  ///<[in] Inputs tesor
                              __global float *matrix_xg,       ///<[in,out] Matrix of summ x*g
                              __global float *matrix_xx,       ///<[in,out] Matrix of summ x*x
                              const int inputs,                 ///< Number of inputs
                              const float l,                   ///< Learning rates
                              const int update                 ///< Update flag
                             )
  {
   const int i = get_global_id(0);
   const int j = get_global_id(1);
   const int wi = i * (inputs + 1) + j * 4;
   float4 xg, xx, weight, inp;
   switch(inputs + 1 - j * 4)
     {
      case 0:
         inp = (float4)(1, 0, 0, 0);
         weight = (float4)(matrix_w[wi], 0, 0, 0);
         break;
      case 1:
         inp = (float4)(matrix_i[j * 4], 1, 0, 0);
         weight = (float4)(matrix_w[wi], matrix_w[wi + 1], 0, 0);
         break;
      case 2:
         inp = (float4)(matrix_i[j * 4], matrix_i[j * 4 + 1], 1, 0);
         weight = (float4)(matrix_w[wi], matrix_w[wi + 1], matrix_w[wi + 2], 0);
         break;
      case 3:
         inp = (float4)(matrix_i[j * 4], matrix_i[j * 4 + 1], matrix_i[j * 4 + 2], 1);
         weight = (float4)(matrix_w[wi], matrix_w[wi + 1], matrix_w[wi + 2], matrix_w[wi + 3]);
         break;
      default:
         inp = (float4)(matrix_i[j * 4], matrix_i[j * 4 + 1], matrix_i[j * 4 + 2], matrix_i[j * 4 + 3]);
         weight = (float4)(matrix_w[wi], matrix_w[wi + 1], matrix_w[wi + 2], matrix_w[wi + 3]);
         break;
     }
   xg = (float4)(matrix_g[i]) * inp;
   xx = pow(inp, 2.0f);
   switch(min(inputs + 1 - j * 4, 3))
     {
      case 3:
         if(update)
           {
            matrix_w[wi + 3] = matrix_w[wi + 3] + l * (matrix_xg[wi + 3] + xg.s3) / (matrix_xx[wi + 3] + xx.s3 + 1.0e-37f);
            matrix_xg[wi + 3] = 0;
            matrix_xx[wi + 3] = 0;
           }
         else
           {
            matrix_xg[wi + 3] += xg.s3;
            matrix_xx[wi + 3] += xx.s3;
           }
      case 2:
         if(update)
           {
            matrix_w[wi + 2] = matrix_w[wi + 2] + l * (matrix_xg[wi + 2] + xg.s2) / (matrix_xx[wi + 2] + xx.s2 + 1.0e-37f);
            matrix_xg[wi + 2] = 0;
            matrix_xx[wi + 2] = 0;
           }
         else
           {
            matrix_xg[wi + 2] += xg.s2;
            matrix_xx[wi + 2] += xx.s2;
           }
      case 1:
         if(update)
           {
            matrix_w[wi + 1] = matrix_w[wi + 1] + l * (matrix_xg[wi + 1] + xg.s1) / (matrix_xx[wi + 1] + xx.s1 + 1.0e-37f);
            matrix_xg[wi + 1] = 0;
            matrix_xx[wi + 1] = 0;
           }
         else
           {
            matrix_xg[wi + 1] += xg.s1;
            matrix_xx[wi + 1] += xx.s1;
           }
      case 0:
         if(update)
           {
            matrix_w[wi] = matrix_w[wi] + l * (matrix_xg[wi] + xg.s0) / (matrix_xx[wi] + xx.s0 + 1.0e-37f);
            matrix_xg[wi] = 0;
            matrix_xx[wi] = 0;
           }
         else
           {
            matrix_xg[wi] += xg.s0;
            matrix_xx[wi] += xx.s0;
           }
         break;
     }
  }
//+------------------------------------------------------------------+
///\ingroup neuron_proof_ff
/// Kernel of the Pooling neuron for Feed forward process (#CNeuronProofOCL)
//+------------------------------------------------------------------+
__kernel void FeedForwardProof(__global float *matrix_i,   ///<[in] Inputs tesor
                               __global float *matrix_o,   ///<[out] Output tensor
                               int inputs,                   ///< Number of inputs
                               int window,                   ///< Size of input window
                               int step                      ///< Step size
                              )
  {
   int i = get_global_id(0);
   int pos = i * step;
   float result = matrix_i[pos];
   for(int k = 1; k < window; k = k + 1)
     {
      int shift = k + pos;
      if(shift >= inputs)
         break;
      result = max(result, matrix_i[shift]);
     }
   matrix_o[i] = result;
  }
//+------------------------------------------------------------------+
///\ingroup neuron_proof_gr
/// Kernel of the Pooling neuron to transfer gradient to previous layer (#CNeuronProofOCL)
//+------------------------------------------------------------------+
__kernel void CalcInputGradientProof(__global float *matrix_i,   ///<[in] Inputs tesor
                                     __global float *matrix_g,  ///<[in] Tensor of gradients at current layer
                                     __global float *matrix_o,  ///<[in] Output tensor
                                     __global float *matrix_ig, ///<[out] Tensor of gradients at previous layer
                                     int outputs,                ///< Number of outputs
                                     int window,                 ///< Size of input window
                                     int step                    ///< Step size
                                    )
  {
   int i = get_global_id(0);
   float prev_gradient = 0;
   float value = matrix_i[i];
   int start = i - window + step;
   start = (start - start % step) / step;
   int stop = (i - i % step) / step + 1;
   for(int out = max(0, start); out < min(outputs, stop); out++)
     {
      if(value == matrix_o[out])
         prev_gradient += matrix_g[out];
     }
   matrix_ig[i] = prev_gradient;
  }
//+------------------------------------------------------------------+
///\ingroup neuron_conv_ff
/// Kernel of the Convolution neuron for Feed forward process (#CNeuronConvOCL)
//+------------------------------------------------------------------+
__kernel void FeedForwardConv(__global float *matrix_w,             ///<[in] Weights matrix (m+1)*n, where m - input window and n - output window
                              __global float *matrix_i,             ///<[in] Inputs tesor
                              __global float *matrix_o,             ///<[out] Output tensor
                              int inputs,                            ///< Number of inputs
                              int step,                              ///< Step size
                              int window_in,                         ///< Size of input window
                              int window_out,                        ///< Size of output window
                              int activation                         ///< Activation type (#ENUM_ACTIVATION)
                             )
  {
   int i = get_global_id(0);
   int w_in = window_in;
   int w_out = window_out;
   float sum = 0;
   float4 inp, weight;
   int shift_out = w_out * i;
   int shift_in = step * i;
   for(int out = 0; out < w_out; out++)
     {
      int shift = (w_in + 1) * out;
      int stop = (w_in <= (inputs - shift_in) ? w_in : (inputs - shift_in));
      for(int k = 0; k <= stop; k += 4)
        {
         switch(stop - k)
           {
            case 0:
               inp = (float4)(1, 0, 0, 0);
               weight = (float4)(matrix_w[shift + k], 0, 0, 0);
               break;
            case 1:
               inp = (float4)(matrix_i[shift_in + k], 1, 0, 0);
               weight = (float4)(matrix_w[shift + k], matrix_w[shift + k + 1], 0, 0);
               break;
            case 2:
               inp = (float4)(matrix_i[shift_in + k], matrix_i[shift_in + k + 1], 1, 0);
               weight = (float4)(matrix_w[shift + k], matrix_w[shift + k + 1], matrix_w[shift + k + 2], 0);
               break;
            case 3:
               inp = (float4)(matrix_i[shift_in + k], matrix_i[shift_in + k + 1], matrix_i[shift_in + k + 2], 1);
               weight = (float4)(matrix_w[shift + k], matrix_w[shift + k + 1], matrix_w[shift + k + 2], matrix_w[shift + k + 3]);
               break;
            default:
               inp = (float4)(matrix_i[shift_in + k], matrix_i[shift_in + k + 1], matrix_i[shift_in + k + 2], matrix_i[shift_in + k + 3]);
               weight = (float4)(matrix_w[shift + k], matrix_w[shift + k + 1], matrix_w[shift + k + 2], matrix_w[shift + k + 3]);
               break;
           }
         sum += dot(inp, weight);
        }
      if(isnan(sum))
         sum = 0;
      switch(activation)
        {
         case 0:
            sum = tanh(sum);
            break;
         case 1:
            sum = 1 / (1 + exp(-clamp(sum, -20.0f, 20.0f)));
            break;
         case 2:
            if(sum < 0)
               sum *= 0.01f;
            break;
         default:
            break;
        }
      matrix_o[out + shift_out] = sum;
     }
  }
//+------------------------------------------------------------------+
///\ingroup neuron_conv_gr
/// Kernel of the Convolution neuron to transfer gradient to previous layer (#CNeuronConvOCL)
//+------------------------------------------------------------------+
__kernel void CalcHiddenGradientConv(__global float *matrix_w,   ///<[in] Weights matrix (m+1)*n, where m - input window and n - output window
                                     __global float *matrix_g,   ///<[in] Tensor of gradients at current layer
                                     __global float *matrix_o,    ///<[in] Output tensor
                                     __global float *matrix_ig,   ///<[out] Tensor of gradients at previous layer
                                     int outputs,                  ///< Number of outputs
                                     int step,                     ///< Step size
                                     int window_in,                ///< Size of input window
                                     int window_out,               ///< Size of output window
                                     int activation,               ///< Activation type (#ENUM_ACTIVATION)
                                     int shift_out                 ///< Shift in output and gradient buffer
                                    )
  {
   int i = get_global_id(0);
   int inputs = get_global_size(0);
   float sum = 0;
   float out = matrix_o[shift_out + i];
   int start = i - window_in + step;
   start = max((start - start % step) / step, 0);
   int stop = (i - i % step) / step + 1;
   if(stop > (outputs / window_out))
      stop = outputs / window_out;
   for(int h = 0; h < window_out; h += 4)
     {
      for(int k = start; k < stop; k++)
        {
         int shift_g = k * window_out + h;
         int shift_w = (stop - k - 1) * step + i % step + h * (window_in + 1);
         if(shift_g >= outputs || shift_w >= (window_in + 1)*window_out)
            break;
         float grad = matrix_g[shift_out + shift_g];
         if(grad == 0.0f)
            continue;
         sum += grad * matrix_w[shift_w];
        }
     }
   if(isnan(sum))
      sum = 0;
   if(sum != 0)
      switch(activation)
        {
         case 0:
            sum = clamp(sum + out, -1.0f, 1.0f) - out;
            sum = sum * (1.0f - pow(out >= 1 || out <= -1 ? 1.0f : out, 2.0f));
            break;
         case 1:
            sum = clamp(sum + out, 0.0f, 1.0f) - out;
            sum = sum * (out * (1.0f - out));
            break;
         case 2:
            if(out < 0.0f)
               sum *= 0.01f;
            break;
         default:
            break;
        }
   matrix_ig[i] = sum;
  }
//+------------------------------------------------------------------+
///\ingroup neuron_conv_opt Convolution Neuron SGD optimization Updating Weights Calculation kernel
/// Describes the process of SGD optimization weights for the Convolution Neuron (#CNeuronConvOCL).
//+------------------------------------------------------------------+
__kernel void UpdateWeightsConvMomentum(__global float *matrix_w,   ///<[in,out] Weights matrix (m+1)*n, where m - input window and n - output window
                                        __global float *matrix_g,    ///<[in] Tensor of gradients at current layer
                                        __global float *matrix_i,    ///<[in] Inputs tesor
                                        __global float *matrix_dw,   ///<[in,out] Matrix of delta weights in last correction
                                        int inputs,                   ///< Number of inputs
                                        float learning_rates,        ///< Learning rates
                                        float momentum,              ///< Momentum multiplier
                                        int window_in,                ///< Size of input window
                                        int window_out,               ///< Size of output window
                                        int step                      ///< Step size
                                       )
  {
   const int i = get_global_id(0);
   const int shift = i % (window_in + 1);
   const int shift_out = (i - shift) / (window_in + 1);
   int total = (inputs - window_in) % step;
   total = (inputs - window_in - total) / step + (total > 0 ? 1 : 0);
   float grad = 0;
   for(int t = 0; t < total; t++)
     {
      if(shift != window_in && (shift + t * window_in) >= inputs)
         break;
      grad += matrix_g[t * window_out + shift_out] * (shift == window_in ? 1 : matrix_i[shift + t * step]);
     }
   float delta = learning_rates * grad + momentum * matrix_dw[i];
   if(!isnan(delta))
     {
      matrix_dw[i] = delta;
      if(delta * grad > 0)
         matrix_w[i] = clamp(matrix_w[i] + delta, -MAX_WEIGHT, MAX_WEIGHT);
     }
  }
//+------------------------------------------------------------------+
///\ingroup neuron_conv_opt Convolution Neuron Adam optimization Updating Weights Calculation kernel
/// Describes the process of Adam optimization weights for the Convolution Neuron (#CNeuronConvOCL).
//+------------------------------------------------------------------+
__kernel void UpdateWeightsConvAdam(__global float *matrix_w,    ///<[in,out] Weights matrix (m+1)*n, where m - input window and n - output window
                                    __global const float *matrix_g,  ///<[in] Tensor of gradients at current layer
                                    __global const float *matrix_i,  ///<[in] Inputs tesor
                                    __global float *matrix_m,        ///<[in] Matrix of first momentum
                                    __global float *matrix_v,        ///<[in] Matrix of seconfd momentum
                                    const int inputs,                 ///< Number of inputs
                                    const float l,                   ///< Learning rates
                                    const float b1,                  ///< First momentum multiplier
                                    const float b2,                  ///< Second momentum multiplier
                                    int window_in,                    ///< Size of input window
                                    int window_out,                   ///< Size of output window
                                    int step                          ///< Step size
                                   )
  {
   const int i = get_global_id(0);
   if(i > window_in)
      return;
//---
   int total = (inputs - (window_in - step)) % step;
   total = (inputs - (window_in - step) - total) / step + (total > 0 ? 1 : 0);
   for(int out = 0; out < window_out; out++)
     {
      if((window_out - out) > 4)
        {
         float4 grad = {0, 0, 0, 0};
         int shift_w = i + out * (window_in + 1);
         for(int t = 0; t < total; t++)
           {
            if(i != window_in && (i + t * window_in) >= inputs)
               break;
            grad += (float4)(matrix_g[t * window_out + out], matrix_g[t * window_out + out + 1], matrix_g[t * window_out + out + 2], matrix_g[t * window_out + out + 3]) * (i == window_in ? 1 : matrix_i[i + t * step]);
           }
         float4 mt = clamp(b1 * (float4)(matrix_m[shift_w], matrix_m[shift_w + window_in + 1], matrix_m[shift_w + 2 * (window_in + 1)], matrix_m[shift_w + 3 * (window_in + 1)]) + (1 - b1) * grad, -1.0e5f, 1.0e5f);
         float4 vt = clamp(b2 * (float4)(matrix_v[shift_w], matrix_v[shift_w + window_in + 1], matrix_v[shift_w + 2 * (window_in + 1)], matrix_v[shift_w + 3 * (window_in + 1)]) + (1 - b2) * pow(grad, 2), 1.0e-6f, 1.0e6f);
         float4 delta = l * mt / sqrt(vt);
         float4 weight = clamp((float4)(matrix_w[shift_w], matrix_w[shift_w + (window_in + 1)], matrix_w[shift_w + 2 * (window_in + 1)], matrix_w[shift_w + 3 * (window_in + 1)]) + delta, -MAX_WEIGHT, MAX_WEIGHT);
         if(delta.s0 * grad.s0 > 0)
            matrix_w[shift_w] = weight.s0;
         if(delta.s1 * grad.s1 > 0)
            matrix_w[shift_w + (window_in + 1)] = weight.s1;
         if(delta.s2 * grad.s2 > 0)
            matrix_w[shift_w + 2 * (window_in + 1)] = weight.s2;
         if(delta.s3 * grad.s3 > 0)
            matrix_w[shift_w + 3 * (window_in + 1)] = weight.s3;
         matrix_m[shift_w] = mt.s0;
         matrix_m[shift_w + (window_in + 1)] = mt.s1;
         matrix_m[shift_w + 2 * (window_in + 1)] = mt.s2;
         matrix_m[shift_w + 3 * (window_in + 1)] = mt.s3;
         matrix_v[shift_w] = vt.s0;
         matrix_v[shift_w + (window_in + 1)] = vt.s1;
         matrix_v[shift_w + 2 * (window_in + 1)] = vt.s2;
         matrix_v[shift_w + 3 * (window_in + 1)] = vt.s3;
         out += 3;
        }
      else
        {
         float grad = 0;
         int shift_w = i + out * (window_in + 1);
         for(int t = 0; t < total; t++)
           {
            if(i != window_in && (i + t * window_in) >= inputs)
               break;
            grad += matrix_g[t * window_out + out] * (i == window_in ? 1 : matrix_i[i + t * step]);
           }
         float mt = clamp(b1 * matrix_m[shift_w] + (1 - b1) * grad, -1.0e5f, 1.0e5f);
         float vt = clamp(b2 * matrix_v[shift_w] + (1 - b2) * pow(grad, 2), 1.0e-6f, 1.0e6f);
         float delta = l * mt / sqrt(vt);
         if(delta * grad > 0)
            matrix_w[shift_w] = clamp(matrix_w[shift_w] + delta, -MAX_WEIGHT, MAX_WEIGHT);
         matrix_m[shift_w] = mt;
         matrix_v[shift_w] = vt;
        }
     }
  }
//+------------------------------------------------------------------+
///\ingroup neuron_conv_opt Convolution Neuron Least Squares optimization Updating Weights Calculation kernel
/// Describes the process of Least Squares optimization weights for the Convolution Neuron (#CNeuronConvOCL).
//+------------------------------------------------------------------+
__kernel void UpdateWeightsConvLS(__global float *matrix_w,    ///<[in,out] Weights matrix (m+1)*n, where m - input window and n - output window
                                  __global const float *matrix_g,  ///<[in] Tensor of gradients at current layer
                                  __global const float *matrix_i,  ///<[in] Inputs tesor
                                  __global float *matrix_xg,        ///<[in] Matrix of summ x*g
                                  __global float *matrix_xx,        ///<[in] Matrix of summ x*x
                                  const int inputs,                 ///< Number of inputs
                                  const float l,                   ///< Learning rates
                                  const int update,                 ///< Update flag
                                  int window_in,                    ///< Size of input window
                                  int window_out,                   ///< Size of output window
                                  int step                          ///< Step size
                                 )
  {
   const int i = get_global_id(0);
   if(i > window_in)
      return;
//---
   int total = (inputs - (window_in - step)) % step;
   total = (inputs - (window_in - step) - total) / step + (total > 0 ? 1 : 0);
   for(int out = 0; out < window_out; out++)
     {
      if((window_out - out) > 4)
        {
         float4 xg = {0, 0, 0, 0};
         float x2 = 0;
         int shift_w = i + out * (window_in + 1);
         for(int t = 0; t < total; t++)
           {
            if(i != window_in && (i + t * window_in) >= inputs)
               break;
            xg += (float4)(matrix_g[t * window_out + out], matrix_g[t * window_out + out + 1], matrix_g[t * window_out + out + 2], matrix_g[t * window_out + out + 3]) * (i == window_in ? 1 : matrix_i[i + t * step]);
            x2 += (i == window_in ? 1 : pow(matrix_i[i + t * step], 2.0f));
           }
         if(update)
           {
            xg = (float4)(matrix_xg[shift_w], matrix_xg[shift_w + window_in + 1], matrix_xg[shift_w + 2 * (window_in + 1)], matrix_xg[shift_w + 3 * (window_in + 1)]) + xg;
            float4 xx = (float4)(matrix_xx[shift_w], matrix_xx[shift_w + window_in + 1], matrix_xx[shift_w + 2 * (window_in + 1)], matrix_xx[shift_w + 3 * (window_in + 1)]) + x2;
            float4 delta = l * xg / (xx + 1.0e-37f);
            float4 weight = (float4)(matrix_w[shift_w], matrix_w[shift_w + (window_in + 1)], matrix_w[shift_w + 2 * (window_in + 1)], matrix_w[shift_w + 3 * (window_in + 1)]) + delta;
            matrix_w[shift_w] = weight.s0;
            matrix_w[shift_w + (window_in + 1)] = weight.s1;
            matrix_w[shift_w + 2 * (window_in + 1)] = weight.s2;
            matrix_w[shift_w + 3 * (window_in + 1)] = weight.s3;
            matrix_xg[shift_w] = 0;
            matrix_xg[shift_w + (window_in + 1)] = 0;
            matrix_xg[shift_w + 2 * (window_in + 1)] = 0;
            matrix_xg[shift_w + 3 * (window_in + 1)] = 0;
            matrix_xx[shift_w] = 0;
            matrix_xx[shift_w + (window_in + 1)] = 0;
            matrix_xx[shift_w + 2 * (window_in + 1)] = 0;
            matrix_xx[shift_w + 3 * (window_in + 1)] = 0;
           }
         else
           {
            matrix_xg[shift_w] += xg.s0;
            matrix_xg[shift_w + (window_in + 1)] += xg.s1;
            matrix_xg[shift_w + 2 * (window_in + 1)] += xg.s2;
            matrix_xg[shift_w + 3 * (window_in + 1)] += xg.s3;
            matrix_xx[shift_w] = matrix_xx[shift_w + (window_in + 1)] = matrix_xx[shift_w + 2 * (window_in + 1)] = matrix_xx[shift_w + 3 * (window_in + 1)] += x2;
           }
         out += 3;
        }
      else
        {
         float xg = 0;
         float xx = 0;
         int shift_w = i + out * (window_in + 1);
         for(int t = 0; t < total; t++)
           {
            if(i != window_in && (i + t * window_in) >= inputs)
               break;
            xg += matrix_g[t * window_out + out] * (i == window_in ? 1 : matrix_i[i + t * step]);
            xx += (i == window_in ? 1 : pow(matrix_i[i + t * step], 2.0f));
           }
         if(update)
           {
            xg = matrix_xg[shift_w] + xg;
            xx = matrix_xx[shift_w] + xx;
            float delta = l * xg / (xx + 1.0e-37f);
            matrix_w[shift_w] = matrix_w[shift_w] + delta;
            matrix_xg[shift_w] = 0;
            matrix_xx[shift_w] = 0;
           }
         else
           {
            matrix_xg[shift_w] += xg;
            matrix_xx[shift_w] += xx;
           }
        }
     }
  }
//+------------------------------------------------------------------+
///\ingroup neuron_atten_ff Attention Neuron Score calculation kernel                                                                  |
/// Describes the Score calculation process for the Neuron of attention layer (#CNeuronAttentionOCL).
///\details Detailed description on <A HREF="https://www.mql5.com/ru/articles/8765#para43">the link.</A>
//+------------------------------------------------------------------+
__kernel void AttentionScore(__global float *querys, ///<[in] Matrix of Querys
                             __global float *keys,   ///<[in] Matrix of Keys
                             __global float *score,  ///<[out] Matrix of Scores
                             int dimension,           ///< Dimension of Key
                             int mask                 ///< 1 - calc only previous units, 0 - calc all
                            )
  {
   int q = get_global_id(0);
   int shift_q = q * dimension;
   int units = get_global_size(0);
   int shift_s = q * units;
   float koef = sqrt((float)dimension);
   if(koef < 1)
      koef = 1;
   float sum = 0;
   for(int k = 0; k < units; k++)
     {
      if(mask > 0 && k > q)
        {
         score[shift_s + k] = 0;
         continue;
        }
      float result = 0;
      int shift_k = k * dimension;
      for(int i = 0; i < dimension; i++)
         result += (querys[shift_q + i] * keys[shift_k + i]);
      result = exp(result / koef);
      if(isnan(result))
         result = 0;
      score[shift_s + k] = result;
      sum += result;
     }
   for(int k = 0; (k < units && sum > 0); k++)
      score[shift_s + k] /= sum;
  }
//+------------------------------------------------------------------+
///\ingroup neuron_atten_ff Attention Neuron Out calculation kernel
/// Describes the Attention out calculation process for the Neuron of attention layer (#CNeuronAttentionOCL).
///\details Detailed description on <A HREF="https://www.mql5.com/ru/articles/8765#para43">the link.</A>
//+------------------------------------------------------------------+
__kernel void AttentionOut(__global float *scores,   ///<[in] Matrix of Scores
                           __global float *values,   ///<[in] Matrix of Values
                           __global float *inputs,   ///<[in] Inputs tesor
                           __global float *out       ///<[out] Output tesor
                          )
  {
   int units = get_global_size(0);
   int u = get_global_id(0);
   int d = get_global_id(1);
   int dimension = get_global_size(1);
   int shift = u * dimension + d;
   float result = 0;
   for(int i = 0; i < units; i++)
      result += scores[u * units + i] * values[i * dimension + d];
   out[shift] = (isnan(result) ? 0 : result) + inputs[shift];
  }
//+------------------------------------------------------------------+
///\ingroup neuron_atten_ff Kernel for calculation Sum of 2 matrixs with multiplyer.
/// Describes the calculation Sum of 2 matrixs.
///\details Detailed description on <A HREF="https://www.mql5.com/ru/articles/8765#para43">the link.</A>
//+------------------------------------------------------------------+
__kernel void SumMatrix(__global float *matrix1,     ///<[in] First matrix
                        __global float *matrix2,     ///<[in] Second matrix
                        __global float *matrix_out,  ///<[out] Output matrix
                        int dimension,               ///< Dimension of matrix
                        float multiplyer,            ///< Multiplyer for output
                        int shift_in1,               ///< Shift for input 1
                        int shift_in2,               ///< Shift for input 2
                        int shift_out                ///< Shift for output
                       )
  {
   const int i = get_global_id(0);
   const int step = get_global_size(0);;
   for(int k = 0; k < dimension; k++)
     {
      int index = i + k * step;
      matrix_out[shift_out + index] = (matrix1[shift_in1 + index] + matrix2[shift_in2 + index]) * multiplyer;
     }
  }
//+------------------------------------------------------------------+
///\ingroup neuron_atten_ff Kernel for calculation Sum of 4 matrixs with multiplyer.
/// Describes the calculation Sum of 4 matrixs.
///\details Detailed description on <A HREF="https://www.mql5.com/ru/articles/8909#para53">the link.</A>
//+------------------------------------------------------------------+
__kernel void Sum5Matrix(__global float *matrix1,     ///<[in] First matrix
                         __global float *matrix2,     ///<[in] Second matrix
                         __global float *matrix3,     ///<[in] Third matrix
                         __global float *matrix4,     ///<[in] Fourth matrix
                         __global float *matrix5,     ///<[in] Fifth matrix
                         __global float *matrix_out,  ///<[out] Output matrix
                         int dimension,                ///< Dimension of matrix
                         float multiplyer             ///< Multiplyer for output
                        )
  {
   const int i = get_global_id(0) * dimension;
   for(int k = 0; k < dimension; k++)
      matrix_out[i + k] = (matrix1[i + k] + matrix2[i + k] + matrix3[i + k] + matrix4[i + k] + matrix5[i + k]) * multiplyer;
  }
//+------------------------------------------------------------------+
///\ingroup neuron_atten_gr Attention layer's neuron Gradients Calculation kernel
/// Describes the gradients calculation process for the Neuron of attention layer (#CNeuronAttentionOCL).
///\details Detailed description on <A HREF="https://www.mql5.com/ru/articles/8765#para44">the link.</A>
/// @param[in] querys Matrix of Querys
/// @param[out] querys_g Matrix of Querys' Gradients
/// @param[in] keys Matrix of Keys
/// @param[out] keys_g Matrix of Keys' Gradients
/// @param[in] values Matrix of Values
/// @param[out] values_g Matrix of Values' Gradients
/// @param[in] scores Matrix of Scores
/// @param[in] gradient Matrix of Gradients from previous iteration
//+------------------------------------------------------------------+
__kernel void AttentionInsideGradients(__global float *querys, __global float *querys_g,
                                       __global float *keys, __global float *keys_g,
                                       __global float *values, __global float *values_g,
                                       __global float *scores,
                                       __global float *gradient)
  {
   int u = get_global_id(0);
   int d = get_global_id(1);
   int units = get_global_size(0);
   int dimension = get_global_size(1);
   float koef = sqrt((float)dimension);
   if(koef < 1)
      koef = 1;
   float vg = 0;
   float qg = 0;
   float kg = 0;
   for(int iu = 0; iu < units; iu++)
     {
      float g = gradient[iu * dimension + d];
      float sc = scores[iu * units + u];
      vg += sc * g;
      //---
      float sqg = 0;
      float skg = 0;
      for(int id = 0; id < dimension; id++)
        {
         sqg += values[iu * dimension + id] * gradient[u * dimension + id];
         skg += values[u * dimension + id] * gradient[iu * dimension + id];
        }
      qg += (scores[u * units + iu] == 0 || scores[u * units + iu] == 1 ? 0.0001f : scores[u * units + iu] * (1 - scores[u * units + iu])) * sqg * keys[iu * dimension + d] / koef;
      //---
      kg += (scores[iu * units + u] == 0 || scores[iu * units + u] == 1 ? 0.0001f : scores[iu * units + u] * (1 - scores[iu * units + u])) * skg * querys[iu * dimension + d] / koef;
     }
   int shift = u * dimension + d;
   values_g[shift] = clamp((isnan(vg) ? 0.0f : vg), -1.0f, 1.0f);
   querys_g[shift] = clamp((isnan(qg) ? 0.0f : qg), -1.0f, 1.0f);
   keys_g[shift] = clamp((isnan(kg) ? 0.0f : kg), -1.0f, 1.0f);
  }
//+------------------------------------------------------------------+
///\ingroup neuron_norm Kernels of matrix normalization process
/// Describes the process of matrix normalization.
///\details Detailed description on <A HREF="https://arxiv.org/abs/1607.06450">the link.</A>
/// @param[in,out] buffer In/Out Matrix
/// @param[in] dimension Dimension of matrix
//+------------------------------------------------------------------+
__kernel void Normalize(__global float *buffer,
                        int dimension)
  {
   int n = get_global_id(0);
   int shift = n * dimension;
   if(dimension <= 0)
      return;
//---
   float mean = 0;
   for(int i = 0; i < dimension; i++)
     {
      if(isnan(buffer[shift + i]))
         buffer[shift + i] = 0;
      else
         mean += buffer[shift + i] / dimension;
     }
   float variance = 0;
   for(int i = 0; i < dimension; i++)
      variance += pow(buffer[shift + i] - mean, 2) / dimension;
   variance = sqrt((isnan(variance) ? 0 : variance));
   for(int i = 0; i < dimension; i++)
      buffer[shift + i] = (buffer[shift + i] - mean) / (variance > 0 ? variance : 1.0f);
  }
//+------------------------------------------------------------------+
///\ingroup neuron_norm Kernels of weights matrix normalization process
/// Describes the process of weights matrix normalization.
///\details Detailed description on <A HREF="https://arxiv.org/abs/1607.06450">the link.</A>
/// @param[in,out] buffer In/Out Matrix
/// @param[in] dimension Dimension of matrix
//+------------------------------------------------------------------+
__kernel void NormalizeWeights(__global float *buffer,
                               int dimension)
  {
   int n = get_global_id(0);
   int shift = n * dimension;
   float sum = 0;
   float k = 1;
   do
     {
      for(int i = 0; (i < dimension && !isnan(sum)); i++)
         sum = pow(buffer[shift + i] / k, 2) / dimension;
      if(isnan(sum))
         k *= 10;
     }
   while(isnan(sum));
   sum = sqrt(sum);
   if(k * sum > 1)
      for(int i = 0; i < dimension; i++)
         buffer[shift + i] /= k * sum;
  }
//+------------------------------------------------------------------+
///\ingroup neuron_atten_ff
/// Describes the process of concatenate 4 matrices.
///\details Detailed description on <A HREF="https://www.mql5.com/ru/articles/8909#para52">the link.</A>
/// @param[in] input1, input2, input3, input4 Input buffers
/// @param[in] window1, window2, window3, window4 Windows for every buffer
/// @param[out] output Output buffer
//+------------------------------------------------------------------+
__kernel void ConcatenateBuffers(__global float *input1, int window1,
                                 __global float *input2, int window2,
                                 __global float *input3, int window3,
                                 __global float *input4, int window4,
                                 __global float *output)
  {
   int n = get_global_id(0);
   int shift = n * (window1 + window2 + window3 + window4);
   int shift_in = n * window1;
   for(int i = 0; i < window1; i++)
      output[shift + i] = input1[shift_in + i];
//---
   shift += window1;
   shift_in = n * window2;
   for(int i = 0; i < window2; i++)
      output[shift + i] = input2[shift_in + i];
//---
   shift += window2;
   shift_in = n * window3;
   for(int i = 0; i < window3; i++)
      output[shift + i] = input3[shift_in + i];
//---
   shift += window3;
   shift_in = n * window4;
   for(int i = 0; i < window4; i++)
      output[shift + i] = input4[shift_in + i];
  }
//+------------------------------------------------------------------+
///\ingroup neuron_atten_gr
/// Describes the process of deconcatenate matrix.
///\details Detailed description on <A HREF="https://www.mql5.com/ru/articles/8909#para53">the link.</A>
/// @param[in] output1, output2, output3, output4 Output buffers
/// @param[in] window1, window2, window3, window4 Windows for every buffer
/// @param[out] inputs Input buffer
//+------------------------------------------------------------------+
__kernel void DeconcatenateBuffers(__global float *output1, int window1,
                                   __global float *output2, int window2,
                                   __global float *output3, int window3,
                                   __global float *output4, int window4,
                                   __global float *inputs)
  {
   int n = get_global_id(0);
//--- Head 1
   int shift = n * (window1 + window2 + window3 + window4);
   int shift_out = n * window1;
   for(int i = 0; i < window1; i++)
      output1[shift_out + i] = inputs[shift + i];
//--- Head 2
   shift += window1;
   shift_out = n * window2;
   for(int i = 0; i < window2; i++)
      output2[shift_out + i] = inputs[shift + i];
//--- Head 3
   shift += window2;
   shift_out = n * window3;
   for(int i = 0; i < window3; i++)
      output3[shift_out + i] = inputs[shift + i];
//--- Head 4
   shift += window3;
   shift_out = n * window4;
   for(int i = 0; i < window4; i++)
      output4[shift_out + i] = inputs[shift + i];
  }
//+------------------------------------------------------------------+
///\ingroup neuron_atten_ff Multi-Heads Attention Neuron Score calculation kernel
/// Describes the Score calculation process for the Neuron of multi-heads attention layer (#CNeuronMLMHAttentionOCL).
///\details Detailed description on <A HREF="https://www.mql5.com/ru/articles/9025#para42">the link.</A>
//+------------------------------------------------------------------+
__kernel void MHAttentionScore(__global float *qkv,    ///<[in] Matrix of Querys, Keys, Values
                               __global float *score,  ///<[out] Matrix of Scores
                               int dimension,          ///< Dimension of Key
                               int mask                ///< 1 - calc only previous units, 0 - calc all
                              )
  {
   int q = get_global_id(0);
   int h = get_global_id(1);
   int units = get_global_size(0);
   int heads = get_global_size(1);
//---
   int shift_q = dimension * (h + 3 * q * heads);
   int shift_s = units * (h + q * heads);
//---
   float koef = sqrt((float)dimension);
   if(koef < 1)
      koef = 1;
   float sum = 0;
   for(int k = 0; k < units; k++)
     {
      if(mask > 0 && k > q)
        {
         score[shift_s + k] = 0;
         continue;
        }
      float result = 0;
      int shift_k = dimension * (h + heads * (3 * k + 1));
      for(int i = 0; i < dimension; i++)
        {
         if((dimension - i) > 4)
           {
            result += dot((float4)(qkv[shift_q + i], qkv[shift_q + i + 1], qkv[shift_q + i + 2], qkv[shift_q + i + 3]),
                          (float4)(qkv[shift_k + i], qkv[shift_k + i + 1], qkv[shift_k + i + 2], qkv[shift_k + i + 3]));
            i += 3;
           }
         else
            result += (qkv[shift_q + i] * qkv[shift_k + i]);
        }
      result = exp(clamp(result / koef, -30.0f, 30.0f));
      if(isnan(result))
         result = 0;
      score[shift_s + k] = result;
      sum += result;
     }
   for(int k = 0; (k < units && sum > 1); k++)
      score[shift_s + k] /= sum;
  }
//+------------------------------------------------------------------+
///\ingroup neuron_atten_ff Multi-heads Attention Neuron Out calculation kernel
/// Describes the Multi-heads Attention out calculation process for the Neuron of multi-heads attention layer (#CNeuronMLMHAttentionOCL).
///\details Detailed description on <A HREF="https://www.mql5.com/ru/articles/9025#para42">the link.</A>
//+------------------------------------------------------------------+
__kernel void MHAttentionOut(__global float *scores, ///<[in] Matrix of Scores
                             __global float *qkv,    ///<[in] Matrix of Values
                             __global float *out,    ///<[out] Output tesor
                             int dimension           ///< Dimension of Value
                            )
  {
   int u = get_global_id(0);
   int units = get_global_size(0);
   int h = get_global_id(1);
   int heads = get_global_size(1);
//---
   int shift_s = units * (h + heads * u);
   int shift_out = dimension * (h + heads * u);
   int layer = 3 * dimension * heads;
//---
   for(int d = 0; d < dimension; d++)
     {
      float result = 0;
      for(int v = 0; v < units; v ++)
        {
         int shift_v = dimension * (h + heads * (3 * v + 2)) + d;
         result += scores[shift_s + v] * qkv[shift_v];
        }
      out[shift_out + d] = result;
     }
  }
//+------------------------------------------------------------------+
///\ingroup neuron_atten_gr Attention layer's neuron Gradients Calculation kernel
/// Describes the gradients calculation process for the Neuron of attention layer (#CNeuronMLMHAttentionOCL).
///\details Detailed description on <A HREF="https://www.mql5.com/ru/articles/9025#para33">the link.</A>
/// @param[in] qkv Matrix of Querys, Keys and Values
/// @param[out] qkv_g Matrix of Querys', Keys' and Values' Gradients
/// @param[in] scores Matrix of Scores
/// @param[in] scores_g Matrix of Scores' Gradients
/// @param[in] gradient Matrix of Gradients from previous iteration
/// @param[in] dimension Dimension of Key vector
//+------------------------------------------------------------------+
__kernel void MHAttentionInsideGradients(__global float *qkv, __global float *qkv_g,
      __global float *scores, __global float *scores_g,
      __global float *gradient, int dimension)
  {
   int u = get_global_id(0);
   int h = get_global_id(1);
   int units = get_global_size(0);
   int heads = get_global_size(1);
   float koef = sqrt((float)dimension);
   if(koef < 1)
      koef = 1;
//--- Calculating score's gradients
   uint shift_s = units * (h + u * heads);
   for(int v = 0; v < units; v++)
     {
      float s = scores[shift_s + v];
      if(s <= 0)
         continue;
      float sg = 0;
      int shift_v = dimension * (h + heads * (3 * v + 2));
      int shift_g = dimension * (h + heads * v);
      for(int d = 0; d < dimension; d++)
         sg += qkv[shift_v + d] * gradient[shift_g + d];
      scores_g[shift_s + v] = sg * (s < 1 ? s * (1 - s) : 1) / koef;
     }
   barrier(CLK_GLOBAL_MEM_FENCE);
//--- Calculating gradients for Query, Key and Value
   uint shift_qg = dimension * (h + 3 * u * heads);
   uint shift_kg = dimension * (h + (3 * u + 1) * heads);
   uint shift_vg = dimension * (h + (3 * u + 2) * heads);
   for(int d = 0; d < dimension; d++)
     {
      float vg = 0;
      float qg = 0;
      float kg = 0;
      for(int l = 0; l < units; l++)
        {
         float sg = scores[shift_s + l];
         if(sg <= 0)
            continue;
         uint shift_q = dimension * (h + 3 * l * heads) + d;
         uint shift_k = dimension * (h + (3 * l + 1) * heads) + d;
         uint shift_g = dimension * (h + heads * l) + d;
         //---
         vg += gradient[shift_g] * sg;
         sg = scores_g[shift_s + l];
         kg += sg * qkv[shift_q];
         qg += sg * qkv[shift_k];
        }
      qkv_g[shift_qg + d] = qg;
      qkv_g[shift_kg + d] = kg;
      qkv_g[shift_vg + d] = vg;
     }
  }
//+------------------------------------------------------------------+
///\ingroup neuron_dropout Kernel for Dropout.
/// Describes the dropout method.
///\details Detailed description on <A HREF="https://www.mql5.com/ru/articles/9112#para32">the link.</A>
//+------------------------------------------------------------------+
__kernel void Dropout(__global float *inputs,     ///<[in] Input matrix
                      __global float *map,      ///<[in] Dropout map matrix
                      __global float *out,      ///<[out] Output matrix
                      int dimension              ///< Dimension of matrix
                     )
  {
   const int i = get_global_id(0) * 4;
   if(i + 3 < dimension)
     {
      float4 k = (float4)(inputs[i], inputs[i + 1], inputs[i + 2], inputs[i + 3]) * (float4)(map[i], map[i + 1], map[i + 2], map[i + 3]);
      out[i] = k.s0;
      out[i + 1] = k.s1;
      out[i + 2] = k.s2;
      out[i + 3] = k.s3;
     }
   else
      for(int k = i; k < min(dimension, i + 4); k++)
         out[i + k] = (inputs[i + k] * map[i + k]);
  }
//+------------------------------------------------------------------+
///\ingroup neuron_norm Kernels of Batch normalization process
/// Describes the process of Batch normalization. (#CNeuronBatchNormOCL)
///\details Detailed description on <A HREF="https://www.mql5.com/ru/articles/9207#para42">the link.</A>
/// @param[in] inputs Input data tenzor
/// @param[in,out] options Tenzor of variables
/// @param[out] output Tenzor of output data
/// @param[in] batch Batch size
/// @param[in] optimization Optimization type
/// @param[in] activation Activation type
//+------------------------------------------------------------------+
__kernel void BatchFeedForward(__global float *inputs,
                               __global float *options,
                               __global float *output,
                               int batch,
                               int optimization,
                               int activation)
  {
   if(batch <= 1)
      return;
   int n = get_global_id(0);
   int shift = n * (optimization == 0 ? 7 : 9);
//---
   for(int i = 0; i < (optimization == 0 ? 7 : 9); i++)
      if(isnan(options[shift + i]))
         options[shift + i] = 0;
//---
   float mean = (options[shift] * ((float)batch - 1.0f) + inputs[n]) / ((float)batch);
   float delt = inputs[n] - mean;
   float variance = options[shift + 1] * ((float)batch - 1.0f) + pow(delt, 2);
//if(options[shift + 1] > 0)
   variance /= (float)batch;
   float nx = (variance > 0 ? delt / sqrt(variance) : 0);
//---
   if(options[shift + 3] == 0)
      options[shift + 3] = 1;
//---
   float res = options[shift + 3] * nx + options[shift + 4];
   switch(activation)
     {
      case 0:
         res = tanh(clamp(res, -20.0f, 20.0f));
         break;
      case 1:
         res = 1 / (1 + exp(-clamp(res, -20.0f, 20.0f)));
         break;
      case 2:
         if(res < 0)
            res *= 0.01f;
         break;
      default:
         break;
     }
//---
   options[shift] = mean;
   options[shift + 1] = variance;
   options[shift + 2] = nx;
   output[n] = res;
  }
//+------------------------------------------------------------------+
///\ingroup neuron_gr
/// Kernel of the Batch neuron to transfer gradient to previous layer (#CNeuronBatchNormOCL)
///\details Detailed description on <A HREF="https://www.mql5.com/ru/articles/9207#para43">the link.</A>
//+------------------------------------------------------------------+
__kernel void CalcHiddenGradientBatch(__global float *options,     ///<[in] Options matrix m*(7 or 9), where m - Number of neurons in previous layer
                                      __global float *matrix_g,   ///<[in] Tensor of gradients at current layer
                                      __global float *matrix_i,   ///<[in] Tensor of previous layer output
                                      __global float *matrix_ig,  ///<[out] Tensor of gradients at previous layer
                                      int activation,             ///< Activation type (#ENUM_ACTIVATION)
                                      int batch,                   ///< Batch size
                                      int optimization            ///< Optimization type
                                     )
  {
   if(batch <= 1)
      return;
//---
   int n = get_global_id(0);
   int shift = n * (optimization == 0 ? 7 : 9);
//---
   float inp = matrix_i[n];
   float gnx = matrix_g[n] * options[shift + 3];
   float temp = (options[shift + 1] > 0 ? 1.0f / sqrt(options[shift + 1]) : 0);
   float gmu = (-temp) * gnx;
   float gvar = (options[shift + 1] > 0 ? (options[shift] * inp) / (2 * pow(options[shift + 1], 3 / 2)) * gnx : 0);
   float gx = temp * gnx + gmu / batch + gvar * 2 * inp / batch * pow((float)(batch - 1) / batch, 2.0f);
//---
   if(isnan(gx))
      gx = 0;
   switch(activation)
     {
      case 0:
         gx = clamp(gx + inp, -1.0f, 1.0f) - inp;
         gx = gx * (1 - pow(inp == 1 || inp == -1 ? 0.99999999f : inp, 2));
         break;
      case 1:
         gx = clamp(gx + inp, 0.0f, 1.0f) - inp;
         gx = gx * (inp == 0 || inp == 1 ? 0.00000001f : (inp * (1 - inp)));
         break;
      case 2:
         if(inp < 0)
            gx *= 0.01f;
         break;
      default:
         break;
     }
   matrix_ig[n] = gx;
  }
//+------------------------------------------------------------------+
///\ingroup neuron_opt Batch normalization Neuron SGD optimization Updating options kernel
/// Describes the process of SGD optimization options for the Batch normalization Neuron (#CNeuronBatchNormOCL).
///\details Detailed description on <A HREF="https://www.mql5.com/ru/articles/9207#para43">the link.</A>
//+------------------------------------------------------------------+
__kernel void UpdateBatchOptionsMomentum(__global float *options,     ///<[in,out] Options matrix m*7, where m - Number of neurons in previous layer
      __global float *matrix_g,   ///<[in] Tensor of gradients at current layer
      float learning_rates,       ///< Learning rates
      float momentum              ///< Momentum multiplier
                                        )
  {
   const int n = get_global_id(0);
   int inputs = get_global_size(0);
   const int shift = n * 7;
   float grad = matrix_g[n];
//---
   float2 delta = learning_rates * grad * (float2)(options[shift + 2], 1) + momentum * (float2)(options[shift + 5], options[shift + 6]);
   if(!isnan(delta.s0) && !isnan(delta.s1))
     {
      options[shift + 5] = delta.s0;
      if(delta.s0 * grad > 0)
         options[shift + 3] = clamp(options[shift + 3] + delta.s0, -MAX_WEIGHT, MAX_WEIGHT);
      if(delta.s1 * grad > 0)
         options[shift + 6] = delta.s1;
      options[shift + 4] += delta.s1 - learning_rates * (l1 * sign(options[shift + 4]) + l2 * options[shift + 4] / inputs);
     }
  }
//+------------------------------------------------------------------+
///\ingroup neuron_opt Batch normalization Neuron Adam optimization Updating options kernel
/// Describes the process of Adam optimization options for the Batch normalization  Neuron (#CNeuronBatchNormOCL).
///\details Detailed description on <A HREF="https://www.mql5.com/ru/articles/9207#para43">the link.</A>
//+------------------------------------------------------------------+
__kernel void UpdateBatchOptionsAdam(__global float *options,     ///<[in,out] Options matrix m*9, where m - Number of neurons in previous layer
                                     __global float *matrix_g,   ///<[in] Tensor of gradients at current layer
                                     const float l,                   ///< Learning rates
                                     const float b1,                  ///< First momentum multiplier
                                     const float b2                  ///< Second momentum multiplier
                                    )
  {
   const int n = get_global_id(0);
   int inputs = get_global_size(0);
   const int shift = n * 9;
   float grad = matrix_g[n];
//---
   float2 mt = b1 * (float2)(options[shift + 5], options[shift + 6]) + (1 - b1) * (float2)(grad * options[shift + 2] - (l1 * sign(options[shift + 3]) + l2 * options[shift + 3]), grad - (l1 * sign(options[shift + 4]) + l2 * options[shift + 4]));
   float2 vt = b2 * (float2)(options[shift + 5], options[shift + 6]) + (1 - b2) * pow((float2)(grad * options[shift + 2], grad), 2);
   float2 delta = l * mt / sqrt(vt + 1.0e-37f);
   if(isnan(delta.s0) || isnan(delta.s1))
      return;
   float2 weight = l * (l1 * sign((float2)(options[shift + 3], options[shift + 4])) + l2 * (float2)(options[shift + 3], options[shift + 4]) / inputs) + delta;
//---
   if(!isnan(weight.s0) && !isnan(weight.s1))
     {
      if(delta.s0 * grad > 0)
         options[shift + 3] = weight.s0;
      if(delta.s1 * grad > 0)
         options[shift + 4] = weight.s1;
      options[shift + 5] = mt.s0;
      options[shift + 6] = mt.s1;
      options[shift + 7] = vt.s0;
      options[shift + 8] = vt.s1;
     }
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
__kernel void VAE_FeedForward(__global float* inputs,
                              __global float* random,
                              __global float* outputs
                             )
  {
   uint i = (uint)get_global_id(0);
   uint total = (uint)get_global_size(0);
   outputs[i] = inputs[i] + exp(0.5f * inputs[i + total]) * random[i];
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
__kernel void VAE_CalcHiddenGradient(__global float* inputs,
                                     __global float* inp_grad,
                                     __global float* random,
                                     __global float* gradient,
                                     const float kld_mult
                                    )
  {
   uint i = (uint)get_global_id(0);
   uint total = (uint)get_global_size(0);
   float log_var = inputs[i + total];
   float kld = kld_mult * 0.5f * (log_var - exp(log_var) - pow(inputs[i], 2.0f) + 1);
   float grad = gradient[i];
   inp_grad[i] = grad / exp(0.5f * log_var) + kld * inputs[i];
   inp_grad[i + total] = 0.5f * (grad * random[i] - kld * (1 - exp(log_var))) ;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
__kernel void LSTM_FeedForward(__global float* inputs, int inputs_size,
                               __global float* weights,
                               __global float* concatenated,
                               __global float* memory,
                               __global float* output
                              )
  {
   uint id = (uint)get_global_id(0);
   uint total = (uint)get_global_size(0);
   uint id2 = (uint) get_local_id(1);
//---
   float sum = 0;
   uint shift = (id + id2 * total) * (total + inputs_size + 1);
   for(uint i = 0; i < total; i += 4)
     {
      if(total - i > 4)
         sum += dot((float4)(output[i], output[i + 1], output[i + 2], output[i + 3]),
                    (float4)(weights[shift + i], weights[shift + i + 1], weights[shift + i + 2], weights[shift + i + 3]));
      else
         for(uint k = i; k < total; k++)
            sum += output[k] * weights[shift + k];
     }
//---
   shift += total;
   for(uint i = 0; i < inputs_size; i += 4)
     {
      if(total - i > 4)
         sum += dot((float4)(inputs[i], inputs[i + 1], inputs[i + 2], inputs[i + 3]),
                    (float4)(weights[shift + i], weights[shift + i + 1], weights[shift + i + 2], weights[shift + i + 3]));
      else
         for(uint k = i; k < total; k++)
            sum += inputs[k] * weights[shift + k];
     }
   sum += weights[shift + inputs_size];
   if(id2 < 3)
      concatenated[id2 * total + id] = 1.0f / (1.0f + exp(sum));
   else
      concatenated[id2 * total + id] = tanh(sum);
//---
   barrier(CLK_LOCAL_MEM_FENCE);
   if(id2 == 0)
     {
      float mem = memory[id + total] = memory[id];
      float fg = concatenated[id];
      float ig = concatenated[id + total];
      float og = concatenated[id + 2 * total];
      float nc = concatenated[id + 3 * total];
      //---
      memory[id] = mem = mem * fg + ig * nc;
      output[id] = og * tanh(mem);
     }
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
__kernel void LSTM_ConcatenatedGradient(__global float* gradient,
                                        __global float* concatenated_gradient,
                                        __global float* memory,
                                        __global float* concatenated
                                       )
  {
   uint id = get_global_id(0);
   uint total = get_global_size(0);
   float t = tanh(memory[id]);
   concatenated_gradient[id + 2 * total] = gradient[id] * t;             //output gate
   float memory_gradient = gradient[id] * concatenated[id + 2 * total];
   memory_gradient *= 1 - pow(t, 2.0f);
   concatenated_gradient[id + 3 * total] = memory_gradient * concatenated[id + total];         //new content
   concatenated_gradient[id + total] = memory_gradient * concatenated[id + 3 * total]; //input gate
   concatenated_gradient[id] = memory_gradient * memory[id + total];     //forgat gate
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
__kernel void LSTM_HiddenGradient(__global float* concatenated_gradient,
                                  __global float* inputs_gradient,
                                  __global float* weights_gradient,
                                  __global float* hidden_state,
                                  __global float* inputs,
                                  __global float* weights,
                                  __global float* output,
                                  const int hidden_size,
                                  const int inputs_size
                                 )
  {
   uint id = get_global_id(0);
   uint total = get_global_size(0);
   uint weights_step = hidden_size + inputs_size + 1;
   for(int i = id; i < (hidden_size + inputs_size); i += total)
     {
      float inp = 0;
      if(i < hidden_size)
        {
         inp = hidden_state[i];
         hidden_state[i] = output[i];
        }
      else
        {
         inp = inputs[i - hidden_size];
         float grad = 0;
         for(uint g = 0; g < 3 * hidden_size; g++)
           {
            float temp = concatenated_gradient[g];
            grad += temp * (1 - temp) * weights[i + g * weights_step];
           }
         for(uint g = 3 * hidden_size; g < 4 * hidden_size; g++)
           {
            float temp = concatenated_gradient[g];
            grad += temp * (1 - pow(temp, 2.0f)) * weights[i + g * weights_step];
           }
         inputs_gradient[i - hidden_size] = grad;
        }
      //---
      for(uint g = 0; g < 3 * hidden_size; g++)
        {
         float temp = concatenated_gradient[g];
         weights[i + g * weights_step] = temp * (1 - temp) * inp;
        }
      for(uint g = 3 * hidden_size; g < 4 * hidden_size; g++)
        {
         float temp = concatenated_gradient[g];
         weights[i + g * weights_step] = temp * (1 - pow(temp, 2.0f)) * inp;
        }
     }
//---
   for(int i = id; i < 4 * hidden_size; i += total)
     {
      float temp = concatenated_gradient[(i + 1) * hidden_size];
      if(i < 3 * hidden_size)
         weights[(i + 1) * weights_step] = temp * (1 - temp);
      else
         weights[(i + 1) * weights_step] = 1 - pow(temp, 2.0f);
     }
  }
//+------------------------------------------------------------------+
///\ingroup LSTM_opt  LSTM Adam Updating Weights Calculation kernel
/// Describes the process of Adam optimization weights for the Neuron LSTM (#CNeuronLSTMOCL).
//+------------------------------------------------------------------+
__kernel void LSTM_UpdateWeightsAdam(__global float *weights,        ///<[in,out] Weights matrix (m+1)*n, where m - number of neurons in previous layer and n - number of neurons in current layer
                                     __global float *weights_gradient,  ///<[in] Tensor of gradients at current layer
                                     __global float *matrix_m,        ///<[in,out] Matrix of first momentum
                                     __global float *matrix_v,        ///<[in,out] Matrix of seconfd momentum
                                     const float l,                   ///< Learning rates
                                     const float b1,                  ///< First momentum multiplier
                                     const float b2                   ///< Second momentum multiplier
                                    )
  {
   const uint id = get_global_id(0);
   const uint total = get_global_size(0);
   const uint id1 = get_global_id(1);
   const uint wi = id1 * total + id;
   float g = weights_gradient[wi];
   float mt = b1 * matrix_m[wi] + (1 - b1) * g;
   float vt = b2 * matrix_v[wi] + (1 - b2) * pow(g, 2);
   float delta = l * (mt / (sqrt(vt) + 1.0e-37f) - (l1 * sign(weights[wi]) + l2 * weights[wi] / total));
   weights[wi] = clamp(weights[wi] + delta, -MAX_WEIGHT, MAX_WEIGHT);
   matrix_m[wi] = mt;
   matrix_v[wi] = vt;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
__kernel void SoftMax_FeedForward(__global float *inputs,
                                  __global float *outputs,
                                  const int total)
  {
   const uint i = (uint)get_global_id(0);
   const uint l = (uint)get_local_id(0);
   const uint h = (uint)get_global_id(1);
   const uint ls = min((uint)get_local_size(0), (uint)LOCAL_ARRAY_SIZE);
   uint shift_head = h * total;
//---
   __local float temp[LOCAL_ARRAY_SIZE];
   uint count = 0;
   if(l < ls)
      do
        {
         uint shift = shift_head + count * ls + l;
         if(shift < ((h + 1) * total))
            temp[l] = (count > 0 ? fmax(fabs(inputs[shift]), temp[l]) : fabs(inputs[shift]));
         count++;
        }
      while((count * ls + l) < total);
   barrier(CLK_LOCAL_MEM_FENCE);
   float max_value = temp[0];
   for(int i = 1; i < ls; i++)
      max_value = fmax(max_value, temp[i]);
//---
   count = 0;
   if(l < ls)
      do
        {
         uint shift = shift_head + count * ls + l;
         temp[l] = (count > 0 ? temp[l] : 0) + (shift < ((h + 1) * total) ? exp(inputs[shift] / max_value) : 0);
         count++;
        }
      while((count * ls + l) < total);
   barrier(CLK_LOCAL_MEM_FENCE);
   count = min(ls, (uint)total);
   do
     {
      count = (count + 1) / 2;
      if(l < ls)
         temp[l] += (l < count && (l + count) < total ? temp[l + count] : 0);
      if(l + count < ls)
         temp[l + count] = 0;
      barrier(CLK_LOCAL_MEM_FENCE);
     }
   while(count > 1);
//---
   float sum = temp[0];
   if(sum != 0)
     {
      count = 0;
      while((count * ls + l) < total)
        {
         uint shift = shift_head + count * ls + l;
         if(shift < ((h + 1)*total))
            outputs[shift] = exp(inputs[shift] / max_value) / (sum + 1e-37f);
         count++;
        }
     }
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
__kernel void SoftMax_HiddenGradient(__global float* outputs,
                                     __global float* output_gr,
                                     __global float* input_gr)
  {
   size_t i = get_global_id(0);
   size_t outputs_total = get_global_size(0);
   size_t h = get_global_id(1);
   uint shift = h * outputs_total;
   float output = outputs[shift + i];
   float result = 0;
   for(int j = 0; j < outputs_total ; j++)
      result += outputs[shift + j] * output_gr[shift + j] * ((float)(i == j) - output);
   input_gr[shift + i] = result;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
__kernel void SoftMax_OutputGradient(__global float* outputs,
                                     __global float* targets,
                                     __global float* output_gr)
  {
   size_t i = get_global_id(0);
   output_gr[i] = (outputs[i] == 0 ? 0 : targets[i] / outputs[i]);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
__kernel void FQF_Cosine(__global float* softmax,
                         __global float* output)
  {
   size_t i = get_global_id(0);
   size_t total = get_global_size(0);
   size_t action = get_global_id(1);
   int shift = action * total;
//---
   float result = 0;
   for(int it = 0; it < i; it++)
      result += softmax[shift + it];
   result += softmax[shift + i] / 2.0f;
   output[shift + i] = cos(i * M_PI_F * result);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
__kernel void FQF_Output(__global float* quantiles,
                         __global float* delta_taus,
                         __global float* output,
                         int total)
  {
   size_t action = get_global_id(0);
   int shift = action * total;
//---
   float result = 0;
   for(int i = 0; i < total; i++)
      result += quantiles[shift + i] * delta_taus[shift + i];
   output[action] = result;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
__kernel void FQF_OutputGradient(__global float* quantiles,
                                 __global float* delta_taus,
                                 __global float* output_gr,
                                 __global float* quantiles_gr,
                                 __global float* taus_gr
                                )
  {
   size_t i = get_global_id(0);
   size_t total = get_global_size(0);
   size_t action = get_global_id(1);
   int shift = action * total;
//---
   float gradient = output_gr[action];
   quantiles_gr[shift + i] = gradient * delta_taus[shift + i];
   taus_gr[shift + i] = gradient * quantiles[shift + i];
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
__kernel void FQF_QuantileGradient(__global float* state_embeding,
                                   __global float* taus_embeding,
                                   __global float* quantiles_gr,
                                   __global float* state_gr,
                                   __global float* taus_gr
                                  )
  {
   size_t i = get_global_id(0);
   size_t total = get_global_size(0);
   size_t action = get_global_id(1);
   int shift = action * total;
//---
   float gradient = quantiles_gr[shift + i];
   state_gr[shift + i] = gradient * taus_embeding[shift + i];
   taus_gr[shift + i] = gradient * state_embeding[shift + i];
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
__kernel void FQF_CosineGradient(__global float* softmax,
                                 __global float* output_gr,
                                 __global float* softmax_gr
                                )
  {
   size_t i = get_global_id(0);
   size_t total = get_global_size(0);
   size_t action = get_global_id(1);
   int shift = action * total;
//---
   float cumul = 0;
   for(int it = 0; it < i; it++)
      cumul += softmax[shift + it];
   float result = -M_PI_F * i * sin(M_PI_F * i * (cumul + softmax[shift + i] / 2)) * output_gr[shift + i];
   for(int it = i + 1; it < total; it++)
     {
      cumul += softmax[shift + it - 1];
      float temp = cumul + softmax[shift + it] / 2;
      result += -M_PI_F * it * sin(M_PI_F * it * temp) * output_gr[shift + it] * softmax[shift + it] / temp;
     }
   softmax_gr[shift + i] += result;
  }
//+------------------------------------------------------------------+
//| Sparse Attention                                                 |
//+------------------------------------------------------------------+
__kernel void MHSparseAttentionScore(__global float *qkv,    ///<[in] Matrix of Querys, Keys, Values
                                     __global float *score,  ///<[out] Matrix of Scores
                                     int dimension,          ///< Dimension of Key
                                     float sparse            ///< less than 1.0 coefficient of sparse
                                    )
  {
   int q = get_global_id(0);
   int h = get_global_id(1);
   int units = get_global_size(0);
   int heads = get_global_size(1);
//---
   int shift_q = dimension * (h + 3 * q * heads);
   int shift_s = units * (h + q * heads);
   int active_units = (int)max((float)(units * sparse), min((float)units, 3.0f));
//---
   float koef = sqrt((float)dimension);
   if(koef < 1)
      koef = 1;
   float sum = 0.0f;
   float min_s = 0.0f;
   float max_s = 0.0f;
   for(int k = 0; k < units; k++)
     {
      float result = 0;
      int shift_k = dimension * (h + heads * (3 * k + 1));
      for(int i = 0; i < dimension; i++)
        {
         if((dimension - i) > 4)
           {
            result += dot((float4)(qkv[shift_q + i], qkv[shift_q + i + 1], qkv[shift_q + i + 2], qkv[shift_q + i + 3]),
                          (float4)(qkv[shift_k + i], qkv[shift_k + i + 1], qkv[shift_k + i + 2], qkv[shift_k + i + 3]));
            i += 3;
           }
         else
            result += (qkv[shift_q + i] * qkv[shift_k + i]);
        }
      score[shift_s + k] = result;
      if(k == 0)
         min_s = max_s = result;
      else
        {
         max_s = max(max_s, result);
         min_s = min(min_s, result);
        }
     }
//---
   int count = units;
   while(count > active_units && min_s < max_s)
     {
      count = 0;
      float temp = max_s;
      for(int k = 0; k < units; k++)
        {
         float value = score[shift_s + k];
         if(value < min_s)
            continue;
         count++;
         if(value < temp && value > min_s)
            temp = value;
        }
      if(count > active_units)
         min_s = temp;
     }
//---
   if(max_s == 0.0f)
      max_s = 1.0f;
   for(int k = 0; k < units; k++)
     {
      float value = score[shift_s + k];
      if(value < min_s)
        {
         score[shift_s + k] = 0.0f;
         continue;
        }
      value = exp(value / max_s / koef);
      score[shift_s + k] = value;
      sum += value;
     }
//---
   for(int k = 0; (k < units && sum > 1); k++)
     {
      float temp = score[shift_s + k];
      if(temp == 0.0f)
         continue;
      score[shift_s + k] = temp / sum;
     }
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
__kernel void MHSparseAttentionOut(__global float *scores, ///<[in] Matrix of Scores
                                   __global float *qkv,    ///<[in] Matrix of Values
                                   __global float *out,    ///<[out] Output tesor
                                   int dimension           ///< Dimension of Value
                                  )
  {
   int u = get_global_id(0);
   int units = get_global_size(0);
   int h = get_global_id(1);
   int heads = get_global_size(1);
//---
   int shift_s = units * (h + heads * u);
   int shift_out = dimension * (h + heads * u);
//---
   for(int d = 0; d < dimension; d++)
     {
      float result = 0;
      for(int v = 0; v < units; v ++)
        {
         float cur_score = scores[shift_s + v];
         if(cur_score == 0)
            continue;
         int shift_v = dimension * (h + heads * (3 * v + 2)) + d;
         result += cur_score * qkv[shift_v];
        }
      out[shift_out + d] = result;
     }
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
__kernel void FeedForwardMultiModels(__global float *matrix_w,///<[in] Weights matrix (m+1)*n, where m - number of neurons in layer and n - number of outputs (neurons in next layer)
                                     __global float *matrix_i,///<[in] Inputs tesor
                                     __global float *matrix_o,///<[out] Output tensor
                                     int inputs,///< Number of inputs
                                     int activation///< Activation type (#ENUM_ACTIVATION)
                                    )
  {
   int i = get_global_id(0);
   int outputs = get_global_size(0);
   int m = get_global_id(1);
   int models = get_global_size(1);
//---
   float sum = 0;
   float4 inp, weight;
   int shift = (inputs + 1) * (i + outputs * m);
   int shift_in = inputs * m;
   int shift_out = outputs * m;
   for(int k = 0; k <= inputs; k = k + 4)
     {
      switch(inputs - k)
        {
         case 0:
            inp = (float4)(1, 0, 0, 0);
            weight = (float4)(matrix_w[shift + k], 0, 0, 0);
            break;
         case 1:
            inp = (float4)(matrix_i[shift_in + k], 1, 0, 0);
            weight = (float4)(matrix_w[shift + k], matrix_w[shift + k + 1], 0, 0);
            break;
         case 2:
            inp = (float4)(matrix_i[shift_in + k], matrix_i[shift_in + k + 1], 1, 0);
            weight = (float4)(matrix_w[shift + k], matrix_w[shift + k + 1], matrix_w[shift + k + 2], 0);
            break;
         case 3:
            inp = (float4)(matrix_i[shift_in + k], matrix_i[shift_in + k + 1], matrix_i[shift_in + k + 2], 1);
            weight = (float4)(matrix_w[shift + k], matrix_w[shift + k + 1], matrix_w[shift + k + 2], matrix_w[shift + k + 3]);
            break;
         default:
            inp = (float4)(matrix_i[shift_in + k], matrix_i[shift_in + k + 1], matrix_i[shift_in + k + 2], matrix_i[shift_in + k + 3]);
            weight = (float4)(matrix_w[shift + k], matrix_w[shift + k + 1], matrix_w[shift + k + 2], matrix_w[shift + k + 3]);
            break;
        }
      float d = dot(inp, weight);
      if(isnan(sum + d))
         continue;
      sum += d;
     }
   if(isnan(sum))
      sum = 0;
   switch(activation)
     {
      case 0:
         sum = tanh(sum);
         break;
      case 1:
         sum = 1 / (1 + exp(-sum));
         break;
      case 2:
         if(sum < 0)
            sum *= 0.01f;
         break;
      default:
         break;
     }
   matrix_o[shift_out + i] = sum;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
__kernel void CalcHiddenGradientMultiModels(__global float *matrix_w,///<[in] Weights matrix (m+1)*n, where m - number of neurons in previous layer and n - number of neurons in current layer
      __global float *matrix_g,///<[in] Tensor of gradients at current layer
      __global float *matrix_o,///<[in] Previous layer Output tensor
      __global float *matrix_ig,///<[out] Tensor of gradients at previous layer
      int outputs,///< Number of outputs
      int activation,///< Activation type (#ENUM_ACTIVATION),
      int model
                                           )
  {
   int i = get_global_id(0);
   int inputs = get_global_size(0);
   int m = get_global_id(1);
   int models = get_global_size(1);
//---
   int shift_in = inputs * m;
   if(model >= 0 && model != m)
     {
      matrix_ig[shift_in + i] = 0;
      return;
     }
//---
   int shift_out = outputs * m;
   int shift_w = (inputs + 1) * outputs * m;
   float sum = 0;
   float out = matrix_o[shift_in + i];
   float4 grad, weight;
   for(int k = 0; k < outputs; k += 4)
     {
      switch(outputs - k)
        {
         case 1:
            weight = (float4)(matrix_w[shift_w + k * (inputs + 1) + i], 0, 0, 0);
            grad = (float4)(matrix_g[shift_out + k], 0, 0, 0);
            break;
         case 2:
            grad = (float4)(matrix_g[shift_out + k], matrix_g[shift_out + k + 1], 0, 0);
            weight = (float4)(matrix_w[shift_w + k * (inputs + 1) + i], matrix_w[shift_w + (k + 1) * (inputs + 1) + i], 0, 0);
            break;
         case 3:
            grad = (float4)(matrix_g[shift_out + k], matrix_g[shift_out + k + 1], matrix_g[shift_out + k + 2], 0);
            weight = (float4)(matrix_w[shift_w + k * (inputs + 1) + i], matrix_w[shift_w + (k + 1) * (inputs + 1) + i], matrix_w[shift_w + (k + 2) * (inputs + 1) + i], 0);
            break;
         default:
            grad = (float4)(matrix_g[shift_out + k], matrix_g[shift_out + k + 1], matrix_g[shift_out + k + 2], matrix_g[shift_out + k + 3]);
            weight = (float4)(matrix_w[shift_w + k * (inputs + 1) + i], matrix_w[shift_w + (k + 1) * (inputs + 1) + i], matrix_w[shift_w + (k + 2) * (inputs + 1) + i], matrix_w[shift_w + (k + 3) * (inputs + 1) + i]);
            break;
        }
      sum += dot(grad, weight);
     }
   if(isnan(sum))
      sum = 0;
   switch(activation)
     {
      case 0:
         out = clamp(out, -1.0f, 1.0f);
         sum = clamp(sum + out, -1.0f, 1.0f) - out;
         sum = sum * max(1 - pow(out, 2), 1.0e-4f);
         break;
      case 1:
         out = clamp(out, 0.0f, 1.0f);
         sum = clamp(sum + out, 0.0f, 1.0f) - out;
         sum = sum * max(out * (1 - out), 1.0e-4f);
         break;
      case 2:
         if(out < 0)
            sum *= 0.01f;
         break;
      default:
         break;
     }
   matrix_ig[shift_in + i] = sum;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
__kernel void UpdateWeightsAdamMultiModels(__global float *matrix_w,        ///<[in,out] Weights matrix (m+1)*n, where m - number of neurons in previous layer and n - number of neurons in current layer
      __global const float *matrix_g,  ///<[in] Tensor of gradients at current layer
      __global const float *matrix_i,  ///<[in] Inputs tesor
      __global float *matrix_m,        ///<[in,out] Matrix of first momentum
      __global float *matrix_v,        ///<[in,out] Matrix of seconfd momentum
      const int inputs,                 ///< Number of inputs
      const float l,                   ///< Learning rates
      const float b1,                  ///< First momentum multiplier
      const float b2,                   ///< Second momentum multiplier
      const int model
                                          )
  {
   const int outputs = get_global_size(0);
   const int i = get_global_id(0);
   const int j = get_global_id(1);
   const int wi = (i + outputs * model) * (inputs + 1) + j * 4;
   float4 m, v, weight, inp;
   int shift_in = j * 4 + inputs * model;
   if((inputs + 1 - j * 4) < 0)
      return;
   switch(inputs + 1 - j * 4)
     {
      case 0:
         inp = (float4)(1, 0, 0, 0);
         weight = (float4)(matrix_w[wi], 0, 0, 0);
         m = (float4)(matrix_m[wi], 0, 0, 0);
         v = (float4)(matrix_v[wi], 0, 0, 0);
         break;
      case 1:
         inp = (float4)(matrix_i[shift_in], 1, 0, 0);
         weight = (float4)(matrix_w[wi], matrix_w[wi + 1], 0, 0);
         m = (float4)(matrix_m[wi], matrix_m[wi + 1], 0, 0);
         v = (float4)(matrix_v[wi], matrix_v[wi + 1], 0, 0);
         break;
      case 2:
         inp = (float4)(matrix_i[shift_in], matrix_i[shift_in + 1], 1, 0);
         weight = (float4)(matrix_w[wi], matrix_w[wi + 1], matrix_w[wi + 2], 0);
         m = (float4)(matrix_m[wi], matrix_m[wi + 1], matrix_m[wi + 2], 0);
         v = (float4)(matrix_v[wi], matrix_v[wi + 1], matrix_v[wi + 2], 0);
         break;
      case 3:
         inp = (float4)(matrix_i[shift_in], matrix_i[shift_in + 1], matrix_i[shift_in + 2], 1);
         weight = (float4)(matrix_w[wi], matrix_w[wi + 1], matrix_w[wi + 2], matrix_w[wi + 3]);
         m = (float4)(matrix_m[wi], matrix_m[wi + 1], matrix_m[wi + 2], matrix_m[wi + 3]);
         v = (float4)(matrix_v[wi], matrix_v[wi + 1], matrix_v[wi + 2], matrix_v[wi + 3]);
         break;
      default:
         inp = (float4)(matrix_i[shift_in], matrix_i[shift_in + 1], matrix_i[shift_in + 2], matrix_i[shift_in + 3]);
         weight = (float4)(matrix_w[wi], matrix_w[wi + 1], matrix_w[wi + 2], matrix_w[wi + 3]);
         m = (float4)(matrix_m[wi], matrix_m[wi + 1], matrix_m[wi + 2], matrix_m[wi + 3]);
         v = (float4)(matrix_v[wi], matrix_v[wi + 1], matrix_v[wi + 2], matrix_v[wi + 3]);
         break;
     }
   float4 g = (float4)(matrix_g[(outputs + 1) * model + i]) * inp;
   float4 mt = b1 * m + (1 - b1) * g;
   float4 vt = b2 * v + (1 - b2) * pow(g, 2);
   float4 delta = l * (mt / (sqrt(vt) + 1.0e-37f) - (l1 * sign(weight) + l2 * weight));
   switch(min(inputs + 1 - j * 4, 3))
     {
      case 3:
         if(delta.s3 * g.s3 > 0)
            matrix_w[wi + 3] = clamp(matrix_w[wi + 3] + delta.s3, -MAX_WEIGHT, MAX_WEIGHT);
         matrix_m[wi + 3] = mt.s3;
         matrix_v[wi + 3] = vt.s3;
      case 2:
         if(delta.s2 * g.s2 > 0)
            matrix_w[wi + 2] = clamp(matrix_w[wi + 2] + delta.s2, -MAX_WEIGHT, MAX_WEIGHT);
         matrix_m[wi + 2] = mt.s2;
         matrix_v[wi + 2] = vt.s2;
      case 1:
         if(delta.s1 * g.s1 > 0)
            matrix_w[wi + 1] = clamp(matrix_w[wi + 1] + delta.s1, -MAX_WEIGHT, MAX_WEIGHT);
         matrix_m[wi + 1] = mt.s1;
         matrix_v[wi + 1] = vt.s1;
      case 0:
         if(delta.s0 * g.s0 > 0)
            matrix_w[wi] = clamp(matrix_w[wi] + delta.s0, -MAX_WEIGHT, MAX_WEIGHT);
         matrix_m[wi] = mt.s0;
         matrix_v[wi] = vt.s0;
         break;
     }
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
__kernel void Concat_FeedForward(__global float *matrix_w,///<[in] Weights matrix (m+1)*n, where m - number of neurons in layer and n - number of outputs (neurons in next layer)
                                 __global float *matrix_i1,///<[in] Inputs 1 tesor
                                 __global float *matrix_i2,///<[in] Inputs 2 tesor
                                 __global float *matrix_o,///<[out] Output tensor
                                 int inputs1,///< Number of inputs
                                 int inputs2,///< Number of inputs
                                 int activation///< Activation type (#ENUM_ACTIVATION)
                                )
  {
   int i = get_global_id(0);
   float sum = 0;
   float4 inp, weight;
   int shift = (inputs1 + inputs2 + 1) * i;
//---
   for(int k = 0; k < inputs1; k += 4)
     {
      switch(inputs1 - k)
        {
         case 1:
            inp = (float4)(matrix_i1[k], 0, 0, 0);
            weight = (float4)(matrix_w[shift + k], 0, 0, 0);
            break;
         case 2:
            inp = (float4)(matrix_i1[k], matrix_i1[k + 1], 0, 0);
            weight = (float4)(matrix_w[shift + k], matrix_w[shift + k + 1], 0, 0);
            break;
         case 3:
            inp = (float4)(matrix_i1[k], matrix_i1[k + 1], matrix_i1[k + 2], 0);
            weight = (float4)(matrix_w[shift + k], matrix_w[shift + k + 1], matrix_w[shift + k + 2], 0);
            break;
         default:
            inp = (float4)(matrix_i1[k], matrix_i1[k + 1], matrix_i1[k + 2], matrix_i1[k + 3]);
            weight = (float4)(matrix_w[shift + k], matrix_w[shift + k + 1], matrix_w[shift + k + 2], matrix_w[shift + k + 3]);
            break;
        }
      float d = dot(inp, weight);
      if(isnan(sum + d))
         continue;
      sum += d;
     }
//---
   shift += inputs1;
   for(int k = 0; k < inputs2; k += 4)
     {
      switch(inputs2 - k)
        {
         case 1:
            inp = (float4)(matrix_i2[k], 0, 0, 0);
            weight = (float4)(matrix_w[shift + k], 0, 0, 0);
            break;
         case 2:
            inp = (float4)(matrix_i2[k], matrix_i2[k + 1], 0, 0);
            weight = (float4)(matrix_w[shift + k], matrix_w[shift + k + 1], 0, 0);
            break;
         case 3:
            inp = (float4)(matrix_i2[k], matrix_i2[k + 1], matrix_i2[k + 2], 0);
            weight = (float4)(matrix_w[shift + k], matrix_w[shift + k + 1], matrix_w[shift + k + 2], 0);
            break;
         default:
            inp = (float4)(matrix_i2[k], matrix_i2[k + 1], matrix_i2[k + 2], matrix_i2[k + 3]);
            weight = (float4)(matrix_w[shift + k], matrix_w[shift + k + 1], matrix_w[shift + k + 2], matrix_w[shift + k + 3]);
            break;
        }
      float d = dot(inp, weight);
      if(isnan(sum + d))
         continue;
      sum += d;
     }
   sum += matrix_w[shift + inputs2];
//---
   if(isnan(sum))
      sum = 0;
   switch(activation)
     {
      case 0:
         sum = tanh(sum);
         break;
      case 1:
         sum = 1 / (1 + exp(-sum));
         break;
      case 2:
         if(sum < 0)
            sum *= 0.01f;
         break;
      default:
         break;
     }
   matrix_o[i] = sum;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
__kernel void Concat_HiddenGradient(__global float *matrix_w,///<[in] Weights matrix (m+1)*n, where m - number of neurons in previous layer and n - number of neurons in current layer
                                    __global float *matrix_g,///<[in] Tensor of gradients at current layer
                                    __global float *matrix_o1,///<[in] Previous layer Output tensor
                                    __global float *matrix_o2,///<[in] Previous layer Output tensor
                                    __global float *matrix_ig1,///<[out] Tensor of gradients at previous layer
                                    __global float *matrix_ig2,///<[out] Tensor of gradients at previous layer
                                    int outputs,///< Number of outputs
                                    int inputs1,
                                    int inputs2,
                                    int activation1,///< Activation type (#ENUM_ACTIVATION)
                                    int activation2///< Activation type (#ENUM_ACTIVATION)
                                   )
  {
   int i = get_global_id(0);
   if(i >= (inputs1 + inputs2))
      return;
   int inputs = inputs1 + inputs2;
   float sum = 0;
   float out = (i < inputs1 ? matrix_o1[i] : matrix_o2[i - inputs1]);
   float4 grad, weight;
   for(int k = 0; k < outputs; k += 4)
     {
      switch(outputs - k)
        {
         case 1:
            weight = (float4)(matrix_w[k * (inputs + 1) + i], 0, 0, 0);
            grad = (float4)(matrix_g[k], 0, 0, 0);
            break;
         case 2:
            grad = (float4)(matrix_g[k], matrix_g[k + 1], 0, 0);
            weight = (float4)(matrix_w[k * (inputs + 1) + i], matrix_w[(k + 1) * (inputs + 1) + i], 0, 0);
            break;
         case 3:
            grad = (float4)(matrix_g[k], matrix_g[k + 1], matrix_g[k + 2], 0);
            weight = (float4)(matrix_w[k * (inputs + 1) + i], matrix_w[(k + 1) * (inputs + 1) + i], matrix_w[(k + 2) * (inputs + 1) + i], 0);
            break;
         default:
            grad = (float4)(matrix_g[k], matrix_g[k + 1], matrix_g[k + 2], matrix_g[k + 3]);
            weight = (float4)(matrix_w[k * (inputs + 1) + i], matrix_w[(k + 1) * (inputs + 1) + i], matrix_w[(k + 2) * (inputs + 1) + i], matrix_w[(k + 3) * (inputs + 1) + i]);
            break;
        }
      sum += dot(grad, weight);
     }
   if(isnan(sum))
      sum = 0;
   if(i < inputs1)
     {
      switch(activation1)
        {
         case 0:
            out = clamp(out, -1.0f, 1.0f);
            sum = clamp(sum + out, -1.0f, 1.0f) - out;
            sum = sum * max(1 - pow(out, 2), 1.0e-4f);
            break;
         case 1:
            out = clamp(out, 0.0f, 1.0f);
            sum = clamp(sum + out, 0.0f, 1.0f) - out;
            sum = sum * max(out * (1 - out), 1.0e-4f);
            break;
         case 2:
            if(out < 0)
               sum *= 0.01f;
            break;
         default:
            break;
        }
      matrix_ig1[i] = sum;
     }
   else
     {
      switch(activation2)
        {
         case 0:
            out = clamp(out, -1.0f, 1.0f);
            sum = clamp(sum + out, -1.0f, 1.0f) - out;
            sum = sum * max(1 - pow(out, 2), 1.0e-4f);
            break;
         case 1:
            out = clamp(out, 0.0f, 1.0f);
            sum = clamp(sum + out, 0.0f, 1.0f) - out;
            sum = sum * max(out * (1 - out), 1.0e-4f);
            break;
         case 2:
            if(out < 0)
               sum *= 0.01f;
            break;
         default:
            break;
        }
      matrix_ig2[i - inputs1] = sum;
     }
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
__kernel void Concat_UpdateWeightsMomentum(__global float *matrix_w, ///<[in,out] Weights matrix (m+1)*n, where m - number of neurons in previous layer and n - number of neurons in current layer
      __global float *matrix_g,         ///<[in] Tensor of gradients at current layer
      __global float *matrix_i1,         ///<[in] Inputs tesor
      __global float *matrix_i2,         ///<[in] Inputs tesor
      __global float *matrix_dw,        ///<[in,out] Matrix of delta weights in last correction
      int inputs1,                        ///< Number of inputs
      int inputs2,                        ///< Number of inputs
      float learning_rates,             ///< Learning rates
      float momentum                    ///< Momentum multiplier
                                          )
  {
   int i = get_global_id(0);
   int j = get_global_id(1);
   if(j > (inputs1 + inputs2))
      return;
   int wi = i * (inputs1 + inputs2 + 1) + j;
   float inp = (j < inputs1 ? matrix_i1[j] : ((j - inputs1) < inputs2 ? matrix_i2[j - inputs1] : 1));
   float delta = learning_rates * matrix_g[i] * inp + momentum * matrix_dw[wi];
   if(!isnan(delta))
     {
      matrix_dw[wi] = delta;
      if((delta * matrix_g[i]) > 0)
         matrix_w[wi] = clamp(matrix_w[wi] + delta, -MAX_WEIGHT, MAX_WEIGHT);
     }
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
__kernel void Concat_UpdateWeightsAdam(__global float *matrix_w,        ///<[in,out] Weights matrix (m+1)*n, where m - number of neurons in previous layer and n - number of neurons in current layer
                                       __global const float *matrix_g,  ///<[in] Tensor of gradients at current layer
                                       __global const float *matrix_i1,  ///<[in] Inputs tesor
                                       __global const float *matrix_i2,  ///<[in] Inputs tesor
                                       __global float *matrix_m,        ///<[in,out] Matrix of first momentum
                                       __global float *matrix_v,        ///<[in,out] Matrix of seconfd momentum
                                       const int inputs1,                 ///< Number of inputs
                                       const int inputs2,                 ///< Number of inputs
                                       const float l,                   ///< Learning rates
                                       const float b1,                  ///< First momentum multiplier
                                       const float b2                   ///< Second momentum multiplier
                                      )
  {
   const int i = get_global_id(0);
   const int j = get_global_id(1);
   if(j > (inputs1 + inputs2))
      return;
   const int wi = i * (inputs1 + inputs2 + 1) + j;
   float inp = (j < inputs1 ? matrix_i1[j] : ((j - inputs1) < inputs2 ? matrix_i2[j - inputs1] : 1));
   float weight = matrix_w[wi];
   float g = matrix_g[i] * inp;
   float mt = b1 * matrix_m[wi] + (1 - b1) * g;
   float vt = b2 * matrix_v[wi] + (1 - b2) * pow(g, 2);
   float delta = l * (mt / (sqrt(vt) + 1.0e-37f) - (l1 * sign(weight) + l2 * weight));
   if(delta * g > 0)
      matrix_w[wi] = clamp(matrix_w[wi] + delta, -MAX_WEIGHT, MAX_WEIGHT);
   matrix_m[wi] = mt;
   matrix_v[wi] = vt;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
__kernel void SoftUpdate(__global float *target,        ///<[in,out] Target matrix
                         __global const float *source,  ///<[in] Source matrix
                         const float tau                ///<[in] Multiplicator Tau
                        )
  {
   const int i = get_global_id(0);
   target[i] = source[i] * tau + (1.0f - tau) * target[i];
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
__kernel void SoftUpdateAdam(__global float *target,
                             __global const float *source,
                             __global float *matrix_m,         ///<[in,out] Matrix of first momentum
                             __global float *matrix_v,         ///<[in,out] Matrix of seconfd momentum
                             const float tau,                  ///<[in] Multiplicator Tau
                             const float b1,                   ///< First momentum multiplier
                             const float b2                    ///< Second momentum multiplier
                            )
  {
   const int i = get_global_id(0);
   float m, v, weight;
   m = matrix_m[i];
   v = matrix_v[i];
   weight = target[i];
   float g = source[i] - weight;
   m = b1 * m + (1 - b1) * g;
   v = b2 * v + (1 - b2) * pow(g, 2);
   float delta = tau * m / (v != 0.0f ? sqrt(v) : 1.0f);
   if(delta * g > 0)
      target[i] = clamp(weight + delta, -MAX_WEIGHT, MAX_WEIGHT);
   matrix_m[i] = m;
   matrix_v[i] = v;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
__kernel void SAC_AlphaLogProbs(__global float *outputs,
                                __global float *quantiles,
                                __global float *probs,
                                __global float *alphas,
                                __global float *log_probs,
                                __global float *random,
                                const int count_quants,
                                const int activation
                               )
  {
   const int i = get_global_id(0);
   int shift = i * count_quants;
   float prob = 0;
   float value = 0;
   float sum = 0;
   float rnd = random[i];
//---
   for(int r = 0; r < count_quants; r++)
     {
      prob = probs[shift + r];
      sum += prob;
      if(sum >= rnd || r == (count_quants - 1))
        {
         value = quantiles[shift + r];
         break;
        }
     }
//---
   switch(activation)
     {
      case 0:
         outputs[i] = tanh(value);
         break;
      case 1:
         outputs[i] = 1 / (1 + exp(-value));
         break;
      case 2:
         if(value < 0)
            outputs[i] = value * 0.01f;
         else
            outputs[i] = value;
         break;
      default:
         outputs[i] = value;
         break;
     }
   log_probs[i] = -alphas[i] * log(prob);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
__kernel void SAC_AlphaGradients(__global float *outputs,
                                 __global float *gradient,
                                 __global float *log_probs,
                                 __global float *alphas_grad,
                                 const int activation
                                )
  {
   const int i = get_global_id(0);
   float out = outputs[i];
//---
   float grad = -gradient[i] * log_probs[i];
   switch(activation)
     {
      case 0:
         out = clamp(out, -1.0f, 1.0f);
         grad = clamp(grad + out, -1.0f, 1.0f) - out;
         grad = grad * max(1 - pow(out, 2), 1.0e-4f);
         break;
      case 1:
         out = clamp(out, 0.0f, 1.0f);
         grad = clamp(grad + out, 0.0f, 1.0f) - out;
         grad = grad * max(out * (1 - out), 1.0e-4f);
         break;
      case 2:
         if(out < 0)
            grad = grad * 0.01f;
         break;
      default:
         break;
     }
//---
   alphas_grad[i] = grad;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
__kernel void SAC_OutputGradient(__global float* quantiles,
                                 __global float* delta_taus,
                                 __global float* output_gr,
                                 __global float* quantiles_gr,
                                 __global float* taus_gr,
                                 __global float* output,
                                 const int count_quants,
                                 const int activation
                                )
  {
   size_t action = get_global_id(0);
   int shift = action * count_quants;
   float quant1 = -1e37f;
   float quant2 = 1e37f;
   int pos1 = -1;
   int pos2 = -1;
   float value = output[action];
//---
   for(int i = 0; i < count_quants; i++)
     {
      float quant = quantiles[shift + i];
      switch(activation)
        {
         case 0:
            quant = tanh(quant);
            break;
         case 1:
            quant = 1 / (1 + exp(-quant));
            break;
         case 2:
            if(quant < 0)
               quant = quant * 0.01f;
            break;
        }
      if(value >= quant && quant1 < quant)
        {
         quant1 = quant;
         pos1 = shift + i;
        }
      if(value < quant && quant2 > quant)
        {
         quant2 = quant;
         pos2 = shift + i;
        }
      quantiles_gr[shift + i] = 0.0f;
      taus_gr[shift + i] = 0.0f;
     }
   float gradient = output_gr[action];
   if(quant1 > -1e37f)
     {
      quantiles_gr[pos1] = gradient * delta_taus[pos1];
      taus_gr[pos1] = gradient * quant1;
     }
   if(quant2 < 1e37f)
     {
      quantiles_gr[pos2] = gradient * delta_taus[pos2];
      taus_gr[pos2] = gradient * quant2;
     }
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
__kernel void SAC_CalcLogProbs(__global float *outputs,
                               __global float *quantiles,
                               __global float *probs,
                               __global float *alphas,
                               __global float *log_probs,
                               const int count_quants,
                               const int activation
                              )
  {
   const int i = get_global_id(0);
   int shift = i * count_quants;
   float quant1 = -1e37f;
   float quant2 = 1e37f;
   float prob1 = 0;
   float prob2 = 0;
   float value = outputs[i];
//---
   for(int q = 0; q < count_quants; q++)
     {
      float quant = quantiles[shift + q];
      switch(activation)
        {
         case 0:
            quant = tanh(quant);
            break;
         case 1:
            quant = 1 / (1 + exp(-quant));
            break;
         case 2:
            if(quant < 0)
               quant *= 0.01f;
            break;
         default:
            break;
        }
      if(value >= quant && quant1 < quant)
        {
         quant1 = quant;
         prob1 = probs[shift + q];
        }
      if(value < quant && quant2 > quant)
        {
         quant2 = quant;
         prob2 = probs[shift + q];
        }
     }
//---
   float prob = fabs(value - quant1) / fabs(quant2 - quant1);
   prob = clamp((1 - prob) * prob1 + prob * prob2, 1.0e-3f, 1.0f);
   log_probs[i] = -alphas[i] * log(prob);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
__kernel void Embedding(__global float *inputs,
                        __global float *outputs,
                        __global float *weights,
                        __global int   *windows,
                        __global float *std,
                        const int stack_size
                       )
  {
   const int window_out = get_global_size(0);
   const int pos = get_local_id(0);
   const int emb = get_global_id(1);
   const int emb_total = get_global_size(1);
   const int shift_out = emb * window_out + pos;
   const int step = emb_total * window_out;
   const uint ls = min((uint)get_local_size(0), (uint)LOCAL_ARRAY_SIZE);
//---
   for(int i = stack_size - 1; i > 0; i--)
      outputs[i * step + shift_out] = outputs[(i - 1) * step + shift_out];
   int shift_in = 0;
   for(int i = 0; i < emb; i++)
      shift_in += windows[i];
   const int shift_weights = (shift_in + emb) * window_out;
   const int window_in = windows[emb];
   const int local_pos = (pos >= ls ? pos % (ls - 1) : pos);
   const int local_orders = (window_out + ls - 1) / ls;
   const int local_order = pos / ls;
//---
   __local float temp[LOCAL_ARRAY_SIZE];
   if(local_order == 0)
      temp[local_pos] = 0;
   barrier(CLK_LOCAL_MEM_FENCE);
//---
   float value = weights[shift_weights + window_in * window_out + pos];
   for(int i = 0; i < window_in; i++)
      value += inputs[shift_in + i] * weights[shift_weights + i * window_out + pos];
   for(int i = 0; i < local_orders; i++)
     {
      if(i == local_order)
         temp[local_pos] += value;
      barrier(CLK_LOCAL_MEM_FENCE);
     }
//---
   int count = ls;
   do
     {
      count = (count + 1) / 2;
      if(pos < count)
         temp[pos] += temp[pos + count];
      barrier(CLK_LOCAL_MEM_FENCE);
     }
   while(count > 1);
//---
   value -= temp[0] / (float)window_out;
   barrier(CLK_LOCAL_MEM_FENCE);
//---
   if(local_order == 0)
      temp[local_pos] = 0;
   barrier(CLK_LOCAL_MEM_FENCE);
//---
   for(int i = 0; i < local_orders; i++)
     {
      if(i == local_order)
         temp[local_pos] += pow(value, 2.0f) / (float)window_out;
      barrier(CLK_LOCAL_MEM_FENCE);
     }
//---
   count = ls;
   do
     {
      count = (count + 1) / 2;
      if(pos < count)
         temp[pos] += temp[pos + count];
      barrier(CLK_LOCAL_MEM_FENCE);
     }
   while(count > 1);
//---
   if(temp[0] > 0)
      value /= sqrt(temp[0]);
//---
   outputs[shift_out] = value;
   if(pos == 0)
      std[emb] = sqrt(temp[0]);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
__kernel void EmbeddingHiddenGradient(__global float *inputs_gradient,
                                      __global float *outputs_gradient,
                                      __global float *weights,
                                      __global int   *windows,
                                      __global float *std,
                                      const int window_out
                                     )
  {
   const int pos = get_global_id(0);
   int emb = -1;
   int count = 0;
   do
     {
      emb++;
      count += windows[emb];
     }
   while(count < pos);
   const int shift_out = emb * window_out;
   const int shift_weights = (pos + emb) * window_out;
//---
   float value = 0;
   for(int i = 0; i < window_out; i++)
      value += outputs_gradient[shift_out + i] * weights[shift_weights + i];
   float s = std[emb];
   if(s > 0)
      value /= s;
//---
   inputs_gradient[pos] = value;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
__kernel void EmbeddingUpdateWeightsAdam(__global float *weights,        ///<[in,out] Weights matrix (m+1)*n, where m - number of neurons in previous layer and n - number of neurons in current layer
      __global const float *gradient,  ///<[in] Tensor of gradients at current layer
      __global const float *inputs,     ///<[in] Inputs tesor
      __global float *matrix_m,        ///<[in,out] Matrix of first momentum
      __global float *matrix_v,        ///<[in,out] Matrix of seconfd momentum
      __global int   *windows,
      __global float *std,
      const int window_out,
      const float l,                   ///< Learning rates
      const float b1,                  ///< First momentum multiplier
      const float b2                   ///< Second momentum multiplier
                                        )
  {
   const int i = get_global_id(0);
   int emb = -1;
   int count = 0;
   int shift = 0;
   do
     {
      emb++;
      shift = count;
      count += (windows[emb] + 1) * window_out;
     }
   while(count <= i);
   const int shift_out = emb * window_out;
   int shift_in = shift / window_out - emb;
   shift = (i - shift) / window_out;
   float inp = 1.0f;
   if(shift < windows[emb])
      inp = inputs[shift_in + shift];
//---
   float weight = weights[i];
   float g = gradient[shift_out] * inp / std[emb];
   float mt = b1 * matrix_m[i] + (1 - b1) * g;
   float vt = b2 * matrix_v[i] + (1 - b2) * pow(g, 2);
   float delta = l * (mt / (sqrt(vt) + 1.0e-37f) - (l1 * sign(weight) + l2 * weight));
   if(delta * g > 0)
      weights[i] = clamp(weights[i] + delta, -MAX_WEIGHT, MAX_WEIGHT);
   matrix_m[i] = mt;
   matrix_v[i] = vt;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
__kernel void Transpose(__global float *matrix_in,       ///<[in] Input matrix
                        __global float *matrix_out       ///<[out] Output matrix
                       )
  {
   const int r = get_global_id(0);
   const int c = get_global_id(1);
   const int rows = get_global_size(0);
   const int cols = get_global_size(1);
//---
   matrix_out[c * rows + r]  = matrix_in[r * cols + c];
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
__kernel void MH2AttentionOut(__global float *q,      ///<[in] Matrix of Querys
                              __global float *kv,     ///<[in] Matrix of Keys
                              __global float *score,  ///<[out] Matrix of Scores
                              __global float *out,    ///<[out] Matrix of attention
                              int dimension           ///< Dimension of Key
                             )
  {
//--- init
   const int q_id = get_global_id(0);
   const int k = get_global_id(1);
   const int h = get_global_id(2);
   const int qunits = get_global_size(0);
   const int kunits = get_global_size(1);
   const int heads = get_global_size(2);
   const int shift_q = dimension * (q_id + qunits * h);
   const int shift_k = dimension * (k + kunits * h);
   const int shift_v = dimension * (k + kunits * (heads + h));
   const int shift_s = q_id * kunits * heads + h * kunits + k;
   const uint ls = min((uint)get_local_size(1), (uint)LOCAL_ARRAY_SIZE);
   float koef = sqrt((float)dimension);
   if(koef < 1)
      koef = 1;
   __local float temp[LOCAL_ARRAY_SIZE];
//--- sum of exp
   uint count = 0;
   if(k < ls)
      do
        {
         if((count * ls) < (kunits - k))
           {
            float sum = 0;
            for(int d = 0; d < dimension; d++)
               sum = q[shift_q + d] * kv[shift_k + d];
            sum = exp(sum / koef);
            if(isnan(sum))
               sum = 0;
            temp[k] = (count > 0 ? temp[k] : 0) + sum;
           }
         count++;
        }
      while((count * ls + k) < kunits);
   barrier(CLK_LOCAL_MEM_FENCE);
   count = min(ls, (uint)kunits);
//---
   do
     {
      count = (count + 1) / 2;
      if(k < ls)
         temp[k] += (k < count && (k + count) < kunits ? temp[k + count] : 0);
      if(k + count < ls)
         temp[k + count] = 0;
      barrier(CLK_LOCAL_MEM_FENCE);
     }
   while(count > 1);
//--- score
   float sum = temp[0];
   float sc = 0;
   if(sum != 0)
     {
      for(int d = 0; d < dimension; d++)
         sc = q[shift_q + d] * kv[shift_k + d];
      sc = exp(sc / koef) / sum;
      if(isnan(sc))
         sc = 0;
     }
   score[shift_s] = sc;
   barrier(CLK_LOCAL_MEM_FENCE);
//--- out
   for(int d = 0; d < dimension; d++)
     {
      uint count = 0;
      if(k < ls)
         do
           {
            if((count * ls) < (kunits - k))
              {
               float sum = kv[shift_v + d] * (count == 0 ? sc : score[shift_s + count * ls]);
               if(isnan(sum))
                  sum = 0;
               temp[k] = (count > 0 ? temp[k] : 0) + sum;
              }
            count++;
           }
         while((count * ls + k) < kunits);
      barrier(CLK_LOCAL_MEM_FENCE);
      //---
      count = min(ls, (uint)kunits);
      do
        {
         count = (count + 1) / 2;
         if(k < ls)
            temp[k] += (k < count && (k + count) < kunits ? temp[k + count] : 0);
         if(k + count < ls)
            temp[k + count] = 0;
         barrier(CLK_LOCAL_MEM_FENCE);
        }
      while(count > 1);
      //---
      out[shift_q + d] = temp[0];
     }
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
__kernel void MH2AttentionInsideGradients(__global float *q, __global float *q_g,
      __global float *kv, __global float *kv_g,
      __global float *scores,
      __global float *gradient,
      int kunits)
  {
//--- init
   const int q_id = get_global_id(0);
   const int d = get_global_id(1);
   const int h = get_global_id(2);
   const int qunits = get_global_size(0);
   const int dimension = get_global_size(1);
   const int heads = get_global_size(2);
   const int shift_q = dimension * (q_id + qunits * h) + d;
   const int shift_k = dimension * (q_id + kunits * h) + d;
   const int shift_v = dimension * (q_id + kunits * (heads + h)) + d;
   const int shift_s = q_id * kunits * heads + h * kunits;
   const int shift_g = h * qunits * dimension + d;
   float koef = sqrt((float)dimension);
   if(koef < 1)
      koef = 1;
//--- Calculating Value's gradients
   int step_score = q_id * kunits * heads;
   for(int v = q_id; v < kunits; v += qunits)
     {
      int shift_score = h * kunits + v;
      float grad = 0;
      for(int g = 0; g < qunits; g++)
         grad += gradient[shift_g + g * dimension] * scores[shift_score + g * step_score];
      kv_g[shift_v + v * dimension] = grad;
     }
//--- Calculating Query's gradients
   float grad = 0;
   float out_g = gradient[shift_g + q_id * dimension];
   int shift_val = (heads + h) * kunits * dimension + d;
   int shift_key = h * kunits * dimension + d;
   for(int k = 0; k < kunits; k++)
     {
      float sc_g = 0;
      float sc = scores[shift_s + k];
      for(int v = 0; v < kunits; v++)
         sc_g += scores[shift_s + v] * out_g * kv[shift_val + v * dimension] * ((float)(k == v) - sc);
      grad += sc_g * kv[shift_key + k * dimension];
     }
   q_g[shift_q] = grad / koef;
//--- Calculating Key's gradients
   for(int k = q_id; k < kunits; k += qunits)
     {
      int shift_score = h * kunits + k;
      int shift_val = (heads + h) * kunits * dimension + d;
      grad = 0;
      float val = kv[shift_v];
      for(int scr = 0; scr < qunits; scr++)
        {
         float sc_g = 0;
         int shift_sc = scr * kunits * heads;
         float sc = scores[shift_sc + k];
         for(int v = 0; v < kunits; v++)
            sc_g += scores[shift_sc + v] * gradient[shift_g + scr * dimension] * val * ((float)(k == v) - sc);
         grad += sc_g * q[shift_q + scr * dimension];
        }
      kv_g[shift_k + k * dimension] = grad / koef;
     }
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
__kernel void CGConv_HiddenGradient(__global float *matrix_g,///<[in] Tensor of gradients at current layer
                                    __global float *matrix_f,///<[in] Previous layer Output tensor
                                    __global float *matrix_s,///<[in] Previous layer Output tensor
                                    __global float *matrix_fg,///<[out] Tensor of gradients at previous layer
                                    __global float *matrix_sg,///<[out] Tensor of gradients at previous layer
                                    int activationf,///< Activation type (#ENUM_ACTIVATION)
                                    int activations///< Activation type (#ENUM_ACTIVATION)
                                   )
  {
   int i = get_global_id(0);
//---
   float grad = matrix_g[i];
   float f = matrix_f[i];
   float s = matrix_s[i];
//---
   float sg = grad * f;
   float fg = grad * s;
//---
   switch(activationf)
     {
      case 0:
         f = clamp(f, -1.0f, 1.0f);
         fg = clamp(fg + f, -1.0f, 1.0f) - f;
         fg = fg * max(1 - pow(f, 2), 1.0e-4f);
         break;
      case 1:
         f = clamp(f, 0.0f, 1.0f);
         fg = clamp(fg + f, 0.0f, 1.0f) - f;
         fg = fg * max(f * (1 - f), 1.0e-4f);
         break;
      case 2:
         if(f < 0)
            fg *= 0.01f;
         break;
      default:
         break;
     }
//---
   switch(activations)
     {
      case 0:
         s = clamp(s, -1.0f, 1.0f);
         sg = clamp(sg + s, -1.0f, 1.0f) - s;
         sg = sg * max(1 - pow(s, 2), 1.0e-4f);
         break;
      case 1:
         s = clamp(s, 0.0f, 1.0f);
         sg = clamp(sg + s, 0.0f, 1.0f) - s;
         sg = sg * max(s * (1 - s), 1.0e-4f);
         break;
      case 2:
         if(s < 0)
            sg *= 0.01f;
         break;
      default:
         break;
     }
//---
   matrix_fg[i] = fg;
   matrix_sg[i] = sg;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
__kernel void XCiTFeedForward(__global float *qkv,
                              __global float *score,
                              __global float *out)
  {
   const size_t d = get_local_id(0);
   const size_t dimension = get_local_size(0);
   const size_t u = get_local_id(1);
   const size_t units = get_local_size(1);
   const size_t h = get_global_id(2);
   const size_t heads = get_global_size(2);
//---
   const uint ls_u = min((uint)units, (uint)LOCAL_ARRAY_SIZE);
   const uint ls_d = min((uint)dimension, (uint)LOCAL_ARRAY_SIZE);
   __local float q[LOCAL_ARRAY_SIZE][LOCAL_ARRAY_SIZE];
   __local float k[LOCAL_ARRAY_SIZE][LOCAL_ARRAY_SIZE];
//--- Normalize Query and Key
   for(int cur_d = 0; cur_d < dimension; cur_d += ls_d)
     {
      float q_val = 0;
      float k_val = 0;
      //---
      if(d < ls_d && (cur_d + d) < dimension && u < ls_u)
        {
         for(int count = u; count < units; count += ls_u)
           {
            int shift = count * dimension * heads * 3 + dimension * h + cur_d + d;
            q_val += pow(qkv[shift], 2.0f);
            k_val += pow(qkv[shift + dimension * heads], 2.0f);
           }
         q[u][d] = q_val;
         k[u][d] = k_val;
        }
      barrier(CLK_LOCAL_MEM_FENCE);
      //---
      uint count = ls_u;
      do
        {
         count = (count + 1) / 2;
         if(d < ls_d)
           {
            if(u < ls_u && u < count && (u + count) < units)
              {
               float q_val = q[u][d] + q[u + count][d];
               float k_val = k[u][d] + k[u + count][d];
               q[u + count][d] = 0;
               k[u + count][d] = 0;
               q[u][d] = q_val;
               k[u][d] = k_val;
              }
           }
         barrier(CLK_LOCAL_MEM_FENCE);
        }
      while(count > 1);
      //---
      int shift = u * dimension * heads * 3 + dimension * h + cur_d;
      qkv[shift] = qkv[shift] / sqrt(q[0][d]);
      qkv[shift + dimension * heads] = qkv[shift + dimension * heads] / sqrt(k[0][d]);
      barrier(CLK_LOCAL_MEM_FENCE);
     }
//--- Score
   int step = dimension * heads * 3;
   for(int cur_r = 0; cur_r < dimension; cur_r += ls_u)
     {
      for(int cur_d = 0; cur_d < dimension; cur_d += ls_d)
        {
         if(u < ls_d && d < ls_d)
            q[u][d] = 0;
         barrier(CLK_LOCAL_MEM_FENCE);
         //---
         if((cur_r + u) < ls_d && (cur_d + d) < ls_d)
           {
            int shift_q = dimension * h + cur_d + d;
            int shift_k = dimension * (heads + h) + cur_r + u;
            float scr = 0;
            for(int i = 0; i < units; i++)
               scr += qkv[shift_q + i * step] * qkv[shift_k + i * step];
            scr = exp(scr);
            score[(cur_r + u)*dimension * heads + dimension * h + cur_d + d] = scr;
            q[u][d] += scr;
           }
        }
      barrier(CLK_LOCAL_MEM_FENCE);
      //---
      int count = ls_d;
      do
        {
         count = (count + 1) / 2;
         if(u < ls_d)
           {
            if(d < ls_d && d < count && (d + count) < dimension)
               q[u][d] += q[u][d + count];
            if(d + count < ls_d)
               q[u][d + count] = 0;
           }
         barrier(CLK_LOCAL_MEM_FENCE);
        }
      while(count > 1);
      //---
      if((cur_r + u) < ls_d)
         score[(cur_r + u)*dimension * heads + dimension * h + d] /= q[u][0];
      barrier(CLK_LOCAL_MEM_FENCE);
     }
//---
   int shift_out = dimension * (u * heads + h) + d;
   int shift_s = dimension * (heads * d + h);
   int shift_v = dimension * (heads * (u * 3 + 2) + h);
   float sum = 0;
   for(int i = 0; i < dimension; i++)
      sum += qkv[shift_v + i] * score[shift_s + i];
   out[shift_out] = sum;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
__kernel void XCiTInsideGradients(__global float *qkv, __global float *qkv_g,
                                  __global float *scores,
                                  __global float *gradient)
  {
//--- init
   const int q = get_global_id(0);
   const int d = get_global_id(1);
   const int h = get_global_id(2);
   const int units = get_global_size(0);
   const int dimension = get_global_size(1);
   const int heads = get_global_size(2);
   const int shift_q = dimension * (heads * 3 * q + h);
   const int shift_k = dimension * (heads * (3 * q + 1) + h);
   const int shift_v = dimension * (heads * (3 * q + 2) + h);
   const int shift_g = dimension * (heads * q + h);
   int shift_score = dimension * h;
   int step_score = dimension * heads;
//--- Calculating Value's gradients
   float sum = 0;
   for(int i = 0; i < dimension; i ++)
      sum += gradient[shift_g + i] * scores[shift_score + d + i * step_score];
   qkv_g[shift_v + d] = sum;
//--- Calculating Query's gradients
   float grad = 0;
   float val = qkv[shift_v + d];
   for(int k = 0; k < dimension; k++)
     {
      float sc_g = 0;
      float sc = scores[shift_score + k];
      for(int v = 0; v < dimension; v++)
         sc_g += scores[shift_score + v] * val * gradient[shift_g + v * dimension] * ((float)(k == v) - sc);
      grad += sc_g * qkv[shift_k + k];
     }
   qkv_g[shift_q] = grad;
//--- Calculating Key's gradients
   grad = 0;
   float out_g = gradient[shift_g];
   for(int scr = 0; scr < dimension; scr++)
     {
      float sc_g = 0;
      int shift_sc = scr * dimension * heads;
      float sc = scores[shift_sc + d];
      for(int v = 0; v < dimension; v++)
         sc_g += scores[shift_sc + v] * out_g * qkv[shift_v + v] * ((float)(d == v) - sc);
      grad += sc_g * qkv[shift_q + scr];
     }
   qkv_g[shift_k + d] = grad;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
__kernel void DOTFeedForward(__global float *qkv,
                             __global float *score,
                             __global float *rpb,
                             __global float *out)
  {
   const size_t d = get_local_id(0);
   const size_t dimension = get_local_size(0);
   const size_t u = get_global_id(1);
   const size_t units = get_global_size(1);
   const size_t h = get_global_id(2);
   const size_t heads = get_global_size(2);
//---
   uint step = 3 * dimension * heads;
   uint start = max((int)u - 1, 0);
   uint stop = min((int)u + 1, (int)units - 1);
   uint shift_q = u * step + h * dimension;
   uint shift_k = start * step + dimension * (heads + h);
   uint shift_score = u * 3 * heads;
//---
   const uint ls_d = min((uint)dimension, (uint)LOCAL_ARRAY_SIZE);
   __local float temp[LOCAL_ARRAY_SIZE][3];
//--- Score
   if(d < ls_d)
     {
      for(uint pos = start; pos <= stop; pos++)
        {
         temp[d][pos - start] = 0;
        }
      for(uint dim = d; dim < dimension; dim += ls_d)
        {
         float q = qkv[shift_q + dim];
         for(uint pos = start; pos <= stop; pos++)
           {
            uint i = pos - start;
            temp[d][i] = temp[d][i] + q * qkv[shift_k + i * step + dim];
           }
        }
      barrier(CLK_LOCAL_MEM_FENCE);
      //---
      int count = ls_d;
      do
        {
         count = (count + 1) / 2;
         if(d < count && (d + count) < dimension)
            for(uint i = 0; i <= (stop - start); i++)
              {
               temp[d][i] += temp[d + count][i];
               temp[d + count][i] = 0;
              }
         barrier(CLK_LOCAL_MEM_FENCE);
        }
      while(count > 1);
     }
//---
   if(d == 0)
     {
      float sum = 0;
      for(uint i = 0; i <= (stop - start); i++)
        {
         temp[0][i] = exp(temp[0][i] + rpb[shift_score + i]);
         sum += temp[0][i];
        }
      for(uint i = 0; i <= (stop - start); i++)
        {
         temp[0][i] = temp[0][i] / sum;
         score[shift_score + i] = temp[0][i];
        }
     }
   barrier(CLK_LOCAL_MEM_FENCE);
//---
   int shift_out = dimension * (u * heads + h) + d;
   int shift_v = dimension * (heads * (u * 3 + 2) + h);
   float sum = 0;
   for(uint i = 0; i <= (stop - start); i++)
      sum += qkv[shift_v + i] * temp[0][i];
   out[shift_out] = sum;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
__kernel void DOTInsideGradients(__global float *qkv, __global float *qkv_g,
                                 __global float *scores,
                                 __global float *rpb, __global float *rpb_g,
                                 __global float *gradient)
  {
//--- init
   const uint u = get_global_id(0);
   const uint d = get_global_id(1);
   const uint h = get_global_id(2);
   const uint units = get_global_size(0);
   const uint dimension = get_global_size(1);
   const uint heads = get_global_size(2);
//---
   uint step = 3 * dimension * heads;
   uint start = max((int)u - 1, 0);
   uint stop = min((int)u + 1, (int)units - 1);
   const uint shift_q = u * step + dimension * h + d;
   const uint shift_k = u * step + dimension * (heads + h) + d;
   const uint shift_v = u * step + dimension * (2 * heads + h) + d;
//--- Calculating Value's gradients
   float sum = 0;
   for(uint i = start; i <= stop; i ++)
     {
      int shift_score = i * 3 * heads;
      if(u == i)
        {
         shift_score += (uint)(u > 0);
        }
      else
        {
         if(u > i)
            shift_score += (uint)(start > 0) + 1;
        }
      uint shift_g = dimension * (i * heads + h) + d;
      sum += gradient[shift_g] * scores[shift_score];
     }
   qkv_g[shift_v] = sum;
//--- Calculating Query's gradients
   float grad = 0;
   uint shift_score = u * heads * 3;
   for(int k = start; k <= stop; k++)
     {
      float sc_g = 0;
      float sc = scores[shift_score + k - start];
      for(int v = start; v <= stop; v++)
         for(int dim=0;dim<dimension;dim++)
            sc_g += scores[shift_score + v - start] * 
                    qkv[v * step + dimension * (2 * heads + h) + dim] * 
                    gradient[dimension * (u * heads + h) + dim] * 
                    ((float)(k == v) - sc);
      grad += sc_g * qkv[k * step + dimension * (heads + h) + d];
      if(d == 0)
         rpb_g[shift_score + k - start] = sc_g;
     }
   qkv_g[shift_q] = grad;
//--- Calculating Key's gradients
   grad = 0;
   for(int q = start; q <= stop; q++)
     {
      float sc_g = 0;
      shift_score = q * heads * 3;
      if(u == q)
        {
         shift_score += (uint)(u > 0);
        }
      else
        {
         if(u > q)
            shift_score += (uint)(start > 0) + 1;
        }
      float sc = scores[shift_score];
      for(int v = start; v <= stop; v++)
        {
         shift_score = v * heads * 3;
         if(u == v)
           {
            shift_score += (uint)(u > 0);
           }
         else
           {
            if(u > v)
               shift_score += (uint)(start > 0) + 1;
           }
         for(int dim=0;dim<dimension;dim++)
            sc_g += scores[shift_score] * qkv[shift_v-d+dim] * 
                    gradient[dimension * (v * heads + h) + d] * ((float)(d == v) - sc);
        }
      grad += sc_g * qkv[q * step + dimension * h + d];
     }
   qkv_g[shift_k] = grad;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
__kernel void RPBUpdateAdam(__global float *target,
                             __global const float *gradient,
                             __global float *matrix_m,         ///<[in,out] Matrix of first momentum
                             __global float *matrix_v,         ///<[in,out] Matrix of seconfd momentum
                             const float b1,                   ///< First momentum multiplier
                             const float b2                    ///< Second momentum multiplier
                            )
  {
   const int i = get_global_id(0);
   float m, v, weight;
   m = matrix_m[i];
   v = matrix_v[i];
   weight = target[i];
   float g = gradient[i];
   m = b1 * m + (1 - b1) * g;
   v = b2 * v + (1 - b2) * pow(g, 2);
   float delta = m / (v != 0.0f ? sqrt(v) : 1.0f);
   target[i] = clamp(weight + delta, -MAX_WEIGHT, MAX_WEIGHT);
   matrix_m[i] = m;
   matrix_v[i] = v;
  }
//+------------------------------------------------------------------+
