//+------------------------------------------------------------------+
//|                                                   kmeans_net.mq5 |
//|                                              Copyright 2021, DNG |
//|                                https://www.mql5.com/en/users/dng |
//+------------------------------------------------------------------+
#property copyright "Copyright 2022, DNG"
#property link      "https://www.mql5.com/en/users/dng"
#property version   "1.00"
//+------------------------------------------------------------------+
//| Includes                                                         |
//+------------------------------------------------------------------+
#include "kmeans.mqh"
#include <Trade\SymbolInfo.mqh>
#include <Indicators\Oscilators.mqh>
//---
#define FileName        Symb.Name()+"_"+EnumToString((ENUM_TIMEFRAMES)Period())+"_"+IntegerToString(Clusters,3)+StringSubstr(__FILE__,0,StringFind(__FILE__,".",0))
#define CSV             __FILE__+".csv"
//---
enum ENUM_SIGNAL
  {
   Sell = -1,
   Undefine = 0,
   Buy = 1
  };
//+------------------------------------------------------------------+
//|   input parameters                                               |
//+------------------------------------------------------------------+
input int                  StudyPeriod =  2;            //Study period, years
input uint                 HistoryBars =  20;            //Depth of history
input int                  Clusters    =  500;           //Clusters
ENUM_TIMEFRAMES            TimeFrame   =  PERIOD_CURRENT;
//---
input group                "---- RSI ----"
input int                  RSIPeriod   =  14;            //Period
input ENUM_APPLIED_PRICE   RSIPrice    =  PRICE_CLOSE;   //Applied price
//---
input group                "---- CCI ----"
input int                  CCIPeriod   =  14;            //Period
input ENUM_APPLIED_PRICE   CCIPrice    =  PRICE_TYPICAL; //Applied price
//---
input group                "---- ATR ----"
input int                  ATRPeriod   =  14;            //Period
//---
input group                "---- MACD ----"
input int                  FastPeriod  =  12;            //Fast
input int                  SlowPeriod  =  26;            //Slow
input int                  SignalPeriod =  9;            //Signal
input ENUM_APPLIED_PRICE   MACDPrice   =  PRICE_CLOSE;   //Applied price
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CSymbolInfo         *Symb;
MqlRates            Rates[];
CNet                *Net;
CBufferFloat        *TempData;
CiRSI               *RSI;
CiCCI               *CCI;
CiATR               *ATR;
CiMACD              *MACD;
CKmeans             *Kmeans;
//---
float               dError;
float               dUndefine;
float               dForecast;
float               dPrevSignal;
datetime             dtStudied;
bool                 bEventStudy;
//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
//---
   Symb = new CSymbolInfo();
   if(CheckPointer(Symb) == POINTER_INVALID || !Symb.Name(_Symbol))
      return INIT_FAILED;
   Symb.Refresh();
//---
   RSI = new CiRSI();
   if(CheckPointer(RSI) == POINTER_INVALID || !RSI.Create(Symb.Name(), TimeFrame, RSIPeriod, RSIPrice))
      return INIT_FAILED;
//---
   CCI = new CiCCI();
   if(CheckPointer(CCI) == POINTER_INVALID || !CCI.Create(Symb.Name(), TimeFrame, CCIPeriod, CCIPrice))
      return INIT_FAILED;
//---
   ATR = new CiATR();
   if(CheckPointer(ATR) == POINTER_INVALID || !ATR.Create(Symb.Name(), TimeFrame, ATRPeriod))
      return INIT_FAILED;
//---
   MACD = new CiMACD();
   if(CheckPointer(MACD) == POINTER_INVALID || !MACD.Create(Symb.Name(), TimeFrame, FastPeriod, SlowPeriod, SignalPeriod, MACDPrice))
      return INIT_FAILED;
//---
   Kmeans = new CKmeans();
   if(CheckPointer(Kmeans) == POINTER_INVALID)
      return INIT_FAILED;
//---
   Net = new CNet(NULL);
   ResetLastError();
   if(CheckPointer(Net) == POINTER_INVALID || !Net.Load(FileName + ".nnw", dError, dUndefine, dForecast, dtStudied, false))
     {
      printf("%s - %d -> Error of read %s prev Net %d", __FUNCTION__, __LINE__, FileName + ".nnw", GetLastError());
      CArrayObj *Topology = new CArrayObj();
      if(CheckPointer(Topology) == POINTER_INVALID)
         return INIT_FAILED;
      //--- 0
      CLayerDescription *desc = new CLayerDescription();
      if(CheckPointer(desc) == POINTER_INVALID)
         return INIT_FAILED;
      desc.count = (int)Clusters;
      desc.type = defNeuronBaseOCL;
      desc.optimization = ADAM;
      desc.activation = None;
      if(!Topology.Add(desc))
         return INIT_FAILED;
      //--- 1
      desc = new CLayerDescription();
      if(CheckPointer(desc) == POINTER_INVALID)
         return INIT_FAILED;
      desc.count = 300;
      desc.type = defNeuron;
      desc.activation = LReLU;
      desc.optimization = ADAM;
      if(!Topology.Add(desc))
         return INIT_FAILED;
      //--- 2
      desc = new CLayerDescription();
      if(CheckPointer(desc) == POINTER_INVALID)
         return INIT_FAILED;
      desc.count = 300;
      desc.type = defNeuron;
      desc.activation = TANH;
      desc.optimization = ADAM;
      if(!Topology.Add(desc))
         return INIT_FAILED;
      //--- 3
      desc = new CLayerDescription();
      if(CheckPointer(desc) == POINTER_INVALID)
         return INIT_FAILED;
      desc.count = 300;
      desc.type = defNeuron;
      desc.activation = TANH;
      desc.optimization = ADAM;
      if(!Topology.Add(desc))
         return INIT_FAILED;
      //--- 4
      desc = new CLayerDescription();
      if(CheckPointer(desc) == POINTER_INVALID)
         return INIT_FAILED;
      desc.count = 3;
      desc.type = defNeuron;
      desc.activation = SIGMOID;
      desc.optimization = ADAM;
      if(!Topology.Add(desc))
         return INIT_FAILED;
      delete Net;
      Net = new CNet(Topology);
      delete Topology;
      if(CheckPointer(Net) == POINTER_INVALID)
         return INIT_FAILED;
      dError = -1;
      dUndefine = 0;
      dForecast = 0;
      dtStudied = 0;
     }
//---
   TempData = new CArrayFloat();
   if(CheckPointer(TempData) == POINTER_INVALID)
      return INIT_FAILED;
//---
   bEventStudy = EventChartCustom(ChartID(), 1, (long)MathMax(0, MathMin(iTime(Symb.Name(), PERIOD_CURRENT, (int)(100 * Net.recentAverageSmoothingFactor * (dForecast >= 70 ? 1 : 10))), dtStudied)), 0, "Init");
//---
   return(INIT_SUCCEEDED);
  }
//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
//---
   if(CheckPointer(Symb) != POINTER_INVALID)
      delete Symb;
//---
   if(CheckPointer(Kmeans) != POINTER_INVALID)
      delete Kmeans;
//---
   if(CheckPointer(RSI) != POINTER_INVALID)
      delete RSI;
//---
   if(CheckPointer(CCI) != POINTER_INVALID)
      delete CCI;
//---
   if(CheckPointer(ATR) != POINTER_INVALID)
      delete ATR;
//---
   if(CheckPointer(MACD) != POINTER_INVALID)
      delete MACD;
//---
   if(CheckPointer(Net) != POINTER_INVALID)
      delete Net;
   if(CheckPointer(TempData) != POINTER_INVALID)
      delete TempData;
  }
//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
  {
//---
   if(!bEventStudy && (dPrevSignal == -2 || dtStudied < SeriesInfoInteger(Symb.Name(), TimeFrame, SERIES_LASTBAR_DATE)))
      bEventStudy = EventChartCustom(ChartID(), 1, (long)MathMax(0, MathMin(iTime(Symb.Name(), PERIOD_CURRENT, (int)(100 * Net.recentAverageSmoothingFactor * (dForecast >= 70 ? 1 : 10))), dtStudied)), 0, "New Bar");
//---
   Comment(StringFormat("Calling event %s; PrevSignal %.5f; Model trained %s -> %s", (string)bEventStudy, dPrevSignal, TimeToString(dtStudied), TimeToString(SeriesInfoInteger(Symb.Name(), TimeFrame, SERIES_LASTBAR_DATE))));
  }
//+------------------------------------------------------------------+
//| Trade function                                                   |
//+------------------------------------------------------------------+
void OnTrade()
  {
//---
  }
//+------------------------------------------------------------------+
//| TradeTransaction function                                        |
//+------------------------------------------------------------------+
void OnTradeTransaction(const MqlTradeTransaction& trans,
                        const MqlTradeRequest& request,
                        const MqlTradeResult& result)
  {
//---
  }
//+------------------------------------------------------------------+
//| ChartEvent function                                              |
//+------------------------------------------------------------------+
void OnChartEvent(const int id,
                  const long &lparam,
                  const double &dparam,
                  const string &sparam)
  {
//---
   if(id == 1001)
     {
      Net.TrainMode(true);
      Train(lparam);
      bEventStudy = false;
      OnTick();
     }
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void Train(datetime StartTrainBar = 0)
  {
   COpenCLMy *opencl = OpenCLCreate(cl_unsupervised);
   if(CheckPointer(opencl) == POINTER_INVALID)
     {
      ExpertRemove();
      return;
     }
   if(!Kmeans.SetOpenCL(opencl))
     {
      delete opencl;
      ExpertRemove();
      return;
     }
//---
   MqlDateTime start_time;
   TimeCurrent(start_time);
   start_time.year -= StudyPeriod;
   if(start_time.year <= 0)
      start_time.year = 1900;
   datetime st_time = StructToTime(start_time);
//---
   int bars = CopyRates(Symb.Name(), TimeFrame, st_time, TimeCurrent(), Rates);
   if(!RSI.BufferResize(bars) || !CCI.BufferResize(bars) || !ATR.BufferResize(bars) || !MACD.BufferResize(bars))
     {
      ExpertRemove();
      return;
     }
   if(!ArraySetAsSeries(Rates, true))
     {
      ExpertRemove();
      return;
     }
//---
   RSI.Refresh();
   CCI.Refresh();
   ATR.Refresh();
   MACD.Refresh();
//---
   int handl = FileOpen(StringFormat("kmeans_%d.net", Clusters), FILE_READ | FILE_BIN);
   if(handl == INVALID_HANDLE)
     {
      ExpertRemove();
      return;
     }
   if(FileReadInteger(handl) != Kmeans.Type())
     {
      ExpertRemove();
      return;
     }
   bool result = Kmeans.Load(handl);
   FileClose(handl);
   if(!result)
     {
      ExpertRemove();
      return;
     }
//---
   int total = bars - (int)HistoryBars - 1;
   double data[], fractals[];
   if(ArrayResize(data, total * 8 * HistoryBars) <= 0 ||
      ArrayResize(fractals, total * 3) <= 0)
     {
      ExpertRemove();
      return;
     }
//---
   for(int i = 0; (i < total && !IsStopped()); i++)
     {
      Comment(StringFormat("Create data: %d of %d", i, total));
      for(int b = 0; b < (int)HistoryBars; b++)
        {
         int bar = i + b;
         int shift = (i * (int)HistoryBars + b) * 8;
         double open = Rates[bar].open;
         data[shift] = open - Rates[bar].low;
         data[shift + 1] = Rates[bar].high - open;
         data[shift + 2] = Rates[bar].close - open;
         data[shift + 3] = RSI.GetData(MAIN_LINE, bar);
         data[shift + 4] = CCI.GetData(MAIN_LINE, bar);
         data[shift + 5] = ATR.GetData(MAIN_LINE, bar);
         data[shift + 6] = MACD.GetData(MAIN_LINE, bar);
         data[shift + 7] = MACD.GetData(SIGNAL_LINE, bar);
        }
      int shift = i * 3;
      int bar = i + 1;
      fractals[shift] = (int)(Rates[bar - 1].high <= Rates[bar].high && Rates[bar + 1].high < Rates[bar].high);
      fractals[shift + 1] = (int)(Rates[bar - 1].low >= Rates[bar].low && Rates[bar + 1].low > Rates[bar].low);
      fractals[shift + 2] = (int)((fractals[shift] + fractals[shift]) == 0);
     }
   if(IsStopped())
     {
      ExpertRemove();
      return;
     }
   CBufferFloat *Data = new CBufferFloat();
   if(CheckPointer(Data) == POINTER_INVALID ||
      !Data.AssignArray(data))
      return;
   CBufferFloat *Fractals = new CBufferFloat();
   if(CheckPointer(Fractals) == POINTER_INVALID ||
      !Fractals.AssignArray(fractals))
      return;
//---
   ResetLastError();
   CBufferFloat *softmax = Kmeans.SoftMax(Data);
   if(CheckPointer(softmax) == POINTER_INVALID)
     {
      printf("Runtime error %d", GetLastError());
      ExpertRemove();
      return;
     }
//---
   if(CheckPointer(TempData) == POINTER_INVALID)
     {
      TempData = new CArrayFloat();
      if(CheckPointer(TempData) == POINTER_INVALID)
        {
         ExpertRemove();
         return;
        }
     }
   delete opencl;
   double prev_un, prev_for, prev_er;
   dUndefine = 0;
   dForecast = 0;
   dError = -1;
   dPrevSignal = 0;
   bool stop = false;
   int count = 0;
   do
     {
      prev_un = dUndefine;
      prev_for = dForecast;
      prev_er = dError;
      ENUM_SIGNAL bar = Undefine;
      //---
      stop = IsStopped();
      for(int it = 0; (it < total - 300 && !IsStopped()); it++)
        {
         int i = (int)((MathRand() * MathRand() / MathPow(32767, 2)) * (total - 300)) + 300;
         TempData.Clear();
         int shift = i * Clusters;
         if(!TempData.Reserve(Clusters))
           {
            if(CheckPointer(Data) == POINTER_DYNAMIC)
               delete Data;
            if(CheckPointer(Fractals) == POINTER_DYNAMIC)
               delete Fractals;
            if(CheckPointer(softmax) == POINTER_DYNAMIC)
               delete softmax;
            if(CheckPointer(opencl) == POINTER_DYNAMIC)
               delete opencl;
            Comment("");
            //---
            ExpertRemove();
            return;
           }
         for(int c = 0; c < Clusters; c++)
            if(!TempData.Add(softmax.At(shift + c)))
              {
               if(CheckPointer(Data) == POINTER_DYNAMIC)
                  delete Data;
               if(CheckPointer(Fractals) == POINTER_DYNAMIC)
                  delete Fractals;
               if(CheckPointer(softmax) == POINTER_DYNAMIC)
                  delete softmax;
               if(CheckPointer(opencl) == POINTER_DYNAMIC)
                  delete opencl;
               Comment("");
               //---
               ExpertRemove();
               return;
              }
         if(!Net.feedForward(TempData))
           {
            if(CheckPointer(Data) == POINTER_DYNAMIC)
               delete Data;
            if(CheckPointer(Fractals) == POINTER_DYNAMIC)
               delete Fractals;
            if(CheckPointer(softmax) == POINTER_DYNAMIC)
               delete softmax;
            if(CheckPointer(opencl) == POINTER_DYNAMIC)
               delete opencl;
            Comment("");
            //---
            ExpertRemove();
            return;
           }
         Net.getResults(TempData);
         //---
         float sum = 0;
         for(int res = 0; res < 3; res++)
           {
            float temp = exp(TempData.At(res));
            sum += temp;
            TempData.Update(res, temp);
           }
         for(int res = 0; (res < 3 && sum > 0); res++)
            TempData.Update(res, TempData.At(res) / sum);
         //---
         switch(TempData.Maximum(0, 3))
           {
            case 1:
               dPrevSignal = (TempData[1] != TempData[2] ? TempData[1] : 0);
               break;
            case 2:
               dPrevSignal = -TempData[2];
               break;
            default:
               dPrevSignal = 0;
               break;
           }
         string s = StringFormat("Study -> Era %d -> %.2f -> Undefine %.2f%% foracast %.2f%%\n %d of %d -> %.2f%% \nError %.2f\n%s -> %.2f ->> Buy %.5f - Sell %.5f - Undef %.5f", count, dError, dUndefine, dForecast, it + 1, total - 300, (double)(it + 1.0) / (total - 300) * 100, Net.getRecentAverageError(), EnumToString(DoubleToSignal(dPrevSignal)), dPrevSignal, TempData[1], TempData[2], TempData[0]);
         Comment(s);
         stop = IsStopped();
         if(!stop)
           {
            shift = i * 3;
            TempData.Clear();
            TempData.Add(Fractals.At(shift + 2));
            TempData.Add(Fractals.At(shift));
            TempData.Add(Fractals.At(shift + 1));
            Net.backProp(TempData,NULL,NULL);
            ENUM_SIGNAL signal = DoubleToSignal(dPrevSignal);
            if(signal != Undefine)
              {
               if((signal == Sell && Fractals.At(shift + 1) == 1) || (signal == Buy && Fractals.At(shift) == 1))
                  dForecast += (100 - dForecast) / Net.recentAverageSmoothingFactor;
               else
                  dForecast -= dForecast / Net.recentAverageSmoothingFactor;
               dUndefine -= dUndefine / Net.recentAverageSmoothingFactor;
              }
            else
              {
               if(Fractals.At(shift + 2) == 1)
                  dUndefine += (100 - dUndefine) / Net.recentAverageSmoothingFactor;
              }
           }
        }
      count++;
      for(int i = 0; i < 300; i++)
        {
         TempData.Clear();
         int shift = i * Clusters;
         if(!TempData.Reserve(Clusters))
           {
            if(CheckPointer(Data) == POINTER_DYNAMIC)
               delete Data;
            if(CheckPointer(Fractals) == POINTER_DYNAMIC)
               delete Fractals;
            if(CheckPointer(softmax) == POINTER_DYNAMIC)
               delete softmax;
            if(CheckPointer(opencl) == POINTER_DYNAMIC)
               delete opencl;
            Comment("");
            //---
            ExpertRemove();
            return;
           }
         for(int c = 0; c < Clusters; c++)
            if(!TempData.Add(softmax.At(shift + c)))
              {
               if(CheckPointer(Data) == POINTER_DYNAMIC)
                  delete Data;
               if(CheckPointer(Fractals) == POINTER_DYNAMIC)
                  delete Fractals;
               if(CheckPointer(softmax) == POINTER_DYNAMIC)
                  delete softmax;
               if(CheckPointer(opencl) == POINTER_DYNAMIC)
                  delete opencl;
               Comment("");
               //---
               ExpertRemove();
               return;
              }
         if(!Net.feedForward(TempData))
           {
            if(CheckPointer(Data) == POINTER_DYNAMIC)
               delete Data;
            if(CheckPointer(Fractals) == POINTER_DYNAMIC)
               delete Fractals;
            if(CheckPointer(softmax) == POINTER_DYNAMIC)
               delete softmax;
            if(CheckPointer(opencl) == POINTER_DYNAMIC)
               delete opencl;
            Comment("");
            //---
            ExpertRemove();
            return;
           }
         Net.getResults(TempData);
         //---
         float sum = 0;
         for(int res = 0; res < 3; res++)
           {
            float temp = exp(TempData.At(res));
            sum += temp;
            TempData.Update(res, temp);
           }
         for(int res = 0; (res < 3 && sum > 0); res++)
            TempData.Update(res, TempData.At(res) / sum);
         //---
         switch(TempData.Maximum(0, 3))
           {
            case 1:
               dPrevSignal = (TempData[1] != TempData[2] ? TempData[1] : 0);
               break;
            case 2:
               dPrevSignal = -TempData[2];
               break;
            default:
               dPrevSignal = 0;
               break;
           }
         if(DoubleToSignal(dPrevSignal) == Undefine)
            DeleteObject(Rates[i + 2].time);
         else
            DrawObject(Rates[i + 2].time, dPrevSignal, Rates[i + 2].high, Rates[i + 2].low);
        }
      if(!stop)
        {
         dError = Net.getRecentAverageError();
         Net.Save(FileName + ".nnw", dError, dUndefine, dForecast, Rates[0].time, false);
         printf("Era %d -> error %.2f %% forecast %.2f", count, dError, dForecast);
         ChartScreenShot(0, FileName + IntegerToString(count) + ".png", 750, 400);
         int h = FileOpen(FileName + ".csv", FILE_READ | FILE_WRITE | FILE_CSV);
         if(h != INVALID_HANDLE)
           {
            FileSeek(h, 0, SEEK_END);
            FileWrite(h, lr, count, dError, dUndefine, dForecast);
            FileFlush(h);
            FileClose(h);
           }
        }
     }
   while((!(DoubleToSignal(dPrevSignal) != Undefine || dForecast > 70) || !(dError < 0.1 && MathAbs(dError - prev_er) < 0.01 && MathAbs(dUndefine - prev_un) < 0.1 && MathAbs(dForecast - prev_for) < 0.1)) && !stop);
//---
   if(CheckPointer(Data) == POINTER_DYNAMIC)
      delete Data;
   if(CheckPointer(Fractals) == POINTER_DYNAMIC)
      delete Fractals;
   if(CheckPointer(softmax) == POINTER_DYNAMIC)
      delete softmax;
   if(CheckPointer(TempData) == POINTER_DYNAMIC)
      delete TempData;
   if(CheckPointer(opencl) == POINTER_DYNAMIC)
      delete opencl;
   Comment("");
//---
   ExpertRemove();
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
ENUM_SIGNAL DoubleToSignal(double value)
  {
   value = NormalizeDouble(value, 1);
   if(MathAbs(value) > 1 || MathAbs(value) <= 0)
      return Undefine;
   if(value > 0)
      return Buy;
   else
      return Sell;
//---
   return Undefine;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void DrawObject(datetime time, double signal, double high, double low)
  {
   double price = 0;
   int arrow = 0;
   color clr = 0;
   ENUM_ARROW_ANCHOR anch = ANCHOR_BOTTOM;
   switch(DoubleToSignal(signal))
     {
      case Buy:
         price = low;
         arrow = 217;
         clr = clrBlue;
         anch = ANCHOR_TOP;
         break;
      case Sell:
         price = high;
         arrow = 218;
         clr = clrRed;
         anch = ANCHOR_BOTTOM;
         break;
     }
   if(price == 0 || arrow == 0)
      return;
//---
   string name = TimeToString(time);
   if(ObjectFind(0, name) < 0)
     {
      ResetLastError();
      if(!ObjectCreate(0, name, OBJ_ARROW, 0, time, 0))
        {
         printf("Error of creating object %d", GetLastError());
         return;
        }
     }
//printf("%s - %d -> %s",__FUNCTION__,__LINE__,name);
   ObjectSetDouble(0, name, OBJPROP_PRICE, price);
   ObjectSetInteger(0, name, OBJPROP_ARROWCODE, arrow);
   ObjectSetInteger(0, name, OBJPROP_COLOR, clr);
   ObjectSetInteger(0, name, OBJPROP_ANCHOR, anch);
   ObjectSetString(0, name, OBJPROP_TOOLTIP, EnumToString(DoubleToSignal(signal)) + " " + DoubleToString(signal, 5));
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void DeleteObject(datetime time)
  {
   string name = TimeToString(time);
   if(ObjectFind(0, name) >= 0)
      ObjectDelete(0, name);
  }
//+------------------------------------------------------------------+
