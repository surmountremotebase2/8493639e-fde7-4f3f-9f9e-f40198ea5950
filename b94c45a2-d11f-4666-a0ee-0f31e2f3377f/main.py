from surmount.base_class import Strategy, TargetAllocation, backtest
from surmount.logging import log
from statistics import stdev
from surmount.technical_indicators import SMA, BB, RSI

class TradingStrategy(Strategy):

   @property
   def assets(self):
      return ["SPY", "QQQ", "VTI", "VXUS"]

   @property
   def interval(self):
      return "1day"

   def run(self, data):
      holdings = data["holdings"]
      data = data["ohlcv"]
      
      allocation_dict = {}
      rsi_dict = {}
      for ticker in self.assets:
         try: rsi_dict[ticker] = RSI(ticker, data, 14)[-1]
         except: rsi_dict[ticker] = 1

      allocation_dict = {i: rsi_dict[i]/(sum(rsi_dict.values())+10) for i in self.assets}
      for key in allocation_dict:
         if abs(allocation_dict[key]-holdings.get(key, 0))>0.02:
            return TargetAllocation(allocation_dict)
      return None
