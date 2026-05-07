from surmount.base_class import Strategy, TargetAllocation
from surmount.logging import log
from statistics import stdev
from surmount.technical_indicators import SMA


class TradingStrategy(Strategy):

   @property
   def assets(self):
      return ["KO", "PEP", "SPY", "TQQQ"]

   @property
   def interval(self):
      return "1day"

   def run(self, data):
      if (len(data["ohlcv"]) < 4): 
         return TargetAllocation({})
      ko_price = data["ohlcv"][-1]["KO"]["close"]
      pep_price = data["ohlcv"][-1]["PEP"]["close"]

      ratio = [data["ohlcv"][i]["KO"]["close"]/data["ohlcv"][i]["PEP"]["close"] for i in range(len(data["ohlcv"]))]
      mean = sum(ratio)/len(ratio)
      dev = stdev(ratio)

      ko_stake = 0
      pep_stake = 0
      spy_stake = 0.8
      tqqq_stake = 0.2

      if ratio[-1] > mean + dev/1.2:
         ko_stake = 0
         pep_stake = 0.85
         spy_stake = 0.1
         tqqq_stake = 0.05
      
      elif ratio[-1] < mean - dev/1.2:
         ko_stake = 0.85
         pep_stake = 0
         spy_stake = 0.1
         tqqq_stake = 0.05

      ma = SMA("SPY", data["ohlcv"], 20)
      if ma: 
         ma = ma[-1]
      else: return TargetAllocation({})

      ma2 = SMA("SPY", data["ohlcv"], 32)
      if ma2:
         ma2 = ma2[-1]
      else: return TargetAllocation({})
      
      if ma < 0.99*ma2:
         ko_stake = ko_stake/3
         pep_stake = pep_stake/3
         spy_stake = 0.3
         tqqq_stake = 0.1
         if ma < 0.98 * ma2:
            spy_stake = 0.2
            tqqq_stake = 0.3

      return TargetAllocation({"KO": ko_stake, "PEP": pep_stake, "SPY": spy_stake, "TQQQ": tqqq_stake})