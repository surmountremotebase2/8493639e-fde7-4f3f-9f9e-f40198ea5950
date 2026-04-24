from surmount.base_class import Strategy, TargetAllocation
from surmount.logging import log
from surmount.data import Asset, InstitutionalOwnership
import pandas_ta as ta
import pandas as pd

def SMAVol(ticker, data, length):
   '''Calculate the moving average of trading volume

   :param ticker: a string ticker
   :param data: data as provided from the OHLCV data function
   :param length: the window

   :return: list with float SMA
   '''
   close = [i[ticker]["volume"] for i in data]
   d = ta.sma(pd.Series(close), length=length)
   if d is None:
      return None
   return d.tolist()

class TradingStrategy(Strategy):
   def __init__(self):
      self.tickers = ["VIRT"]
      self.data_list = []

   @property
   def interval(self):
      return "1day"

   @property
   def assets(self):
      return self.tickers

   @property
   def data(self):
      return self.data_list

   def run(self, data):
      vols = [i["VIRT"]["volume"] for i in data["ohlcv"]]
      smavols = SMAVol("VIRT", data["ohlcv"], 30)
      smavols2 = SMAVol("VIRT", data["ohlcv"], 10)

      if len(vols)<=4:
            return TargetAllocation({})

      try:
         if smavols2[-1]/smavols[-1]-1>0:
               out = smavols2[-1]/smavols[-1]-1
         else: out = 0
      except: return None
      return TargetAllocation({"VIRT": min(0.95, (out*5)**(1/3))})