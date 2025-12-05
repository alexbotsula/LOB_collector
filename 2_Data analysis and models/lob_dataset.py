import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Union
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

class LOBDataset2(Dataset):
    def __init__(self, \
            data: List[Dict], \
            window_size: int = 10, \
            prediction_horizon: int = 60, \
            ntile: int = 10, \
            time_step: str = '30T'):
        self.window_size = window_size
        self.prediction_horizon = prediction_horizon
        self.ntile = ntile
        self.data = data
        self.time_step = time_step
        self.tensor_data = self._to_tensor(data)
        self.target_data = self._to_target(self.tensor_data)

    def _to_target(self, tensor_data: torch.Tensor) -> torch.Tensor:
        prev = (tensor_data[:, :-self.prediction_horizon, self.ntile] + tensor_data[:, :-self.prediction_horizon, self.ntile-1]) / 2
        next = (tensor_data[:, self.prediction_horizon:, self.ntile] + tensor_data[:, self.prediction_horizon:, self.ntile-1]) / 2
        target = (next - prev) > 0 
        return target

    def _ntile(self, prices: np.ndarray, volumes: np.ndarray) -> np.ndarray:
        cum_vol = np.cumsum(volumes)
        cum_pct = cum_vol / cum_vol[-1]
        levels = np.linspace(1/self.ntile, 1-1/self.ntile, self.ntile)
        indices = np.searchsorted(cum_pct, levels)
        price_levels = prices[indices]
        return price_levels

    def _to_tensor(self, data: List[Dict]) -> torch.Tensor:
        df = data.reset_index()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        tensor_list = [self._to_tensor_snapshot(group) for _, group in df.resample(self.time_step)]

        return torch.cat(tensor_list, dim=1)


    def _to_tensor_snapshot(self, data: List[Dict]) -> torch.Tensor:
        df = self._preprocess_data(data)
        self._t_df = df
        #Iterate across symbols
        # Tensor to accumulate all prices
        all_prices = torch.tensor([], dtype=torch.float32)
        for symbol in df.index.get_level_values('symbol').unique():
            symb_prices = torch.tensor([], dtype=torch.float32)
            for obs in df.loc[(slice(None), symbol), :].index.get_level_values('order_number').unique().sort_values():
                obs_data = df.loc[(obs, symbol), :]
                ask_volumes = np.array([ask[1] for ask in obs_data['asks']], dtype=np.float32)
                bid_volumes = np.array([bid[1] for bid in obs_data['bids']], dtype=np.float32)
                ask_prices = np.array([ask[0] for ask in obs_data['asks']], dtype=np.float32)
                bid_prices = np.array([bid[0] for bid in obs_data['bids']], dtype=np.float32)

                ask_prices_ntiles = self._ntile(ask_prices, ask_volumes)
                bid_prices_ntiles = self._ntile(bid_prices, bid_volumes)
                mid_price = (ask_prices_ntiles[0] + bid_prices_ntiles[0]) / 2
                # scale = ask_prices_ntiles[-1] - bid_prices_ntiles[-1]
                # ask_price_norm = torch.tensor((ask_prices_ntiles - mid_price) / scale, dtype=torch.float32)
                # bid_price_norm = torch.tensor((bid_prices_ntiles - mid_price) / scale, dtype=torch.float32)
                ask_price = torch.tensor(ask_prices_ntiles, dtype=torch.float32)
                bid_price = torch.tensor(bid_prices_ntiles, dtype=torch.float32)
                obs_prices = torch.cat([bid_price.flip(0), ask_price]) #Bid prices should go descending
                symb_prices = torch.cat([symb_prices, obs_prices.unsqueeze(0)], dim=0)

            if all_prices.shape[0] == 0:
                all_prices = torch.cat([all_prices, symb_prices.unsqueeze(0)], dim=0)   
            else:
                min_n_obs = np.min([all_prices.shape[1], symb_prices.shape[0]]) # Time dimensions
                all_prices = torch.cat([all_prices[:, :min_n_obs, :], symb_prices[:min_n_obs, :].unsqueeze(0)], dim=0)

        return all_prices
    
    def _preprocess_data(self, data: List[Dict]) -> pd.DataFrame:
        df = data.reset_index()
        #Order number of each record for a symbol
        df['order_number'] = df.sort_values('timestamp').groupby('symbol').cumcount()
        #Set index to order_number + symbol
        df.set_index('order_number', inplace=True)
        df.set_index('symbol', append=True, inplace=True)
        return df

    def plot_LOB(self, symbol_index: str, time_index_range: Tuple[int, int]):
        slice_data = self._normalise(self.tensor_data[[symbol_index], time_index_range[0]:time_index_range[1], :])[0]
        # Plot as heatmap
        plt.imshow(slice_data, aspect='auto')
        plt.show()

    def _normalise(self, tensor_data: torch.Tensor) -> torch.Tensor:
        mid_price = (tensor_data[:, :, [self.ntile]] + tensor_data[:, :, [self.ntile-1]]) / 2
        scale = tensor_data[:, :, [0]] - tensor_data[:, :, [2 * self.ntile - 1]]
        return (tensor_data - mid_price) / scale

    def __len__(self):
        return self.tensor_data.shape[1] - self.window_size - self.prediction_horizon

    def __getitem__(self, idx):
        return self._normalise(self.tensor_data[:, idx:idx + self.window_size, :]), self.target_data[:, idx]

    