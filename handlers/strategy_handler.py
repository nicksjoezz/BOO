import pandas as pd
import time
from strategy_utils import ut_bot
from indicators import add_indicators
from model_manager import model_manager

class StrategyHandler:
    def __init__(self, bot):
        self.bot = bot
        self.strat_params = [
            (1, 10), (2, 20), (3, 30), (1, 20), (2, 10),
            (3, 20), (1, 30), (2, 30), (3, 10), (1.5, 15)
        ]

    async def check_signals(self, history_df):
        if len(history_df) < 200:
            return None

        symbol = self.bot.config['symbol']
        strategy_idx = int(self.bot.config['strategy'])
        a, c = self.strat_params[strategy_idx-1]

        try:
            # 1. Indicator calculation
            df = history_df.copy()
            df = add_indicators(df)

            # 2. Check Trend First (IMBA Algo)
            # We look at the last closed candle (index -2)
            last_closed = df.iloc[-2]
            is_uptrend = last_closed['imba_is_uptrend'] == 1
            is_downtrend = last_closed['imba_is_downtrend'] == 1
            trend_str = "UPTREND" if is_uptrend else ("DOWNTREND" if is_downtrend else "NO TREND")

            closed_candle_time = time.strftime('%H:%M:%S', time.gmtime(last_closed['epoch']))
            self.bot.log(f"TREND STATUS: [{closed_candle_time}] {trend_str} (IMBA Algo)")

            if not (is_uptrend or is_downtrend):
                self.bot.log(f"SIGNAL STATUS: [{closed_candle_time}] No trade allowed in neutral market.")
                return None

            # 3. Look for matching UT Bot signals
            df = ut_bot(df, a=a, c=c)
            raw_sig = df.iloc[-2]

            # Align signals with trend
            buy_triggered = raw_sig['buy'] and is_uptrend
            sell_triggered = raw_sig['sell'] and is_downtrend

            if buy_triggered or sell_triggered:
                side = 'BUY' if buy_triggered else 'SELL'
                self.bot.log(f"SIGNAL STATUS: [{closed_candle_time}] {side} Signal found in {trend_str}. Verifying with Neural Filter...")

                # ML Filter (VIP Similarity & XGBoost)
                # This model has been trained on thousands of past signals to distinguish VIP winners from losers.
                ml = model_manager.get_model(symbol, strategy_idx)
                if ml:
                    df_ml = ml.filter_signals(df)
                    ml_sig = df_ml.iloc[-2]
                    if ml_sig['buy'] or ml_sig['sell']:
                        self.bot.log(f"NEURAL FILTER: [{closed_candle_time}] [PASSED] VIP Similarity & Probability high. Executing {side} trade.")
                        return 'CALL' if buy_triggered else 'PUT'
                    else:
                        self.bot.log(f"NEURAL FILTER: [{closed_candle_time}] [BLOCKED] Signal resembles historical losses (LOSER pattern).")
                else:
                    self.bot.log(f"NEURAL FILTER: [{closed_candle_time}] [INACTIVE] Executing raw signal.")
                    return 'CALL' if buy_triggered else 'PUT'
            else:
                self.bot.log(f"SIGNAL STATUS: [{closed_candle_time}] No signal found on closed candle.")

        except Exception as e:
            self.bot.log(f"StrategyHandler Error: {e}")

        return None
