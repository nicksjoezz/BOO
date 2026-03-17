import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import os
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

class MLFilter:
    def __init__(self):
        # XGBoost parameters optimized for 5m Rise/Fall (v4 configuration)
        self.model = xgb.XGBClassifier(
            objective='binary:logistic',
            n_estimators=200,      # User suggested 200
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=10,   # User suggested 10 to prevent overfitting
            gamma=0.2,
            random_state=42,
            eval_metric='aucpr',   # User suggested precision-recall optimization
            scale_pos_weight=1.0
        )

        # VIP Similarity Engine (v8 Architecture)
        self.scaler = StandardScaler()
        self.vip_nn = NearestNeighbors(n_neighbors=10, metric='euclidean')
        self.loser_nn = NearestNeighbors(n_neighbors=10, metric='euclidean')

        # Cross-Comparison Engine (To find Repetitive Patterns)
        self.all_signals_nn = NearestNeighbors(n_neighbors=20, metric='euclidean')
        self.train_labels = None

        self.vip_patterns = None  # Historical winner features
        self.loser_patterns = None # Historical loser features

        self.is_trained = False
        self.trained_at = None
        self.best_threshold = 0.65

        # Advanced Feature Set following the requested architecture (v7)
        self.indicator_cols = [
            # RSI features
            'rsi7', 'rsi14', 'rsi7_slope', 'rsi7_strength', 'rsi_agreement',
            'bull_divergence', 'bear_divergence', 'rsi7_overbought', 'rsi7_oversold',
            # MACD features
            'macd_hist', 'macd_hist_rising', 'macd_hist_strength', 'macd_above_zero',
            'macd_cross_up', 'macd_cross_down', 'hist_acceleration',
            # Bollinger Bands features
            'bb20_position', 'bb10_position', 'bb20_width', 'volatility_expanding',
            'bb_squeeze', 'above_bb20_mid', 'bb_outside_upper', 'bb_outside_lower',
            # ATR features
            'atr14', 'atr_percentile', 'atr_ratio', 'candle_vs_atr', 'atr_expanding',
            'vol_dead', 'vol_extreme', 'vol_normal',
            # Stochastic features
            'stoch_k_fast', 'stoch_fast_bull', 'stoch_fast_bear', 'stoch_overbought',
            'stoch_oversold', 'stoch_agreement', 'stoch_slope',
            # CCI features
            'cci14', 'cci7', 'cci_cross_up', 'cci_cross_down', 'cci_break_up',
            'cci_break_down', 'cci_agreement', 'cci_extreme_up', 'cci_extreme_dn',
            # EMA Ribbon features
            'ema_bullish_stack', 'ema_bearish_stack', 'price_vs_ema3', 'price_vs_ema8',
            'price_vs_ema20', 'price_vs_ema50', 'ribbon_width', 'ema8_slope',
            'momentum_agrees',
            # Candle Pattern features
            'body_ratio', 'wick_ratio', 'is_bullish', 'candle_streak',
            'bull_engulf', 'bear_engulf', 'is_doji', 'gap',
            # Time & Meta features
            'hour', 'day_of_week', 'session',
            'rsi7_lag_1', 'macd_lag_1', 'close_change_lag_1'
        ]

        # Meta-features from Similarity Engine
        self.similarity_cols = ['vip_score', 'vip_dist', 'loser_dist', 'win_consensus']
        self.feature_cols = self.similarity_cols + self.indicator_cols

    def prepare_indicators(self, df, positional_indices):
        valid_indices = [idx for idx in positional_indices if 0 <= idx < len(df)]
        if not valid_indices:
            return np.zeros((0, len(self.indicator_cols)), dtype=np.float32)

        feature_data = df.iloc[valid_indices][self.indicator_cols].copy()
        feature_data = feature_data.fillna(0).astype(np.float32)
        return feature_data.values

    def get_similarity_meta_features(self, X_scaled):
        """
        Extracts distance-based meta-features for the XGBoost model.
        """
        if self.vip_patterns is None or self.loser_patterns is None:
            return np.zeros((len(X_scaled), 4))

        # 1. Distances to VIPs (repetitive winners)
        dist_vip, _ = self.vip_nn.kneighbors(X_scaled)
        avg_dist_vip = np.mean(dist_vip, axis=1)

        # 2. Distances to Losers (repetitive losses)
        dist_loser, _ = self.loser_nn.kneighbors(X_scaled)
        avg_dist_loser = np.mean(dist_loser, axis=1)

        # 3. VIP Score (Relative similarity)
        vip_score = (avg_dist_loser - avg_dist_vip) / (avg_dist_loser + avg_dist_vip + 1e-9)

        # 4. Win Consensus (Look at 20 most similar historical signals)
        # What percentage of the most similar past trades were winners?
        consensus = np.zeros(len(X_scaled))
        if self.train_labels is not None:
            _, indices = self.all_signals_nn.kneighbors(X_scaled)
            for i, neighbors in enumerate(indices):
                neighbor_labels = self.train_labels[neighbors]
                consensus[i] = np.mean(neighbor_labels)

        return np.column_stack([vip_score, avg_dist_vip, avg_dist_loser, consensus])

    def train(self, df, trades):
        """
        XGBOOST PIPELINE (v3):
        1. Collect all signals with labels (1=Win, 0=Loss)
        2. Filter only signals with high-quality engineered features.
        3. Dynamically balance classes via scale_pos_weight.
        4. Train ensemble of correction trees.
        5. Calibrate threshold using Utility Score for best participation/accuracy balance.
        """
        if trades.empty or len(trades) < 150:
            return False

        df_work = df.reset_index(drop=True)
        epoch_to_idx = {epoch: idx for idx, epoch in enumerate(df_work['epoch'])}

        X_indices, y = [], []
        for _, trade in trades.iterrows():
            entry_epoch = trade['entry_time']
            if entry_epoch not in epoch_to_idx: continue

            entry_idx = epoch_to_idx[entry_epoch]
            signal_idx = entry_idx - 1
            if signal_idx < 0: continue

            # Verify indicators are not NaN
            if pd.isna(df_work.iloc[signal_idx][self.indicator_cols]).any():
                continue

            X_indices.append(signal_idx)
            y.append(1 if trade['win'] else 0)

        if len(y) < 150:
            return False

        X_inds_raw = self.prepare_indicators(df_work, X_indices)
        y = np.array(y)

        # 1. Normalize indicators for similarity engine
        X_inds_scaled = self.scaler.fit_transform(X_inds_raw)

        # 2. Fit Similarity Engine (The "Memory")
        # We store winners and losers separately for distance checks
        # and combined for the consensus engine.
        self.vip_patterns = X_inds_scaled[y == 1]
        self.loser_patterns = X_inds_scaled[y == 0]

        # We need to store the labels in the same order as the combined patterns
        # if we are going to reconstruct the all_signals_nn from saved patterns.
        self.train_labels = np.concatenate([np.ones(len(self.vip_patterns)), np.zeros(len(self.loser_patterns))])

        if len(self.vip_patterns) < 10 or len(self.loser_patterns) < 10:
            return False

        self.vip_nn.fit(self.vip_patterns)
        self.loser_nn.fit(self.loser_patterns)

        # Re-stack to ensure the order matches our reconstructed train_labels
        X_stacked = np.vstack([self.vip_patterns, self.loser_patterns])
        self.all_signals_nn.fit(X_stacked)

        # 3. Generate Meta-Features from Similarity for XGBoost
        # To avoid data leakage, we use a 2-fold stacking approach:
        # Split data, fit NN on one half, generate features for the other half.
        X_meta = np.zeros((len(X_inds_scaled), 4))
        indices = np.arange(len(X_inds_scaled))
        mid = len(X_inds_scaled) // 2

        # Fold 1: Train on second half, predict first half
        X1, X2 = X_inds_scaled[:mid], X_inds_scaled[mid:]
        y1, y2 = y[:mid], y[mid:]

        for fold in [1, 2]:
            train_X = X2 if fold == 1 else X1
            train_y = y2 if fold == 1 else y1
            test_X = X1 if fold == 1 else X2

            # Temporary NNs for the fold
            v_nn = NearestNeighbors(n_neighbors=min(len(train_y[train_y==1]), 10), metric='euclidean')
            l_nn = NearestNeighbors(n_neighbors=min(len(train_y[train_y==0]), 10), metric='euclidean')
            a_nn = NearestNeighbors(n_neighbors=min(len(train_y), 20), metric='euclidean')

            v_pats = train_X[train_y == 1]
            l_pats = train_X[train_y == 0]

            if len(v_pats) > 0: v_nn.fit(v_pats)
            if len(l_pats) > 0: l_nn.fit(l_pats)
            a_nn.fit(train_X)

            # Generate meta-features for the test part of the fold
            d_v, _ = v_nn.kneighbors(test_X)
            d_l, _ = l_nn.kneighbors(test_X)
            av_v = np.mean(d_v, axis=1)
            av_l = np.mean(d_l, axis=1)
            v_sc = (av_l - av_v) / (av_l + av_v + 1e-9)

            _, idxs = a_nn.kneighbors(test_X)
            cons = np.array([np.mean(train_y[ni]) for ni in idxs])

            meta = np.column_stack([v_sc, av_v, av_l, cons])
            if fold == 1: X_meta[:mid] = meta
            else: X_meta[mid:] = meta

        # Combine [Similarity Meta Features] + [Raw Indicators]
        X_combined = np.column_stack([X_meta, X_inds_scaled])

        # 4. Class Imbalance FIX
        num_neg = np.sum(y == 0)
        num_pos = np.sum(y == 1)
        if num_pos > 0:
            self.model.scale_pos_weight = num_neg / num_pos

        # 5. Train XGBoost model on the Combined Feature Set
        self.model.fit(X_combined, y)

        # --- Accuracy Safety Gate ---
        # Evaluate training accuracy as a baseline sanity check
        train_preds = self.model.predict(X_combined)
        train_acc = np.mean(train_preds == y)

        # Explicitly clean up training predictions
        del train_preds

        if train_acc < 0.52:
            # If the model can't even fit the training data better than a coin flip, reject it.
            return False

        # --- Threshold Tuning Phase ---
        probs = self.model.predict_proba(X_combined)[:, 1]
        best_utility = -1
        best_t = 0.62 # Target Floor

        total_signals = len(y)
        # We need enough volume to be profitable, but enough winrate to survive.
        participation_floor = total_signals * 0.40 # Target at least 40% retention

        found_above_floor = False
        for t in np.linspace(0.55, 0.75, 41):
            mask = probs >= t
            subset_y = y[mask]
            num_trades = len(subset_y)

            if num_trades < participation_floor: continue

            found_above_floor = True
            win_rate = np.mean(subset_y)
            # Utility favors higher accuracy above 50%
            utility = (win_rate - 0.5) * np.log1p(num_trades)

            if utility > best_utility:
                best_utility = utility
                best_t = t

        if not found_above_floor:
            # If no threshold hits the volume floor, use the one that gives most volume at >55% accuracy
            best_t = 0.55

        self.best_threshold = best_t
        self.is_trained = True
        self.trained_at = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')
        return True

    def filter_signals(self, df):
        if not self.is_trained: return df

        df_work = df.copy().reset_index(drop=True)
        # Process both sides on the same dataframe copy
        for side in ['buy', 'sell']:
            indices = df_work.index[df_work[side]].tolist()
            if not indices: continue

            X_inds_raw = self.prepare_indicators(df_work, indices)
            if len(X_inds_raw) == 0: continue

            # Scale indicators
            X_inds_scaled = self.scaler.transform(X_inds_raw)

            # 1. Generate Similarity Meta-Features
            X_meta = self.get_similarity_meta_features(X_inds_scaled)

            # 2. Combine for XGBoost
            X_combined = np.column_stack([X_meta, X_inds_scaled])

            # 3. XGBoost Probability (now has similarity baked into its inputs)
            probs = self.model.predict_proba(X_combined)[:, 1]

            # Similarity score is the first column of meta-features
            vip_scores = X_meta[:, 0]

            for i, idx in enumerate(indices):
                # Signal must pass XGBoost threshold
                # The XGBoost model now inherently weights similarity as its primary feature
                # We also use a win_consensus check: more than 40% of similar past signals should be winners
                win_cons = X_meta[i, 3]
                is_vip_pass = vip_scores[i] > -0.1 and win_cons >= 0.4

                if probs[i] < self.best_threshold or not is_vip_pass:
                    df_work.at[idx, side] = False

        df_work.index = df.index
        return df_work

    def save(self, filepath):
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'vip_patterns': self.vip_patterns,
            'loser_patterns': self.loser_patterns,
            'train_labels': self.train_labels,
            'X_scaled': self.vip_patterns if self.vip_patterns is None else np.vstack([self.vip_patterns, self.loser_patterns]), # For consensus
            'trained_at': self.trained_at,
            'features': self.feature_cols,
            'threshold': self.best_threshold,
            'algo': 'vip_similarity_v8'
        }, filepath)

    def load(self, filepath):
        if os.path.exists(filepath):
            try:
                data = joblib.load(filepath)
                if isinstance(data, dict):
                    self.model = data['model']
                    self.scaler = data.get('scaler', StandardScaler())
                    self.vip_patterns = data.get('vip_patterns')
                    self.loser_patterns = data.get('loser_patterns')
                    self.train_labels = data.get('train_labels')
                    self.trained_at = data.get('trained_at')
                    self.best_threshold = data.get('threshold', 0.62)

                    # Re-fit NNs if patterns exist
                    if self.vip_patterns is not None and len(self.vip_patterns) >= 10:
                        self.vip_nn.fit(self.vip_patterns)
                    if self.loser_patterns is not None and len(self.loser_patterns) >= 10:
                        self.loser_nn.fit(self.loser_patterns)

                    # Re-fit the consensus engine
                    X_scaled = data.get('X_scaled')
                    if X_scaled is not None:
                        self.all_signals_nn.fit(X_scaled)
                else:
                    self.model = data
                self.is_trained = True
                return True
            except: pass
        return False
