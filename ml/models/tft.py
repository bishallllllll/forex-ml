import numpy as np
import pandas as pd
import torch
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import QuantileLoss
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression

from ml.config import config


def add_time_idx(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["time_idx"] = np.arange(len(df))
    return df


def prepare_tft_data(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame | None = None,
    target_col: str = "close",
) -> tuple[TimeSeriesDataSet, TimeSeriesDataSet | None]:
    mc = config.models
    df = add_time_idx(df_train)
    feature_cols = [c for c in df.columns if c not in [
        "time_idx", "pair", "pair_id", "open", "high", "low", "volume", target_col
    ]]
    training = TimeSeriesDataSet(
        df,
        time_idx="time_idx",
        target=target_col,
        group_ids=["pair_id"],
        max_encoder_length=mc.tft_max_encoder_length,
        max_prediction_length=mc.tft_max_prediction_length,
        static_categoricals=["pair_id"],
        time_varying_known_categoricals=[],
        time_varying_known_reals=["time_idx"],
        time_varying_unknown_reals=feature_cols,
        target_normalizer=GroupNormalizer(groups=["pair_id"]),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
    )
    if df_val is not None:
        val = add_time_idx(df_val)
        validation = TimeSeriesDataSet.from_dataset(training, val, predict=True, stop_randomization=True)
        return training, validation
    return training, None


def train_tft(
    training: TimeSeriesDataSet,
    validation: TimeSeriesDataSet | None = None,
) -> tuple[TemporalFusionTransformer, Trainer]:
    mc = config.models
    train_dl = training.to_dataloader(train=True, batch_size=mc.tft_batch_size, num_workers=0)
    val_dl = validation.to_dataloader(train=False, batch_size=mc.tft_batch_size, num_workers=0) if validation else None

    model = TemporalFusionTransformer.from_dataset(
        training,
        hidden_size=mc.tft_hidden_size,
        attention_head_size=mc.tft_attention_heads,
        loss=QuantileLoss(),
        learning_rate=mc.tft_learning_rate,
        hidden_continuous_size=mc.tft_hidden_size // 2,
        output_size=7,
    )

    callbacks = [EarlyStopping(monitor="val_loss", patience=10, mode="min")] if val_dl else []
    trainer = Trainer(
        max_epochs=mc.tft_max_epochs,
        accelerator="auto",
        callbacks=callbacks,
        enable_progress_bar=False,
        logger=False,
    )
    trainer.fit(model, train_dl, val_dl)
    return model, trainer


def tft_predict_direction(
    model: TemporalFusionTransformer,
    df: pd.DataFrame,
    current_close: pd.Series,
) -> np.ndarray:
    df = add_time_idx(df)
    mc = config.models
    loader = model.to_dataloader(df, batch_size=mc.tft_batch_size, num_workers=0)
    preds = model.predict(loader, mode="raw", return_index=False, return_y=False)
    if isinstance(preds, torch.Tensor):
        preds = preds.numpy()
    pred_close = preds[..., 0].squeeze()
    direction = (pred_close > current_close.values).astype(int)
    return direction, pred_close


def calibrate_tft(
    df_cal: pd.DataFrame,
    model: TemporalFusionTransformer,
    y_cal: np.ndarray,
) -> IsotonicRegression:
    _, pred_close = tft_predict_direction(model, df_cal, df_cal["close"])
    closer = pred_close / df_cal["close"].values
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(closer, y_cal)
    return calibrator


def tft_calibrated_probs(
    df: pd.DataFrame,
    model: TemporalFusionTransformer,
    calibrator: IsotonicRegression,
) -> np.ndarray:
    _, pred_close = tft_predict_direction(model, df, df["close"])
    closer = pred_close / df["close"].values
    return calibrator.predict(closer)
