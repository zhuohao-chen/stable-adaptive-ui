"""Loading and preprocessing of the LSApp dataset.

LSApp (Aliannejadi et al., ACM TOIS 2021) is a public log of sequential app
usage events from 292 participants: https://github.com/aliannejadi/LSApp.
Each row has ``user_id, session_id, timestamp, app_name, event_type`` where
``event_type`` is one of ``Opened, Closed, User Interaction, Broken``.

Pipeline:

1. keep ``Opened`` events and merge re-opens of the same app within
   ``merge_seconds`` into one *launch* (the raw log often records several
   open/close pairs within a few seconds);
2. per user, split the observed days chronologically into train and test;
3. pick the user's ``n_candidates`` most-launched apps in the training part as
   the shortcut candidates;
4. estimate ``V[p, a]`` = mean number of launches of app ``a`` in
   time-of-day period ``p`` per training day (a frequency-based predictor);
5. keep, for every test day and period, the actual launch counts so that the
   chosen layouts can be evaluated offline.
"""

from __future__ import annotations

import tarfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

LSAPP_URL = "https://github.com/aliannejadi/LSApp/raw/main/lsapp.tsv.gz"


def download_lsapp(dest: str | Path = "data/lsapp.tsv") -> Path:
    """Download LSApp from GitHub and extract ``lsapp.tsv`` to ``dest``."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    archive = dest.with_suffix(".tsv.gz")
    urllib.request.urlretrieve(LSAPP_URL, archive)  # noqa: S310 - fixed https URL
    with tarfile.open(archive, "r:gz") as tar:
        member = next(m for m in tar.getmembers() if m.name.endswith("lsapp.tsv"))
        with tar.extractfile(member) as src, open(dest, "wb") as out:
            out.write(src.read())
    archive.unlink()
    return dest


def load_launches(path: str | Path, merge_seconds: float = 60.0) -> pd.DataFrame:
    """Read LSApp and return one row per app launch.

    Args:
        path: Path to ``lsapp.tsv``.
        merge_seconds: Consecutive ``Opened`` events of the same app by the
            same user closer than this are treated as a single launch.

    Returns:
        DataFrame with columns ``user_id``, ``timestamp`` (datetime) and
        ``app`` sorted by user and time.
    """
    raw = pd.read_csv(path, sep="\t", usecols=["user_id", "timestamp", "app_name", "event_type"])
    return launches_from_events(raw, merge_seconds)


def launches_from_events(events: pd.DataFrame, merge_seconds: float = 60.0) -> pd.DataFrame:
    """Convert raw LSApp events into de-duplicated launches (see :func:`load_launches`)."""
    opened = events.loc[events["event_type"] == "Opened", ["user_id", "timestamp", "app_name"]]
    opened = opened.rename(columns={"app_name": "app"})
    opened = opened.assign(timestamp=pd.to_datetime(opened["timestamp"]))
    opened = opened.sort_values(["user_id", "timestamp"], kind="stable").reset_index(drop=True)
    same_user = opened["user_id"].eq(opened["user_id"].shift())
    same_app = opened["app"].eq(opened["app"].shift())
    gap = opened["timestamp"].diff().dt.total_seconds()
    duplicate = same_user & same_app & (gap <= merge_seconds)
    return opened.loc[~duplicate].reset_index(drop=True)


@dataclass
class UserData:
    """Per-user data prepared for the planning experiments.

    Attributes:
        user_id: LSApp user identifier.
        apps: Names of the ``n`` candidate apps (column order of the arrays).
        period_minutes: Length of one planning period.
        train_counts: Launch counts, shape ``(D_train, P, n)``.
        test_counts: Launch counts of candidate apps, shape ``(D_test, P, n)``.
        test_totals: Launch counts of *all* apps, shape ``(D_test, P)``; the
            denominator for coverage.
    """

    user_id: int
    apps: list[str]
    period_minutes: int
    train_counts: np.ndarray
    test_counts: np.ndarray
    test_totals: np.ndarray

    @property
    def periods_per_day(self) -> int:
        """Number of planning periods per day ``P``."""
        return self.train_counts.shape[1]

    @property
    def values(self) -> np.ndarray:
        """Predicted value ``V[p, a]``: mean launches per training day, shape ``(P, n)``."""
        return self.train_counts.mean(axis=0)


def _daily_counts(
    launches: pd.DataFrame, days: pd.Index, apps: list[str] | None, period_minutes: int
) -> np.ndarray:
    """Return launch counts with shape ``(len(days), P, len(apps))`` (or ``(D, P)``)."""
    periods = 24 * 60 // period_minutes
    day = launches["timestamp"].dt.normalize()
    minute = launches["timestamp"].dt.hour * 60 + launches["timestamp"].dt.minute
    period = (minute // period_minutes).to_numpy()
    day_idx = days.get_indexer(day)
    if apps is None:
        out = np.zeros((len(days), periods))
        np.add.at(out, (day_idx, period), 1)
        return out
    app_idx = pd.Index(apps).get_indexer(launches["app"])
    keep = app_idx >= 0
    out = np.zeros((len(days), periods, len(apps)))
    np.add.at(out, (day_idx[keep], period[keep], app_idx[keep]), 1)
    return out


def build_user_data(
    launches: pd.DataFrame,
    user_id: int,
    n_candidates: int = 10,
    period_minutes: int = 60,
    train_frac: float = 0.7,
) -> UserData:
    """Prepare train/test arrays for one user.

    Days are calendar days between the user's first and last launch (days
    without launches are kept, as zero counts). The first ``train_frac`` of
    them form the training set.

    Raises:
        ValueError: If the user has fewer than two days or fewer than
            ``n_candidates`` distinct apps in the training part.
    """
    if (24 * 60) % period_minutes:
        raise ValueError("period_minutes must divide 1440")
    user = launches.loc[launches["user_id"] == user_id]
    if user.empty:
        raise ValueError(f"unknown user {user_id}")
    first, last = user["timestamp"].min().normalize(), user["timestamp"].max().normalize()
    days = pd.date_range(first, last, freq="D")
    if len(days) < 2:
        raise ValueError(f"user {user_id} has fewer than two days of data")
    n_train = min(max(1, int(round(train_frac * len(days)))), len(days) - 1)
    split = days[n_train]
    train = user.loc[user["timestamp"] < split]
    test = user.loc[user["timestamp"] >= split]
    apps = train["app"].value_counts().index[:n_candidates].tolist()
    if len(apps) < n_candidates:
        raise ValueError(f"user {user_id} has only {len(apps)} distinct apps in training")
    return UserData(
        user_id=int(user_id),
        apps=apps,
        period_minutes=period_minutes,
        train_counts=_daily_counts(train, days[:n_train], apps, period_minutes),
        test_counts=_daily_counts(test, days[n_train:], apps, period_minutes),
        test_totals=_daily_counts(test, days[n_train:], None, period_minutes),
    )


def eligible_users(
    launches: pd.DataFrame, min_days: int = 14, min_apps: int = 10, min_launches: int = 200
) -> list[int]:
    """Users with enough history for the experiments, sorted by id."""
    stats = launches.groupby("user_id").agg(
        first=("timestamp", "min"),
        last=("timestamp", "max"),
        apps=("app", "nunique"),
        launches=("app", "size"),
    )
    span = (stats["last"].dt.normalize() - stats["first"].dt.normalize()).dt.days + 1
    mask = (span >= min_days) & (stats["apps"] >= min_apps) & (stats["launches"] >= min_launches)
    return sorted(int(u) for u in stats.index[mask])


def dataset_summary(launches: pd.DataFrame) -> dict:
    """Descriptive statistics of the launch log, for the report's dataset section."""
    per_user = launches.groupby("user_id").agg(
        launches=("app", "size"),
        apps=("app", "nunique"),
        span_days=("timestamp", lambda s: (s.max().normalize() - s.min().normalize()).days + 1),
    )
    return {
        "users": int(launches["user_id"].nunique()),
        "apps": int(launches["app"].nunique()),
        "launches": int(len(launches)),
        "first": str(launches["timestamp"].min()),
        "last": str(launches["timestamp"].max()),
        "per_user": per_user.describe().round(2).to_dict(),
        "top_apps": launches["app"].value_counts().head(10).to_dict(),
    }
