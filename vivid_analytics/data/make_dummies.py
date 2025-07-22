import random
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import List

import pandas as pd

# * Try to import logger, fallback to print if not available
try:
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from logger_config import get_data_logger, log_success, log_error, log_info

    USE_LOGGER = True
except ImportError:
    USE_LOGGER = False

DEFAULT_ROWS = 150
DEFAULT_SEED = 42
random.seed(DEFAULT_SEED)
DEFAULT_OUT = Path("data/raw")
START_DATE = datetime(2024, 5, 15)
END_DATE = datetime(2024, 7, 15)
CITIES_WEIGHTS = [
    ("Moscow", 0.35),
    ("Saint Petersburg", 0.25),
    ("Novosibirsk", 0.07),
    ("Yekaterinburg", 0.06),
    ("Kazan", 0.05),
    ("Nizhny Novgorod", 0.04),
    ("Chelyabinsk", 0.03),
    ("Samara", 0.03),
    ("Omsk", 0.02),
    ("Rostov-on-Don", 0.02),
    ("Other", 0.08),
]

ORDER_STATUSES = ["created", "paid", "delivered", "returned", "cancelled"]

# * Initialize logger if available
if USE_LOGGER:
    logger = get_data_logger()

# * Fallback logging functions


def log_message(level: str, message: str):
    """Fallback logging function."""
    if USE_LOGGER:
        if level == "INFO":
            log_info(logger, message)
        elif level == "SUCCESS":
            log_success(logger, message)
        elif level == "ERROR":
            log_error(logger, Exception(message), "Data generation error")
    else:
        print(f"[{level}] {message}")


# ----- Helper utilities -------------------------------------------------------


def weighted_choice(choices_with_weights: List[tuple[str, float]]) -> str:
    choices, weights = zip(*choices_with_weights)
    return random.choices(choices, weights=weights, k=1)[0]


def random_date(start: datetime, end: datetime) -> datetime:
    delta = end - start
    return start + timedelta(seconds=random.randint(0, int(delta.total_seconds())))


# ----- Core generators --------------------------------------------------------


def gen_users(n: int) -> pd.DataFrame:
    """Generate dummy user data."""
    log_message("INFO", f"Generating {n} user records")

    users = []
    for uid in range(1, n + 1):
        region = weighted_choice(CITIES_WEIGHTS)
        reg_dt = random_date(START_DATE, END_DATE)
        remaining_days = (END_DATE - reg_dt).days
        # Ensure last_login_date is later than registration_date when possible
        offset = random.randint(1, remaining_days) if remaining_days > 0 else 0
        login_dt = (reg_dt + timedelta(days=offset)).date().isoformat()
        users.append(
            {
                "user_id": uid,
                "region": region,
                "registration_date": random_date(START_DATE, END_DATE)
                .date()
                .isoformat(),
                "is_active": random.choice([True, False]),
                "last_login_date": login_dt,
            }
        )

    log_message("SUCCESS", f"Generated {len(users)} user records")
    return pd.DataFrame(users)


def gen_orders(n: int, max_orders_per_user: int = 5) -> pd.DataFrame:
    """Generate dummy order data."""
    log_message(
        "INFO",
        f"Generating orders for {n} users (max {max_orders_per_user} orders per user)",
    )

    orders = []
    order_id = 1
    for user_id in range(1, n + 1):
        num_orders = random.randint(0, max_orders_per_user)
        for _ in range(num_orders):
            orders.append(
                {
                    "order_id": order_id,
                    "user_id": user_id,
                    "order_date": random_date(START_DATE, END_DATE).date().isoformat(),
                    "order_amount": round(random.uniform(500, 5000), 2),
                    "status": random.choice(ORDER_STATUSES),
                }
            )
            order_id += 1

    # Ensure at least one order per user exists for proper conversion tests
    if not orders:
        error_msg = "No orders generated; adjust max_orders_per_user."
        log_message("ERROR", error_msg)
        raise RuntimeError(error_msg)

    log_message("SUCCESS", f"Generated {len(orders)} order records")
    return pd.DataFrame(orders)


# ----- Main I/O ----------------------------------------------------------------


def make_dummy_csvs(out_dir: Path = Path("data/raw")) -> None:
    """Generate and save dummy CSV files."""
    log_message("INFO", f"Starting dummy data generation to {out_dir}")

    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        log_message("INFO", f"Created output directory: {out_dir}")

        users_df = gen_users(DEFAULT_ROWS)
        orders_df = gen_orders(DEFAULT_ROWS)

        users_file = out_dir / "users.csv"
        orders_file = out_dir / "orders.csv"

        users_df.to_csv(users_file, index=False)
        orders_df.to_csv(orders_file, index=False)

        log_message("SUCCESS", f"Wrote {len(users_df):>4} users to {users_file}")
        log_message("SUCCESS", f"Wrote {len(orders_df):>4} orders to {orders_file}")

    except Exception as e:
        log_message("ERROR", f"Failed to generate dummy CSV files: {str(e)}")
        raise
