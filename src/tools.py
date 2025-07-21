import pandas as pd
from langchain_core.tools import tool
from analytics import active_users_by_region
from logger_config import get_tools_logger, log_info, log_success, log_error

# * Initialize logger
logger = get_tools_logger()


@tool
def get_active_users_by_region_tool(
    start_date: str, end_date: str, csv_path: str = "data/raw/users.csv"
) -> dict:
    """
    Get active users by region for a given date range.

    Args:
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        csv_path: Path to the users CSV file

    Returns:
        Dictionary with region names as keys and user counts as values
    """
    log_info(
        logger,
        f"Tool called: get_active_users_by_region_tool({start_date}, {end_date}, {csv_path})",
    )

    try:
        # * Load user data
        log_info(logger, f"Loading user data from {csv_path}")
        df_users = pd.read_csv(csv_path)
        log_info(logger, f"Loaded {len(df_users)} user records")

        # * Calculate active users by region
        result = active_users_by_region(df_users, start_date, end_date)

        log_success(logger, f"Successfully calculated active users by region: {result}")
        return result

    except Exception as e:
        log_error(logger, e, "Error in get_active_users_by_region_tool")
        return {"error": str(e)}


if __name__ == "__main__":
    # * Test the tool
    log_info(logger, "Testing get_active_users_by_region_tool")

    try:
        result = get_active_users_by_region_tool.invoke(
            {"start_date": "2024-06-01", "end_date": "2024-06-10"}
        )
        log_success(logger, f"Test result: {result}")
    except Exception as e:
        log_error(logger, e, "Error testing tool")
