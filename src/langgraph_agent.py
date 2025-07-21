import pandas as pd
from pathlib import Path
from typing import Dict, Optional, Union, Any
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from analytics import (
    active_users_by_region, 
    registration_to_purchase_conversion_rate, 
    average_order_check_by_region, 
    users_without_orders_by_region, 
    top_regions_by_registrations, 
    cancelled_orders_share,
    customer_lifetime_value,
    repeat_customers_percentage,
    registration_dynamic,
    visitors_without_purchase
)
from dotenv import load_dotenv
import os
from logger_config import (
    get_langgraph_logger,
    log_success,
    log_error,
    log_info,
    log_warning,
)

# * Load environment variables from .env file
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# * Initialize logger
logger = get_langgraph_logger()

# * Debug: Check if API key is loaded
if openai_api_key:
    log_info(
        logger, f"API Key loaded successfully (starts with: {openai_api_key[:7]}...)"
    )
else:
    log_error(
        logger, Exception("OPENAI_API_KEY not found"), "Environment variable missing"
    )
    log_info(logger, "Available environment variables:")
    for key in os.environ.keys():
        if "OPENAI" in key.upper():
            log_info(logger, f"  - {key}")


@tool
def calculate_active_users_by_region(
    start_date: str, end_date: str, csv_path: Optional[str] = None
) -> Union[Dict[str, int], Dict[str, str]]:
    """
    Calculate active users by region for a given date range from CSV data.
    
    Args:
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format  
        csv_path: Optional path to users CSV file. Defaults to data/raw/users.csv
        
    Returns:
        Dictionary mapping region names to active user counts, or error dict
    """
    log_info(
        logger,
        f"Tool called with: start_date={start_date}, end_date={end_date}, csv_path={csv_path}",
    )
    
    try:
        # * Default CSV path if not provided
        if csv_path is None:
            csv_path = "data/raw/users.csv"
            
        log_info(logger, f"Looking for CSV file at: {Path(csv_path).absolute()}")
        
        # * Check if file exists
        if not Path(csv_path).exists():
            log_error(
                logger,
                FileNotFoundError(f"CSV file not found at {csv_path}"),
                "File validation",
            )
            log_info(logger, f"Current working directory: {Path.cwd()}")
            log_info(logger, "Files in data/raw/ (if directory exists):")
            data_raw_path = Path("data/raw")
            if data_raw_path.exists():
                for file in data_raw_path.iterdir():
                    log_info(logger, f"  - {file.name}")
            else:
                log_warning(logger, "Directory data/raw/ does not exist")
                
            # * Try to generate dummy data
            log_info(logger, "Attempting to generate dummy data...")
            try:
                from data.make_dummies import make_dummy_csvs
                make_dummy_csvs()
                log_success(logger, "Dummy data generated successfully")
            except Exception as gen_error:
                log_error(logger, gen_error, "Failed to generate dummy data")
                return {
                    "error": f"CSV file not found at {csv_path} and failed to generate dummy data: {str(gen_error)}"
                }
            
        # * Check again after potential generation
        if not Path(csv_path).exists():
            error_msg = f"CSV file still not found at {csv_path} even after attempting to generate dummy data"
            log_error(
                logger, FileNotFoundError(error_msg), "File validation after generation"
            )
            return {"error": error_msg}
            
        file_size = Path(csv_path).stat().st_size
        log_success(logger, f"CSV file found! Size: {file_size} bytes")
            
        # * Load the CSV data
        df_users = pd.read_csv(csv_path)
        log_info(
            logger,
            f"Loaded CSV with {len(df_users)} rows and {len(df_users.columns)} columns",
        )
        log_info(logger, f"Columns: {list(df_users.columns)}")
        
        # * Validate required columns
        required_columns = ["user_id", "region", "is_active", "last_login_date"]
        missing_columns = [
            col for col in required_columns if col not in df_users.columns
        ]
        if missing_columns:
            log_error(
                logger,
                ValueError(f"Missing required columns: {missing_columns}"),
                "Column validation",
            )
            return {"error": f"Missing required columns: {missing_columns}"}
            
        log_success(logger, "All required columns present")
        
        # * Calculate active users by region
        log_info(
            logger,
            f"Calculating active users by region for period {start_date} to {end_date}",
        )
        result = active_users_by_region(df_users, start_date, end_date)
        log_success(logger, f"Calculation completed. Result: {result}")
        
        return result
        
    except Exception as e:
        log_error(logger, e, "Exception in calculate_active_users_by_region")
        return {"error": f"Failed to calculate active users: {str(e)}"}


@tool
def calculate_conversion_rate(
    start_date: str, 
    end_date: str, 
    conversion_window_days: int = 30,
    users_csv_path: Optional[str] = None,
    orders_csv_path: Optional[str] = None,
) -> Union[Dict[str, float], Dict[str, str]]:
    """
    Calculate registration to purchase conversion rate for users registered in a given period.
    
    Args:
        start_date: Start date for registration period in YYYY-MM-DD format
        end_date: End date for registration period in YYYY-MM-DD format
        conversion_window_days: Days after registration to consider for conversion (default: 30)
        users_csv_path: Optional path to users CSV file. Defaults to data/raw/users.csv
        orders_csv_path: Optional path to orders CSV file. Defaults to data/raw/orders.csv
        
    Returns:
        Dictionary with conversion metrics or error dict
    """
    log_info(
        logger,
        f"Tool called: conversion rate from {start_date} to {end_date}, window: {conversion_window_days} days",
    )
    
    try:
        # * Default CSV paths if not provided
        if users_csv_path is None:
            users_csv_path = "data/raw/users.csv"
        if orders_csv_path is None:
            orders_csv_path = "data/raw/orders.csv"
            
        log_info(
            logger,
            f"Looking for CSV files: users={users_csv_path}, orders={orders_csv_path}",
        )
        
        # * Check if files exist
        missing_files = []
        if not Path(users_csv_path).exists():
            missing_files.append(users_csv_path)
        if not Path(orders_csv_path).exists():
            missing_files.append(orders_csv_path)
            
        if missing_files:
            log_warning(logger, f"Missing CSV files: {missing_files}")
            # * Try to generate dummy data
            log_info(logger, "Attempting to generate dummy data...")
            try:
                from data.make_dummies import make_dummy_csvs
                make_dummy_csvs()
                log_success(logger, "Dummy data generated successfully")
            except Exception as gen_error:
                log_error(logger, gen_error, "Failed to generate dummy data")
                return {
                    "error": f"CSV files not found and failed to generate dummy data: {str(gen_error)}"
                }
        
        # * Check again after potential generation
        if not Path(users_csv_path).exists() or not Path(orders_csv_path).exists():
            error_msg = "Required CSV files still not found even after attempting to generate dummy data"
            log_error(
                logger, FileNotFoundError(error_msg), "File validation after generation"
            )
            return {"error": error_msg}
            
        # * Load the CSV data
        df_users = pd.read_csv(users_csv_path)
        df_orders = pd.read_csv(orders_csv_path)
        log_info(
            logger,
            f"Loaded users CSV: {len(df_users)} rows, orders CSV: {len(df_orders)} rows",
        )
        
        # * Validate required columns
        required_user_columns = ["user_id", "registration_date"]
        required_order_columns = ["user_id", "order_date"]
        
        missing_user_columns = [
            col for col in required_user_columns if col not in df_users.columns
        ]
        missing_order_columns = [
            col for col in required_order_columns if col not in df_orders.columns
        ]
        
        if missing_user_columns or missing_order_columns:
            error_msg = f"Missing columns - Users: {missing_user_columns}, Orders: {missing_order_columns}"
            log_error(logger, ValueError(error_msg), "Column validation")
            return {"error": error_msg}
            
        log_success(logger, "All required columns present")
        
        # * Calculate conversion rate
        log_info(logger, "Calculating registration to purchase conversion rate")
        result = registration_to_purchase_conversion_rate(
            df_users, df_orders, start_date, end_date, conversion_window_days
        )
        log_success(logger, f"Conversion rate calculation completed: {result}")
        
        return result
        
    except Exception as e:
        log_error(logger, e, "Exception in calculate_conversion_rate")
        return {"error": f"Failed to calculate conversion rate: {str(e)}"}


@tool
def calculate_average_order_check_by_region(
    start_date: str, 
    end_date: str, 
    users_csv_path: Optional[str] = None,
    orders_csv_path: Optional[str] = None,
) -> Union[Dict[str, float], Dict[str, str]]:
    """
    Calculate average order check (average order value) grouped by region for a given period.
    
    Args:
        start_date: Start date for order period in YYYY-MM-DD format
        end_date: End date for order period in YYYY-MM-DD format
        users_csv_path: Optional path to users CSV file. Defaults to data/raw/users.csv
        orders_csv_path: Optional path to orders CSV file. Defaults to data/raw/orders.csv
        
    Returns:
        Dictionary with regional average order check metrics or error dict
    """
    log_info(
        logger,
        f"Tool called: average order check by region from {start_date} to {end_date}",
    )
    
    try:
        # * Default CSV paths if not provided
        if users_csv_path is None:
            users_csv_path = "data/raw/users.csv"
        if orders_csv_path is None:
            orders_csv_path = "data/raw/orders.csv"
            
        log_info(
            logger,
            f"Looking for CSV files: users={users_csv_path}, orders={orders_csv_path}",
        )
        
        # * Check if files exist
        missing_files = []
        if not Path(users_csv_path).exists():
            missing_files.append(users_csv_path)
        if not Path(orders_csv_path).exists():
            missing_files.append(orders_csv_path)
            
        if missing_files:
            log_warning(logger, f"Missing CSV files: {missing_files}")
            # * Try to generate dummy data
            log_info(logger, "Attempting to generate dummy data...")
            try:
                from data.make_dummies import make_dummy_csvs
                make_dummy_csvs()
                log_success(logger, "Dummy data generated successfully")
            except Exception as gen_error:
                log_error(logger, gen_error, "Failed to generate dummy data")
                return {
                    "error": f"CSV files not found and failed to generate dummy data: {str(gen_error)}"
                }
        
        # * Check again after potential generation
        if not Path(users_csv_path).exists() or not Path(orders_csv_path).exists():
            error_msg = "Required CSV files still not found even after attempting to generate dummy data"
            log_error(
                logger, FileNotFoundError(error_msg), "File validation after generation"
            )
            return {"error": error_msg}
            
        # * Load the CSV data
        df_users = pd.read_csv(users_csv_path)
        df_orders = pd.read_csv(orders_csv_path)
        log_info(
            logger,
            f"Loaded users CSV: {len(df_users)} rows, orders CSV: {len(df_orders)} rows",
        )
        
        # * Validate required columns
        required_user_columns = ["user_id", "region"]
        required_order_columns = ["user_id", "order_date", "order_amount"]
        
        missing_user_columns = [
            col for col in required_user_columns if col not in df_users.columns
        ]
        missing_order_columns = [
            col for col in required_order_columns if col not in df_orders.columns
        ]
        
        if missing_user_columns or missing_order_columns:
            error_msg = f"Missing columns - Users: {missing_user_columns}, Orders: {missing_order_columns}"
            log_error(logger, ValueError(error_msg), "Column validation")
            return {"error": error_msg}
            
        log_success(logger, "All required columns present")
        
        # * Calculate average order check by region
        log_info(logger, "Calculating average order check by region")
        result = average_order_check_by_region(
            df_users, df_orders, start_date, end_date
        )
        log_success(logger, f"Average order check calculation completed: {result}")
        
        return result
        
    except Exception as e:
        log_error(logger, e, "Exception in calculate_average_order_check_by_region")
        return {"error": f"Failed to calculate average order check: {str(e)}"}


@tool
def calculate_users_without_orders_by_region(
    start_date: str, 
    end_date: str, 
    users_csv_path: Optional[str] = None,
    orders_csv_path: Optional[str] = None,
) -> Union[Dict[str, int], Dict[str, str]]:
    """
    Calculate the number of users who registered but never made any orders, grouped by region.
    
    Args:
        start_date: Start date for registration period in YYYY-MM-DD format
        end_date: End date for registration period in YYYY-MM-DD format
        users_csv_path: Optional path to users CSV file. Defaults to data/raw/users.csv
        orders_csv_path: Optional path to orders CSV file. Defaults to data/raw/orders.csv
        
    Returns:
        Dictionary with regional non-purchasing user metrics or error dict
    """
    log_info(
        logger,
        f"Tool called: users without orders by region from {start_date} to {end_date}",
    )
    
    try:
        # * Default CSV paths if not provided
        if users_csv_path is None:
            users_csv_path = "data/raw/users.csv"
        if orders_csv_path is None:
            orders_csv_path = "data/raw/orders.csv"
            
        log_info(
            logger,
            f"Looking for CSV files: users={users_csv_path}, orders={orders_csv_path}",
        )
        
        # * Check if files exist
        missing_files = []
        if not Path(users_csv_path).exists():
            missing_files.append(users_csv_path)
        if not Path(orders_csv_path).exists():
            missing_files.append(orders_csv_path)
            
        if missing_files:
            log_warning(logger, f"Missing CSV files: {missing_files}")
            # * Try to generate dummy data
            log_info(logger, "Attempting to generate dummy data...")
            try:
                from data.make_dummies import make_dummy_csvs
                make_dummy_csvs()
                log_success(logger, "Dummy data generated successfully")
            except Exception as gen_error:
                log_error(logger, gen_error, "Failed to generate dummy data")
                return {
                    "error": f"CSV files not found and failed to generate dummy data: {str(gen_error)}"
                }
        
        # * Check again after potential generation
        if not Path(users_csv_path).exists() or not Path(orders_csv_path).exists():
            error_msg = "Required CSV files still not found even after attempting to generate dummy data"
            log_error(
                logger, FileNotFoundError(error_msg), "File validation after generation"
            )
            return {"error": error_msg}
            
        # * Load the CSV data
        df_users = pd.read_csv(users_csv_path)
        df_orders = pd.read_csv(orders_csv_path)
        log_info(
            logger,
            f"Loaded users CSV: {len(df_users)} rows, orders CSV: {len(df_orders)} rows",
        )
        
        # * Validate required columns
        required_user_columns = ["user_id", "registration_date", "region"]
        required_order_columns = ["user_id"]
        
        missing_user_columns = [
            col for col in required_user_columns if col not in df_users.columns
        ]
        missing_order_columns = [
            col for col in required_order_columns if col not in df_orders.columns
        ]
        
        if missing_user_columns or missing_order_columns:
            error_msg = f"Missing columns - Users: {missing_user_columns}, Orders: {missing_order_columns}"
            log_error(logger, ValueError(error_msg), "Column validation")
            return {"error": error_msg}
            
        log_success(logger, "All required columns present")
        
        # * Calculate users without orders by region
        log_info(logger, "Calculating users without orders by region")
        result = users_without_orders_by_region(
            df_users, df_orders, start_date, end_date
        )
        log_success(logger, f"Non-purchasing users calculation completed: {result}")
        
        return result
        
    except Exception as e:
        log_error(logger, e, "Exception in calculate_users_without_orders_by_region")
        return {"error": f"Failed to calculate users without orders: {str(e)}"}


@tool
def calculate_top_regions_by_registrations(
    start_date: str, 
    end_date: str, 
    top_k: int = 5,
    csv_path: Optional[str] = None,
) -> Union[Dict[str, Any], Dict[str, str]]:
    """
    Find the top K regions by registration count in a given time period.
    
    Args:
        start_date: Start date for registration period in YYYY-MM-DD format
        end_date: End date for registration period in YYYY-MM-DD format
        top_k: Number of top regions to return (default: 5)
        csv_path: Optional path to users CSV file. Defaults to data/raw/users.csv
        
    Returns:
        Dictionary with top regions ranked by registration count or error dict
    """
    log_info(
        logger,
        f"Tool called: top {top_k} regions by registrations from {start_date} to {end_date}",
    )
    
    try:
        # * Default CSV path if not provided
        if csv_path is None:
            csv_path = "data/raw/users.csv"
            
        log_info(logger, f"Looking for CSV file at: {Path(csv_path).absolute()}")
        
        # * Check if file exists
        if not Path(csv_path).exists():
            log_error(
                logger,
                FileNotFoundError(f"CSV file not found at {csv_path}"),
                "File validation",
            )
            # * Try to generate dummy data
            log_info(logger, "Attempting to generate dummy data...")
            try:
                from data.make_dummies import make_dummy_csvs
                make_dummy_csvs()
                log_success(logger, "Dummy data generated successfully")
            except Exception as gen_error:
                log_error(logger, gen_error, "Failed to generate dummy data")
                return {
                    "error": f"CSV file not found at {csv_path} and failed to generate dummy data: {str(gen_error)}"
                }
            
        # * Check again after potential generation
        if not Path(csv_path).exists():
            error_msg = f"CSV file still not found at {csv_path} even after attempting to generate dummy data"
            log_error(
                logger, FileNotFoundError(error_msg), "File validation after generation"
            )
            return {"error": error_msg}
            
        # * Load the CSV data
        df_users = pd.read_csv(csv_path)
        log_info(
            logger,
            f"Loaded CSV with {len(df_users)} rows and {len(df_users.columns)} columns",
        )
        
        # * Validate required columns
        required_columns = ["user_id", "registration_date", "region"]
        missing_columns = [
            col for col in required_columns if col not in df_users.columns
        ]
        if missing_columns:
            log_error(
                logger,
                ValueError(f"Missing required columns: {missing_columns}"),
                "Column validation",
            )
            return {"error": f"Missing required columns: {missing_columns}"}
            
        log_success(logger, "All required columns present")
        
        # * Calculate top regions by registrations
        log_info(logger, f"Finding top {top_k} regions by registration count")
        result = top_regions_by_registrations(
            df_users, start_date, end_date, top_k
        )
        log_success(logger, f"Top regions calculation completed: {result}")
        
        return result
        
    except Exception as e:
        log_error(logger, e, "Exception in calculate_top_regions_by_registrations")
        return {"error": f"Failed to calculate top regions: {str(e)}"}


@tool
def calculate_cancelled_orders_share(
    start_date: str, 
    end_date: str, 
    csv_path: Optional[str] = None,
) -> Union[float, Dict[str, str]]:
    """
    Calculate the share (percentage) of cancelled orders in a given time period.
    
    Args:
        start_date: Start date for order period in YYYY-MM-DD format
        end_date: End date for order period in YYYY-MM-DD format
        csv_path: Optional path to orders CSV file. Defaults to data/raw/orders.csv
        
    Returns:
        Float representing the percentage of cancelled orders (0-100) or error dict
    """
    log_info(
        logger,
        f"Tool called: cancelled orders share from {start_date} to {end_date}",
    )
    
    try:
        # * Default CSV path if not provided
        if csv_path is None:
            csv_path = "data/raw/orders.csv"
            
        log_info(logger, f"Looking for CSV file at: {Path(csv_path).absolute()}")
        
        # * Check if file exists
        if not Path(csv_path).exists():
            log_error(
                logger,
                FileNotFoundError(f"CSV file not found at {csv_path}"),
                "File validation",
            )
            # * Try to generate dummy data
            log_info(logger, "Attempting to generate dummy data...")
            try:
                from data.make_dummies import make_dummy_csvs
                make_dummy_csvs()
                log_success(logger, "Dummy data generated successfully")
            except Exception as gen_error:
                log_error(logger, gen_error, "Failed to generate dummy data")
                return {
                    "error": f"CSV file not found at {csv_path} and failed to generate dummy data: {str(gen_error)}"
                }
            
        # * Check again after potential generation
        if not Path(csv_path).exists():
            error_msg = f"CSV file still not found at {csv_path} even after attempting to generate dummy data"
            log_error(
                logger, FileNotFoundError(error_msg), "File validation after generation"
            )
            return {"error": error_msg}
            
        # * Load the CSV data
        df_orders = pd.read_csv(csv_path)
        log_info(
            logger,
            f"Loaded CSV with {len(df_orders)} rows and {len(df_orders.columns)} columns",
        )
        
        # * Validate required columns
        required_columns = ["order_date", "status"]
        missing_columns = [
            col for col in required_columns if col not in df_orders.columns
        ]
        if missing_columns:
            log_error(
                logger,
                ValueError(f"Missing required columns: {missing_columns}"),
                "Column validation",
            )
            return {"error": f"Missing required columns: {missing_columns}"}
            
        log_success(logger, "All required columns present")
        
        # * Calculate cancelled orders share
        log_info(logger, "Calculating cancelled orders share")
        result = cancelled_orders_share(df_orders, start_date, end_date)
        log_success(logger, f"Cancelled orders share calculation completed: {result}%")
        
        return result
        
    except Exception as e:
        log_error(logger, e, "Exception in calculate_cancelled_orders_share")
        return {"error": f"Failed to calculate cancelled orders share: {str(e)}"}


@tool
def calculate_customer_lifetime_value(
    start_date: str, 
    end_date: str, 
    users_csv_path: Optional[str] = None,
    orders_csv_path: Optional[str] = None,
) -> Union[float, Dict[str, str]]:
    """
    Calculate the Customer Lifetime Value (CLV) for users who made orders in a given period.
    
    Args:
        start_date: Start date for order period in YYYY-MM-DD format
        end_date: End date for order period in YYYY-MM-DD format
        users_csv_path: Optional path to users CSV file. Defaults to data/raw/users.csv
        orders_csv_path: Optional path to orders CSV file. Defaults to data/raw/orders.csv
        
    Returns:
        Dictionary with CLV metrics or error dict
    """
    log_info(
        logger,
        f"Tool called: customer lifetime value from {start_date} to {end_date}",
    )
    
    try:
        # * Default CSV paths if not provided
        if users_csv_path is None:
            users_csv_path = "data/raw/users.csv"
        if orders_csv_path is None:
            orders_csv_path = "data/raw/orders.csv"
            
        log_info(
            logger,
            f"Looking for CSV files: users={users_csv_path}, orders={orders_csv_path}",
        )
        
        # * Check if files exist
        missing_files = []
        if not Path(users_csv_path).exists():
            missing_files.append(users_csv_path)
        if not Path(orders_csv_path).exists():
            missing_files.append(orders_csv_path)
            
        if missing_files:
            log_warning(logger, f"Missing CSV files: {missing_files}")
            # * Try to generate dummy data
            log_info(logger, "Attempting to generate dummy data...")
            try:
                from data.make_dummies import make_dummy_csvs
                make_dummy_csvs()
                log_success(logger, "Dummy data generated successfully")
            except Exception as gen_error:
                log_error(logger, gen_error, "Failed to generate dummy data")
                return {
                    "error": f"CSV files not found and failed to generate dummy data: {str(gen_error)}"
                }
        
        # * Check again after potential generation
        if not Path(users_csv_path).exists() or not Path(orders_csv_path).exists():
            error_msg = "Required CSV files still not found even after attempting to generate dummy data"
            log_error(
                logger, FileNotFoundError(error_msg), "File validation after generation"
            )
            return {"error": error_msg}
            
        # * Load the CSV data
        df_users = pd.read_csv(users_csv_path)
        df_orders = pd.read_csv(orders_csv_path)
        log_info(
            logger,
            f"Loaded users CSV: {len(df_users)} rows, orders CSV: {len(df_orders)} rows",
        )
        
        # * Validate required columns
        required_user_columns = ["user_id", "registration_date"]
        required_order_columns = ["user_id", "order_date", "order_amount"]
        
        missing_user_columns = [
            col for col in required_user_columns if col not in df_users.columns
        ]
        missing_order_columns = [
            col for col in required_order_columns if col not in df_orders.columns
        ]
        
        if missing_user_columns or missing_order_columns:
            error_msg = f"Missing columns - Users: {missing_user_columns}, Orders: {missing_order_columns}"
            log_error(logger, ValueError(error_msg), "Column validation")
            return {"error": error_msg}
            
        log_success(logger, "All required columns present")
        
        # * Calculate customer lifetime value
        log_info(logger, "Calculating customer lifetime value")
        result = customer_lifetime_value(
            df_users, df_orders, start_date, end_date
        )
        log_success(logger, f"Customer lifetime value calculation completed: {result}")
        
        return result
        
    except Exception as e:
        log_error(logger, e, "Exception in calculate_customer_lifetime_value")
        return {"error": f"Failed to calculate customer lifetime value: {str(e)}"}


@tool
def calculate_repeat_customers_percentage(
    start_date: str, 
    end_date: str, 
    csv_path: Optional[str] = None,
) -> Union[float, Dict[str, str]]:
    """
    Calculate the percentage of users who made repeat purchases (orders) in a given time period.
    
    Args:
        start_date: Start date for order period in YYYY-MM-DD format
        end_date: End date for order period in YYYY-MM-DD format
        csv_path: Optional path to orders CSV file. Defaults to data/raw/orders.csv
        
    Returns:
        Float representing the percentage of repeat customers (0-100) or error dict
    """
    log_info(
        logger,
        f"Tool called: repeat customers percentage from {start_date} to {end_date}",
    )
    
    try:
        # * Default CSV path if not provided
        if csv_path is None:
            csv_path = "data/raw/orders.csv"
            
        log_info(logger, f"Looking for CSV file at: {Path(csv_path).absolute()}")
        
        # * Check if file exists
        if not Path(csv_path).exists():
            log_error(
                logger,
                FileNotFoundError(f"CSV file not found at {csv_path}"),
                "File validation",
            )
            # * Try to generate dummy data
            log_info(logger, "Attempting to generate dummy data...")
            try:
                from data.make_dummies import make_dummy_csvs
                make_dummy_csvs()
                log_success(logger, "Dummy data generated successfully")
            except Exception as gen_error:
                log_error(logger, gen_error, "Failed to generate dummy data")
                return {
                    "error": f"CSV file not found at {csv_path} and failed to generate dummy data: {str(gen_error)}"
                }
            
        # * Check again after potential generation
        if not Path(csv_path).exists():
            error_msg = f"CSV file still not found at {csv_path} even after attempting to generate dummy data"
            log_error(
                logger, FileNotFoundError(error_msg), "File validation after generation"
            )
            return {"error": error_msg}
            
        # * Load the CSV data
        df_orders = pd.read_csv(csv_path)
        log_info(
            logger,
            f"Loaded CSV with {len(df_orders)} rows and {len(df_orders.columns)} columns",
        )
        
        # * Validate required columns
        required_columns = ["user_id", "order_date"]
        missing_columns = [
            col for col in required_columns if col not in df_orders.columns
        ]
        if missing_columns:
            log_error(
                logger,
                ValueError(f"Missing required columns: {missing_columns}"),
                "Column validation",
            )
            return {"error": f"Missing required columns: {missing_columns}"}
            
        log_success(logger, "All required columns present")
        
        # * Calculate repeat customers percentage
        log_info(logger, "Calculating repeat customers percentage")
        result = repeat_customers_percentage(df_orders, start_date, end_date)
        log_success(logger, f"Repeat customers percentage calculation completed: {result}%")
        
        return result
        
    except Exception as e:
        log_error(logger, e, "Exception in calculate_repeat_customers_percentage")
        return {"error": f"Failed to calculate repeat customers percentage: {str(e)}"}


@tool
def calculate_registration_dynamic(
    start_date: str, 
    end_date: str, 
    users_csv_path: Optional[str] = None,
) -> Union[Dict[str, Any], Dict[str, str]]:
    """
    Calculate registration dynamics showing trends and patterns over time.
    
    Args:
        start_date: Start date for registration period in YYYY-MM-DD format
        end_date: End date for registration period in YYYY-MM-DD format
        users_csv_path: Optional path to users CSV file. Defaults to data/raw/users.csv
        
    Returns:
        Dictionary with registration dynamics including time series, peaks, trends, or error dict
    """
    log_info(
        logger,
        f"Tool called: registration dynamic from {start_date} to {end_date}",
    )
    
    try:
        # * Default CSV path if not provided
        if users_csv_path is None:
            users_csv_path = "data/raw/users.csv"
            
        log_info(logger, f"Looking for CSV file at: {Path(users_csv_path).absolute()}")
        
        # * Check if file exists
        if not Path(users_csv_path).exists():
            log_error(
                logger,
                FileNotFoundError(f"CSV file not found at {users_csv_path}"),
                "File validation",
            )
            # * Try to generate dummy data
            log_info(logger, "Attempting to generate dummy data...")
            try:
                from data.make_dummies import make_dummy_csvs
                make_dummy_csvs()
                log_success(logger, "Dummy data generated successfully")
            except Exception as gen_error:
                log_error(logger, gen_error, "Failed to generate dummy data")
                return {
                    "error": f"CSV file not found at {users_csv_path} and failed to generate dummy data: {str(gen_error)}"
                }
            
        # * Check again after potential generation
        if not Path(users_csv_path).exists():
            error_msg = f"CSV file still not found at {users_csv_path} even after attempting to generate dummy data"
            log_error(
                logger, FileNotFoundError(error_msg), "File validation after generation"
            )
            return {"error": error_msg}
            
        # * Load the CSV data
        df_users = pd.read_csv(users_csv_path)
        log_info(
            logger,
            f"Loaded CSV with {len(df_users)} rows and {len(df_users.columns)} columns",
        )
        
        # * Validate required columns
        required_columns = ["user_id", "registration_date"]
        missing_columns = [
            col for col in required_columns if col not in df_users.columns
        ]
        if missing_columns:
            log_error(
                logger,
                ValueError(f"Missing required columns: {missing_columns}"),
                "Column validation",
            )
            return {"error": f"Missing required columns: {missing_columns}"}
            
        log_success(logger, "All required columns present")
        
        # * Calculate registration dynamic
        log_info(logger, "Calculating registration dynamic")
        result = registration_dynamic(
            df_users, start_date, end_date
        )
        log_success(logger, f"Registration dynamic calculation completed: {result}")
        
        return result
        
    except Exception as e:
        log_error(logger, e, "Exception in calculate_registration_dynamic")
        return {"error": f"Failed to calculate registration dynamic: {str(e)}"}


@tool
def calculate_visitors_without_purchase(
    start_date: str, 
    end_date: str, 
    users_csv_path: Optional[str] = None,
    orders_csv_path: Optional[str] = None,
) -> Union[int, Dict[str, str]]:
    """
    Calculate the number of visitors who visited the site but never made any purchases.
    
    Args:
        start_date: Start date for analysis period in YYYY-MM-DD format
        end_date: End date for analysis period in YYYY-MM-DD format
        users_csv_path: Optional path to users CSV file. Defaults to data/raw/users.csv
        orders_csv_path: Optional path to orders CSV file. Defaults to data/raw/orders.csv
        
    Returns:
        Integer count of visitors without purchases or error dict
    """
    log_info(
        logger,
        f"Tool called: visitors without purchase from {start_date} to {end_date}",
    )
    
    try:
        # * Default CSV paths if not provided
        if users_csv_path is None:
            users_csv_path = "data/raw/users.csv"
        if orders_csv_path is None:
            orders_csv_path = "data/raw/orders.csv"
            
        log_info(
            logger,
            f"Looking for CSV files: users={users_csv_path}, orders={orders_csv_path}",
        )
        
        # * Check if files exist
        missing_files = []
        if not Path(users_csv_path).exists():
            missing_files.append(users_csv_path)
        if not Path(orders_csv_path).exists():
            missing_files.append(orders_csv_path)
            
        if missing_files:
            log_warning(logger, f"Missing CSV files: {missing_files}")
            # * Try to generate dummy data
            log_info(logger, "Attempting to generate dummy data...")
            try:
                from data.make_dummies import make_dummy_csvs
                make_dummy_csvs()
                log_success(logger, "Dummy data generated successfully")
            except Exception as gen_error:
                log_error(logger, gen_error, "Failed to generate dummy data")
                return {
                    "error": f"CSV files not found and failed to generate dummy data: {str(gen_error)}"
                }
        
        # * Check again after potential generation
        if not Path(users_csv_path).exists() or not Path(orders_csv_path).exists():
            error_msg = "Required CSV files still not found even after attempting to generate dummy data"
            log_error(
                logger, FileNotFoundError(error_msg), "File validation after generation"
            )
            return {"error": error_msg}
            
        # * Load the CSV data
        df_users = pd.read_csv(users_csv_path)
        df_orders = pd.read_csv(orders_csv_path)
        log_info(
            logger,
            f"Loaded users CSV: {len(df_users)} rows, orders CSV: {len(df_orders)} rows",
        )
                 
        # * Validate required columns
        required_user_columns = ["user_id", "last_login_date"]
        required_order_columns = ["user_id", "order_date"]
        
        missing_user_columns = [
            col for col in required_user_columns if col not in df_users.columns
        ]
        missing_order_columns = [
            col for col in required_order_columns if col not in df_orders.columns
        ]
        
        if missing_user_columns or missing_order_columns:
            error_msg = f"Missing columns - Users: {missing_user_columns}, Orders: {missing_order_columns}"
            log_error(logger, ValueError(error_msg), "Column validation")
            return {"error": error_msg}
            
        log_success(logger, "All required columns present")
        
        # * Calculate visitors without purchase
        log_info(logger, "Calculating visitors without purchase")
        result = visitors_without_purchase(
            df_users, df_orders, start_date, end_date
        )
        log_success(logger, f"Visitors without purchase calculation completed: {result}")
        
        return result
        
    except Exception as e:
        log_error(logger, e, "Exception in calculate_visitors_without_purchase")
        return {"error": f"Failed to calculate visitors without purchase: {str(e)}"}


def create_analytics_agent(model_name: str = "gpt-4o-mini"):
    """
    Create a LangGraph agent with analytics capabilities.
    
    Args:
        model_name: OpenAI model to use
        
    Returns:
        LangGraph agent configured with analytics tools
    """
    log_info(logger, f"Creating analytics agent with model: {model_name}")
    
    # * Check if API key is available
    if not openai_api_key:
        error_msg = (
            "OpenAI API key not found. Please set OPENAI_API_KEY environment variable."
        )
        log_error(logger, ValueError(error_msg), "API key validation")
        raise ValueError(error_msg)
    
    # * Initialize the model (will automatically use OPENAI_API_KEY from environment)
    log_info(logger, "Initializing ChatOpenAI model...")
    model = ChatOpenAI(model=model_name, temperature=0.7)
    
    # * Create the agent with tools
    log_info(logger, "Creating react agent with analytics tools...")
    agent = create_react_agent(
        model=model,
        tools=[
            calculate_active_users_by_region, 
            calculate_conversion_rate, 
            calculate_average_order_check_by_region,
            calculate_users_without_orders_by_region,
            calculate_top_regions_by_registrations,
            calculate_cancelled_orders_share,
            calculate_customer_lifetime_value,
            calculate_repeat_customers_percentage,
            calculate_registration_dynamic,
            calculate_visitors_without_purchase
        ],
        prompt="""You are a data analytics assistant. You can help analyze user data 
        and calculate various metrics including:
        
        1. Active users by region for specific time periods
        2. Registration to purchase conversion rates
        3. Average order check (average order value) by region
        4. Users who registered but never made orders (non-purchasing users) by region
        5. Top K regions by registration count
        6. Share of cancelled orders (cancellation rate)
        7. Customer Lifetime Value (CLV) for users who made orders
        8. Percentage of repeat customers (users who made multiple orders)
        9. Registration dynamics over time (trends, peaks, growth patterns)
        10. Visitors who registered but never made any orders
         
         When users ask about:
         - Active users or regional analytics: use calculate_active_users_by_region
         - Conversion rates, registration to purchase, or user conversion: use calculate_conversion_rate
         - Average order value, order check, or spending by region: use calculate_average_order_check_by_region
         - Non-purchasing users, users without orders, or lost opportunities: use calculate_users_without_orders_by_region
         - Top regions, most popular regions, or regional rankings: use calculate_top_regions_by_registrations
         - Cancelled orders, cancellation rate, or order failures: use calculate_cancelled_orders_share
         - Customer Lifetime Value (CLV or LTV): use calculate_customer_lifetime_value
         - Repeat customers, multiple orders, or returning users: use calculate_repeat_customers_percentage
         - Registration trends, dynamics, growth patterns, or time series: use calculate_registration_dynamic
         - Visitors without purchase: use calculate_visitors_without_purchase
        
        Always provide clear and concise explanations of the metrics. They should be 15 words or less.""",
    )
    
    log_success(logger, "Analytics agent created successfully")
    return agent
