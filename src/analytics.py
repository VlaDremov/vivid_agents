import pandas as pd
from typing import Dict, Any
from logger_config import get_analytics_logger, log_info, log_success, log_debug

# * Initialize logger
logger = get_analytics_logger()


def active_users_by_region(
    df_users: pd.DataFrame, start_date: str, end_date: str, region_col: str = "region"
) -> Dict[str, Any]:
    """
    Returns the number of active users in a given period, grouped by region.
    
    A user is considered active if:
    1. They have is_active = True, AND
    2. Either their last_login_date OR registration_date falls within the specified period
    
    Args:
        df_users: DataFrame with user data
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        region_col: Column name for region grouping (default: "region")
        
    Returns:
        Dictionary mapping region names to active user counts
    """
    log_info(
        logger, f"Calculating active users by region from {start_date} to {end_date}"
    )
    log_debug(logger, f"Input DataFrame shape: {df_users.shape}")
    
    # * Make a copy to avoid modifying original data
    df = df_users.copy()
    
    # * Convert date columns to datetime
    log_debug(logger, "Converting date columns to datetime format")
    df["last_login_date"] = pd.to_datetime(df["last_login_date"])
    df["registration_date"] = pd.to_datetime(df["registration_date"])
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    
    # * Filter for active users only
    log_debug(logger, "Filtering for active users (is_active=True)")
    initial_count = len(df)
    df = df[df["is_active"]]
    active_count = len(df)
    log_debug(logger, f"Active users: {active_count}/{initial_count}")
    
    # * Filter for users who were active in the given period
    log_debug(logger, "Applying date range filter")
    mask = ((df["last_login_date"] >= start) & (df["last_login_date"] <= end)) | (
        (df["registration_date"] >= start) & (df["registration_date"] <= end)
    )
    df = df[mask]
    period_active_count = len(df)
    log_debug(logger, f"Users active in period: {period_active_count}")
    
    # * Group by region and count unique users
    log_debug(logger, f"Grouping by {region_col} and counting unique users")
    result = df.groupby(region_col)["user_id"].nunique().to_dict()
    
    total_users = sum(result.values())
    region_count = len(result)
    log_success(
        logger,
        f"Calculation completed: {total_users} active users across {region_count} regions",
    )
    log_debug(logger, f"Results by region: {result}")
    
    return result


def registration_to_purchase_conversion_rate(
    df_users: pd.DataFrame,
    df_orders: pd.DataFrame,
    start_date: str,
    end_date: str,
    conversion_window_days: int = 30,
) -> Dict[str, float]:
    """
    Calculate the conversion rate from registration to first purchase within a timeframe.
    
    Conversion rate = (Users who registered and made first purchase) / (Total users registered) * 100
    
    Args:
        df_users: DataFrame with user data (must include user_id, registration_date)
        df_orders: DataFrame with order data (must include user_id, order_date)
        start_date: Start date for registration period in YYYY-MM-DD format
        end_date: End date for registration period in YYYY-MM-DD format
        conversion_window_days: Days after registration to consider for conversion (default: 30)
        
    Returns:
        Dictionary with conversion metrics:
        - 'conversion_rate': Overall conversion rate percentage
        - 'registered_users': Total users registered in period
        - 'converted_users': Users who made first purchase within conversion window
        - 'average_days_to_purchase': Average days from registration to first purchase
    """
    log_info(
        logger,
        f"Calculating registration to purchase conversion rate from {start_date} to {end_date} "
        f"(conversion window: {conversion_window_days} days)",
    )
    log_debug(
        logger, f"Input DataFrames - Users: {df_users.shape}, Orders: {df_orders.shape}"
    )
    
    # * Make copies to avoid modifying original data
    users = df_users.copy()
    orders = df_orders.copy()
    
    # * Convert date columns to datetime
    log_debug(logger, "Converting date columns to datetime format")
    users["registration_date"] = pd.to_datetime(users["registration_date"])
    orders["order_date"] = pd.to_datetime(orders["order_date"])
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    
    # * Filter users who registered in the specified period
    log_debug(logger, "Filtering users by registration date")
    registered_mask = (users["registration_date"] >= start) & (
        users["registration_date"] <= end
    )
    registered_users = users[registered_mask].copy()
    total_registered = len(registered_users)
    log_debug(logger, f"Users registered in period: {total_registered}")
    
    if total_registered == 0:
        log_debug(logger, "No users registered in the specified period")
        return {
            "conversion_rate": 0.0,
            "registered_users": 0,
            "converted_users": 0,
            "average_days_to_purchase": 0.0,
        }
    
    # * Find first order for each user
    log_debug(logger, "Finding first order for each user")
    orders_sorted = orders.sort_values(["user_id", "order_date"])
    first_orders = orders_sorted.groupby("user_id").first().reset_index()
    log_debug(logger, f"Users with orders: {len(first_orders)}")
    
    # * Merge registered users with their first orders
    log_debug(logger, "Merging registered users with first orders")
    merged = registered_users.merge(
        first_orders[["user_id", "order_date"]], on="user_id", how="left"
    )
    
    # * Calculate days from registration to first purchase
    merged["days_to_purchase"] = (
        merged["order_date"] - merged["registration_date"]
    ).dt.days
    
    # * Filter for conversions within the conversion window
    log_debug(logger, f"Filtering conversions within {conversion_window_days} days")
    # * Create boolean masks for conversion filtering
    has_purchase = merged["days_to_purchase"].notna()
    valid_timeframe = (merged["days_to_purchase"] >= 0) & (
        merged["days_to_purchase"] <= conversion_window_days
    )
    conversion_mask = has_purchase & valid_timeframe
    converted_users = merged[conversion_mask]
    total_converted = len(converted_users)
    
    # * Calculate conversion rate
    conversion_rate = (
        (total_converted / total_registered) * 100 if total_registered > 0 else 0.0
    )

    result = {"conversion_rate": round(conversion_rate, 2)}
    
    log_success(
        logger,
        f"Conversion analysis completed: {result['conversion_rate']}% conversion rate",
    )
    log_debug(logger, f"Detailed results: {result}")
    
    return result


def average_order_check_by_region(
    df_users: pd.DataFrame,
    df_orders: pd.DataFrame,
    start_date: str,
    end_date: str,
    region_col: str = "region"
) -> Dict[str, float]:
    """
    Calculate the average order check (average order value) grouped by region for a given period.
    
    Args:
        df_users: DataFrame with user data (must include user_id, region)
        df_orders: DataFrame with order data (must include user_id, order_date, order_amount)
        start_date: Start date for order period in YYYY-MM-DD format
        end_date: End date for order period in YYYY-MM-DD format
        region_col: Column name for region grouping (default: "region")
        
    Returns:
        Dictionary with regional average order check metrics:
        - Region names as keys with average order values
        - 'total_orders': Total number of orders in period
        - 'overall_average': Overall average order check across all regions
    """
    log_info(
        logger,
        f"Calculating average order check by region from {start_date} to {end_date}"
    )
    log_debug(
        logger, f"Input DataFrames - Users: {df_users.shape}, Orders: {df_orders.shape}"
    )
    
    # * Make copies to avoid modifying original data
    users = df_users.copy()
    orders = df_orders.copy()
    
    # * Convert date columns to datetime
    log_debug(logger, "Converting date columns to datetime format")
    orders["order_date"] = pd.to_datetime(orders["order_date"])
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    
    # * Filter orders by date range
    log_debug(logger, "Filtering orders by date range")
    date_mask = (orders["order_date"] >= start) & (orders["order_date"] <= end)
    filtered_orders = orders[date_mask].copy()
    total_orders = len(filtered_orders)
    log_debug(logger, f"Orders in period: {total_orders}")
    
    if total_orders == 0:
        log_debug(logger, "No orders found in the specified period")
        return {
            "total_orders": 0,
            "overall_average": 0.0
        }
    
    # * Merge orders with user data to get regional information
    log_debug(logger, "Merging orders with user data to get regional information")
    orders_with_region = filtered_orders.merge(
        users[["user_id", region_col]], on="user_id", how="left"
    )
    
    # * Handle orders from users without region data
    orders_with_region[region_col] = orders_with_region[region_col].fillna("Unknown")
    log_debug(logger, f"Orders after regional merge: {len(orders_with_region)}")
    
    # * Calculate average order check by region
    log_debug(logger, "Calculating average order check by region")
    regional_averages = orders_with_region.groupby(region_col)["order_amount"].agg([
        'mean',  # Average order value
        'count'  # Number of orders
    ]).round(2)
    
    # * Convert to dictionary format
    result = {}
    for region, stats in regional_averages.iterrows():
        result[region] = {
            "average_order_check": stats['mean'],
            "order_count": int(stats['count'])
        }
    
    # * Calculate overall statistics
    overall_average = orders_with_region["order_amount"].mean()
    result["_summary"] = {
        "overall_average": round(overall_average, 2),
    }
    
    log_success(
        logger,
        f"Average order check calculation completed: {result['_summary']['overall_average']} "
    )
    log_debug(logger, f"Regional breakdown: {result}")
    
    return result["_summary"]


def users_without_orders_by_region(
    df_users: pd.DataFrame,
    df_orders: pd.DataFrame,
    start_date: str,
    end_date: str,
    region_col: str = "region"
) -> Dict[str, Any]:
    """
    Calculate the number of users who registered but never made any orders, grouped by region.
    
    Args:
        df_users: DataFrame with user data (must include user_id, registration_date, region)
        df_orders: DataFrame with order data (must include user_id)
        start_date: Start date for registration period in YYYY-MM-DD format
        end_date: End date for registration period in YYYY-MM-DD format
        region_col: Column name for region grouping (default: "region")
        
    Returns:
        Dictionary with regional non-purchasing user metrics:
        - Region names as keys with non-purchasing user counts
        - '_summary': Overall statistics including total non-purchasing users
    """
    log_info(
        logger,
        f"Calculating users without orders by region from {start_date} to {end_date}"
    )
    log_debug(
        logger, f"Input DataFrames - Users: {df_users.shape}, Orders: {df_orders.shape}"
    )
    
    # * Make copies to avoid modifying original data
    users = df_users.copy()
    orders = df_orders.copy()
    
    # * Convert date columns to datetime
    log_debug(logger, "Converting date columns to datetime format")
    users["registration_date"] = pd.to_datetime(users["registration_date"])
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    
    # * Filter users who registered in the specified period
    log_debug(logger, "Filtering users by registration date")
    registered_mask = (users["registration_date"] >= start) & (
        users["registration_date"] <= end
    )
    registered_users = users[registered_mask].copy()
    total_registered = len(registered_users)
    log_debug(logger, f"Users registered in period: {total_registered}")
    
    if total_registered == 0:
        log_debug(logger, "No users registered in the specified period")
        return {
            "_summary": {
                "users_without_orders": 0,
            }
        }
    
    # * Get unique user IDs who have made orders
    log_debug(logger, "Identifying users who have made orders")
    users_with_orders = set(orders["user_id"].unique())
    log_debug(logger, f"Users with orders: {len(users_with_orders)}")
    
    # * Identify registered users who have never made orders
    log_debug(logger, "Finding users without orders")
    registered_users["has_orders"] = registered_users["user_id"].isin(users_with_orders)
    users_without_orders = registered_users[~registered_users["has_orders"]].copy()
    total_without_orders = len(users_without_orders)
    log_debug(logger, f"Users without orders: {total_without_orders}")
    
    # * Handle users without region data
    users_without_orders[region_col] = users_without_orders[region_col].fillna("Unknown")
    
    # * Group by region and count users without orders
    log_debug(logger, "Grouping non-purchasing users by region")
    regional_counts = users_without_orders.groupby(region_col)["user_id"].count().to_dict()
    
    # * Calculate non-purchasing rate
    non_purchasing_rate = (total_without_orders / total_registered) * 100 if total_registered > 0 else 0.0
    
    # * Build result dictionary
    result = regional_counts.copy()
    result["_summary"] = {
        "users_without_orders": total_without_orders
    }
    
    log_success(
        logger,
        f"Non-purchasing users analysis completed: {total_without_orders} users ({non_purchasing_rate:.2f}%) "
        f"across {len(regional_counts)} regions"
    )
    log_debug(logger, f"Regional breakdown: {result}")
    
    return result["_summary"]


def top_regions_by_registrations(
    df_users: pd.DataFrame,
    start_date: str,
    end_date: str,
    top_k: int = 5,
    region_col: str = "region"
) -> Dict[str, Any]:
    """
    Find the top K regions by registration count in a given time period.
    
    Args:
        df_users: DataFrame with user data (must include registration_date, region)
        start_date: Start date for registration period in YYYY-MM-DD format
        end_date: End date for registration period in YYYY-MM-DD format
        top_k: Number of top regions to return (default: 5)
        region_col: Column name for region grouping (default: "region")
        
    Returns:
        Dictionary with top regions ranked by registration count:
        - Ordered list of regions with registration counts
        - Summary statistics including total registrations
    """
    log_info(
        logger,
        f"Finding top {top_k} regions by registrations from {start_date} to {end_date}"
    )
    log_debug(logger, f"Input DataFrame shape: {df_users.shape}")
    
    # * Make a copy to avoid modifying original data
    users = df_users.copy()
    
    # * Convert date columns to datetime
    log_debug(logger, "Converting date columns to datetime format")
    users["registration_date"] = pd.to_datetime(users["registration_date"])
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    
    # * Filter users who registered in the specified period
    log_debug(logger, "Filtering users by registration date")
    registered_mask = (users["registration_date"] >= start) & (
        users["registration_date"] <= end
    )
    registered_users = users[registered_mask].copy()
    total_registered = len(registered_users)
    log_debug(logger, f"Users registered in period: {total_registered}")
    
    if total_registered == 0:
        log_debug(logger, "No users registered in the specified period")
        return {
            "top_regions": [],
            "_summary": {
                "total_registrations": 0,
                "regions_analyzed": 0,
                "top_k_requested": top_k
            }
        }
    
    # * Handle users without region data
    registered_users[region_col] = registered_users[region_col].fillna("Unknown")
    
    # * Group by region and count registrations
    log_debug(logger, f"Grouping registrations by {region_col}")
    regional_counts = registered_users.groupby(region_col)["user_id"].count()
    
    # * Sort regions by registration count (descending) and get top K
    log_debug(logger, f"Sorting regions and selecting top {top_k}")
    top_regions_series = regional_counts.sort_values(ascending=False).head(top_k)
    
    # * Convert to list of dictionaries for better structure
    top_regions_list = []
    for rank, (region, count) in enumerate(top_regions_series.items(), 1):
        percentage = (count / total_registered) * 100
        top_regions_list.append({
            "rank": rank,
            "region": region,
            "registrations": int(count),
            "percentage": round(percentage, 2)
        })
    
    # * Build result dictionary
    result = {
        "top_regions": top_regions_list,
        "_summary": {
            "total_registrations": total_registered,
            "regions_analyzed": len(regional_counts),
            "top_k_requested": top_k,
            "top_k_returned": len(top_regions_list)
        }
    }
    
    log_success(
        logger,
        f"Top regions analysis completed: {len(top_regions_list)} regions identified "
        f"from {total_registered} total registrations across {len(regional_counts)} regions"
    )
    log_debug(logger, f"Top regions: {[r['region'] for r in top_regions_list]}")
    
    return result["top_regions"]


def cancelled_orders_share(
    df_orders: pd.DataFrame,
    start_date: str,
    end_date: str,
    status_col: str = "status"
) -> float:
    """
    Calculate the share (percentage) of cancelled orders in a given time period.
    
    Args:
        df_orders: DataFrame with order data (must include order_date, status)
        start_date: Start date for order period in YYYY-MM-DD format
        end_date: End date for order period in YYYY-MM-DD format
        status_col: Column name for order status (default: "status")
        
    Returns:
        Float representing the percentage of cancelled orders (0-100)
    """
    log_info(
        logger,
        f"Calculating cancelled orders share from {start_date} to {end_date}"
    )
    log_debug(logger, f"Input DataFrame shape: {df_orders.shape}")
    
    # * Make a copy to avoid modifying original data
    orders = df_orders.copy()
    
    # * Convert date columns to datetime
    log_debug(logger, "Converting date columns to datetime format")
    orders["order_date"] = pd.to_datetime(orders["order_date"])
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    
    # * Filter orders by date range
    log_debug(logger, "Filtering orders by date range")
    date_mask = (orders["order_date"] >= start) & (orders["order_date"] <= end)
    filtered_orders = orders[date_mask].copy()
    total_orders = len(filtered_orders)
    log_debug(logger, f"Total orders in period: {total_orders}")
    
    if total_orders == 0:
        log_debug(logger, "No orders found in the specified period")
        return 0.0
    
    # * Count cancelled orders (assuming "cancelled" status)
    log_debug(logger, "Counting cancelled orders")
    cancelled_orders = filtered_orders[filtered_orders[status_col].str.lower() == "cancelled"]
    cancelled_count = len(cancelled_orders)
    log_debug(logger, f"Cancelled orders: {cancelled_count}")
    
    # * Calculate cancellation share
    cancellation_rate = (cancelled_count / total_orders) * 100 if total_orders > 0 else 0.0
    
    log_success(
        logger,
        f"Cancelled orders analysis completed: {cancellation_rate:.2f}% "
        f"({cancelled_count}/{total_orders} orders)"
    )
    
    return round(cancellation_rate, 2)


def customer_lifetime_value(
    df_users: pd.DataFrame,
    df_orders: pd.DataFrame,
    start_date: str,
    end_date: str
) -> float:
    """
    Calculate the average Customer Lifetime Value (CLV) across all customers.
    
    CLV is calculated as the total revenue generated by each customer divided by the number of customers.
    Only considers customers who registered within the specified period.
    
    Args:
        df_users: DataFrame with user data (must include user_id, registration_date)
        df_orders: DataFrame with order data (must include user_id, order_amount)
        start_date: Start date for customer registration period in YYYY-MM-DD format
        end_date: End date for customer registration period in YYYY-MM-DD format
        
    Returns:
        Float representing the average customer lifetime value
    """
    log_info(
        logger,
        f"Calculating customer lifetime value from {start_date} to {end_date}"
    )
    log_debug(
        logger, f"Input DataFrames - Users: {df_users.shape}, Orders: {df_orders.shape}"
    )
    
    # * Make copies to avoid modifying original data
    users = df_users.copy()
    orders = df_orders.copy()
    
    # * Convert date columns to datetime
    log_debug(logger, "Converting date columns to datetime format")
    users["registration_date"] = pd.to_datetime(users["registration_date"])
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    
    # * Filter users who registered in the specified period
    log_debug(logger, "Filtering users by registration date")
    registered_mask = (users["registration_date"] >= start) & (
        users["registration_date"] <= end
    )
    registered_users = users[registered_mask].copy()
    total_customers = len(registered_users)
    log_debug(logger, f"Customers registered in period: {total_customers}")
    
    if total_customers == 0:
        log_debug(logger, "No customers registered in the specified period")
        return 0.0
    
    # * Calculate total revenue per customer
    log_debug(logger, "Calculating total revenue per customer")
    customer_revenue = orders.groupby("user_id")["order_amount"].sum().reset_index()
    customer_revenue.columns = ["user_id", "total_revenue"]
    log_debug(logger, f"Customers with orders: {len(customer_revenue)}")
    
    # * Merge with registered users to get only relevant customers
    log_debug(logger, "Merging customer revenue with registered users")
    registered_customer_revenue = registered_users.merge(
        customer_revenue, on="user_id", how="left"
    )
    
    # * Fill NaN values with 0 for customers with no orders
    registered_customer_revenue["total_revenue"] = registered_customer_revenue["total_revenue"].fillna(0)
    
    # * Calculate average lifetime value
    total_revenue = registered_customer_revenue["total_revenue"].sum()
    average_clv = total_revenue / total_customers if total_customers > 0 else 0.0
    
    # * Log summary statistics
    customers_with_orders = len(registered_customer_revenue[registered_customer_revenue["total_revenue"] > 0])
    customers_without_orders = total_customers - customers_with_orders
    
    log_success(
        logger,
        f"Lifetime value calculation completed: ${average_clv:.2f} average CLV "
        f"(${total_revenue:.2f} total revenue from {total_customers} customers)"
    )
    log_debug(
        logger, 
        f"Customer breakdown: {customers_with_orders} with orders, {customers_without_orders} without orders"
    )
    
    return round(average_clv, 2)


def repeat_customers_percentage(
    df_orders: pd.DataFrame,
    start_date: str,
    end_date: str
) -> float:
    """
    Calculate the percentage of users who made more than 1 order in a given timeframe.
    
    Args:
        df_orders: DataFrame with order data (must include user_id, order_date)
        start_date: Start date for order period in YYYY-MM-DD format
        end_date: End date for order period in YYYY-MM-DD format
        
    Returns:
        Float representing the percentage of users who made multiple orders (0-100)
    """
    log_info(
        logger,
        f"Calculating repeat customers percentage from {start_date} to {end_date}"
    )
    log_debug(logger, f"Input DataFrame shape: {df_orders.shape}")
    
    # * Make a copy to avoid modifying original data
    orders = df_orders.copy()
    
    # * Convert date columns to datetime
    log_debug(logger, "Converting date columns to datetime format")
    orders["order_date"] = pd.to_datetime(orders["order_date"])
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    
    # * Filter orders by date range
    log_debug(logger, "Filtering orders by date range")
    date_mask = (orders["order_date"] >= start) & (orders["order_date"] <= end)
    filtered_orders = orders[date_mask].copy()
    total_orders = len(filtered_orders)
    log_debug(logger, f"Total orders in period: {total_orders}")
    
    if total_orders == 0:
        log_debug(logger, "No orders found in the specified period")
        return 0.0
    
    # * Count orders per user
    log_debug(logger, "Counting orders per user")
    orders_per_user = filtered_orders.groupby("user_id").size()
    total_unique_customers = len(orders_per_user)
    log_debug(logger, f"Total unique customers with orders: {total_unique_customers}")
    
    # * Count users with more than 1 order
    log_debug(logger, "Identifying repeat customers (more than 1 order)")
    repeat_customers = orders_per_user[orders_per_user > 1]
    repeat_customers_count = len(repeat_customers)
    log_debug(logger, f"Repeat customers: {repeat_customers_count}")
    
    # * Calculate repeat customer percentage
    repeat_percentage = (repeat_customers_count / total_unique_customers) * 100 if total_unique_customers > 0 else 0.0
    
    # * Log summary statistics
    single_order_customers = total_unique_customers - repeat_customers_count
    max_orders = orders_per_user.max() if len(orders_per_user) > 0 else 0
    
    log_success(
        logger,
        f"Repeat customers analysis completed: {repeat_percentage:.2f}% "
        f"({repeat_customers_count}/{total_unique_customers} customers made multiple orders)"
    )
    log_debug(
        logger, 
        f"Customer breakdown: {repeat_customers_count} repeat, {single_order_customers} single-order, max orders: {max_orders}"
    )
    
    return round(repeat_percentage, 2)


def registration_dynamic(
    df_users: pd.DataFrame,
    start_date: str,
    end_date: str,
    frequency: str = "D"
) -> Dict[str, Any]:
    """
    Display registration dynamics over a timeframe showing trends and patterns.
    
    Args:
        df_users: DataFrame with user data (must include user_id, registration_date)
        start_date: Start date for analysis period in YYYY-MM-DD format
        end_date: End date for analysis period in YYYY-MM-DD format
        frequency: Time frequency for aggregation ('D' for daily, 'W' for weekly, 'M' for monthly)
        
    Returns:
        Dictionary with registration dynamics including:
        - daily/weekly/monthly registration counts
        - total registrations
        - peak periods
        - growth trends
        - summary statistics
    """
    log_info(
        logger,
        f"Calculating registration dynamics from {start_date} to {end_date} (frequency: {frequency})"
    )
    log_debug(logger, f"Input DataFrame shape: {df_users.shape}")
    
    # * Make a copy to avoid modifying original data
    users = df_users.copy()
    
    # * Convert date columns to datetime
    log_debug(logger, "Converting date columns to datetime format")
    users["registration_date"] = pd.to_datetime(users["registration_date"])
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    
    # * Filter users by registration date
    log_debug(logger, "Filtering users by registration date")
    registered_mask = (users["registration_date"] >= start) & (
        users["registration_date"] <= end
    )
    registered_users = users[registered_mask].copy()
    total_registrations = len(registered_users)
    log_debug(logger, f"Total registrations in period: {total_registrations}")
    
    if total_registrations == 0:
        log_debug(logger, "No registrations found in the specified period")
        return {
            "total_registrations": 0,
            "daily_registrations": {},
            "peak_day": None,
            "lowest_day": None,
            "average_daily": 0.0,
            "growth_trend": "No data",
            "summary": "No registrations found in the specified period"
        }
    
    # * Group registrations by specified frequency
    log_debug(logger, f"Grouping registrations by {frequency} frequency")
    registered_users.set_index("registration_date", inplace=True)
    
    # * Create time series with registration counts
    registration_counts = registered_users.groupby(pd.Grouper(freq=frequency)).size()
    
    # * Convert to dictionary for better structure
    time_series_dict = {}
    for date, count in registration_counts.items():
        if count > 0:  # Only include periods with registrations
            date_str = date.strftime("%Y-%m-%d")
            time_series_dict[date_str] = int(count)
    
    log_debug(logger, f"Registration periods with activity: {len(time_series_dict)}")
    
    # * Calculate summary statistics
    if len(time_series_dict) > 0:
        counts_list = list(time_series_dict.values())
        peak_count = max(counts_list)
        lowest_count = min(counts_list)
        average_count = sum(counts_list) / len(counts_list)
        
        # * Find peak and lowest days
        peak_day = None
        lowest_day = None
        for date, count in time_series_dict.items():
            if count == peak_count:
                peak_day = date
            if count == lowest_count:
                lowest_day = date
        
        # * Calculate growth trend (compare first and last periods)
        dates_sorted = sorted(time_series_dict.keys())
        if len(dates_sorted) >= 2:
            first_period_count = time_series_dict[dates_sorted[0]]
            last_period_count = time_series_dict[dates_sorted[-1]]
            if first_period_count > 0:
                growth_rate = ((last_period_count - first_period_count) / first_period_count) * 100
                if growth_rate > 10:
                    growth_trend = f"Growing (+{growth_rate:.1f}%)"
                elif growth_rate < -10:
                    growth_trend = f"Declining ({growth_rate:.1f}%)"
                else:
                    growth_trend = f"Stable ({growth_rate:.1f}%)"
            else:
                growth_trend = "Growing (from zero)"
        else:
            growth_trend = "Insufficient data"
    else:
        peak_count = 0
        lowest_count = 0
        average_count = 0.0
        peak_day = None
        lowest_day = None
        growth_trend = "No data"
    
    # * Build result dictionary
    frequency_label = {"D": "daily", "W": "weekly", "M": "monthly"}.get(frequency, frequency)
    
    result = {
        "total_registrations": total_registrations,
        f"{frequency_label}_registrations": time_series_dict,
        "peak_day": peak_day,
        "peak_count": peak_count,
        "lowest_day": lowest_day,
        "lowest_count": lowest_count,
        f"average_{frequency_label}": round(average_count, 2),
        "growth_trend": growth_trend,
        "active_periods": len(time_series_dict),
        "summary": f"{total_registrations} total registrations across {len(time_series_dict)} active {frequency_label} periods"
    }
    
    log_success(
        logger,
        f"Registration dynamics analysis completed: {total_registrations} registrations, "
        f"peak: {peak_count} on {peak_day}, trend: {growth_trend}"
    )
    log_debug(logger, f"Time series data points: {len(time_series_dict)}")
    
    return result


def visitors_without_purchase(
    df_users: pd.DataFrame,
    df_orders: pd.DataFrame,
    start_date: str,
    end_date: str
) -> int:
    """
    Calculate the number of users who visited the site but didn't make any purchases over a timeframe.
    
    A visitor without purchase is defined as a user who:
    1. Had some activity (last_login_date within the period), AND
    2. Did not make any orders during the same period
    
    Args:
        df_users: DataFrame with user data (must include user_id, last_login_date)
        df_orders: DataFrame with order data (must include user_id, order_date)
        start_date: Start date for analysis period in YYYY-MM-DD format
        end_date: End date for analysis period in YYYY-MM-DD format
        
    Returns:
        Integer representing the count of visitors who didn't make purchases
    """
    log_info(
        logger,
        f"Calculating visitors without purchase from {start_date} to {end_date}"
    )
    log_debug(
        logger, f"Input DataFrames - Users: {df_users.shape}, Orders: {df_orders.shape}"
    )
    
    # * Make copies to avoid modifying original data
    users = df_users.copy()
    orders = df_orders.copy()
    
    # * Convert date columns to datetime
    log_debug(logger, "Converting date columns to datetime format")
    users["last_login_date"] = pd.to_datetime(users["last_login_date"])
    orders["order_date"] = pd.to_datetime(orders["order_date"])
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    
    # * Filter users who had activity (visited) during the period
    log_debug(logger, "Filtering users who visited during the period")
    visited_mask = (users["last_login_date"] >= start) & (
        users["last_login_date"] <= end
    )
    visitors = users[visited_mask].copy()
    total_visitors = len(visitors)
    log_debug(logger, f"Total visitors in period: {total_visitors}")
    
    if total_visitors == 0:
        log_debug(logger, "No visitors found in the specified period")
        return 0
    
    # * Filter orders made during the same period
    log_debug(logger, "Filtering orders made during the period")
    order_mask = (orders["order_date"] >= start) & (orders["order_date"] <= end)
    period_orders = orders[order_mask].copy()
    log_debug(logger, f"Total orders in period: {len(period_orders)}")
    
    # * Get unique user IDs who made orders during the period
    log_debug(logger, "Identifying users who made purchases during the period")
    users_with_orders = set(period_orders["user_id"].unique())
    users_with_orders_count = len(users_with_orders)
    log_debug(logger, f"Users with orders in period: {users_with_orders_count}")
    
    # * Identify visitors who did not make any orders
    log_debug(logger, "Finding visitors without purchases")
    visitors_without_orders = visitors[~visitors["user_id"].isin(users_with_orders)]
    visitors_without_purchase_count = len(visitors_without_orders)
    
    # * Calculate percentage for context (logged but not returned)
    non_purchase_rate = (visitors_without_purchase_count / total_visitors) * 100 if total_visitors > 0 else 0.0
    
    log_success(
        logger,
        f"Visitors without purchase analysis completed: {visitors_without_purchase_count} visitors "
        f"({non_purchase_rate:.2f}%) out of {total_visitors} total visitors"
    )
    log_debug(
        logger, 
        f"Breakdown: {users_with_orders_count} visitors made purchases, "
        f"{visitors_without_purchase_count} visitors did not make purchases"
    )
    
    return visitors_without_purchase_count
