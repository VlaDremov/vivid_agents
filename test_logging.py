#!/usr/bin/env python3
"""
Test script to demonstrate the centralized logging system.
"""

import sys
import pandas as pd
from pathlib import Path

# * Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

# * Import after path modification
try:
    from logger_config import (
        get_logger, 
        log_info, 
        log_success, 
        log_warning, 
        log_error, 
        log_debug
    )
except ImportError as e:
    print(f"Failed to import logger_config: {e}")
    sys.exit(1)


def main():
    """Test the logging system."""
    logger = get_logger(__name__)
    
    log_info(logger, "Starting logging system test")
    log_info(logger, "=" * 50)
    
    try:
        # * Test data generation with logging
        log_info(logger, "Testing data generation module...")
        from src.data.make_dummies import make_dummy_csvs
        make_dummy_csvs()
        
        # * Test analytics with logging
        log_info(logger, "Testing analytics module...")
        from src.analytics import active_users_by_region
        df_users = pd.read_csv("data/raw/users.csv")
        result = active_users_by_region(df_users, "2024-06-01", "2024-06-10")
        
        log_success(logger, "Analytics test completed successfully")
        log_info(logger, f"Sample result: {dict(list(result.items())[:3])}")
        
        # * Test different log levels
        log_debug(logger, "This is a debug message")
        log_info(logger, "This is an info message")
        log_success(logger, "This is a success message")
        log_warning(logger, "This is a warning message")
        
        # * Test error logging
        try:
            raise ValueError("This is a test error for logging demonstration")
        except Exception as e:
            log_error(logger, e, "Testing error logging")
        
        log_success(logger, "All logging tests completed successfully!")
        log_info(logger, "Check the logs/ directory for detailed log files")
        
    except Exception as e:
        log_error(logger, e, "Error during logging system test")
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 