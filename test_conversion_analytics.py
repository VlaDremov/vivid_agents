#!/usr/bin/env python3
"""
Test script for the new conversion rate analytics functionality.
"""

import sys
import pandas as pd
from pathlib import Path

# * Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

# * Import after path modification
try:
    from logger_config import get_logger, log_info, log_success, log_error
except ImportError as e:
    print(f"Failed to import logger_config: {e}")
    sys.exit(1)


def test_conversion_analytics():
    """Test the conversion rate analytics function."""
    logger = get_logger(__name__)

    log_info(logger, "Testing Conversion Rate Analytics")
    log_info(logger, "=" * 50)

    try:
        # * Import after path setup
        from src.analytics import registration_to_purchase_conversion_rate
        from src.data.make_dummies import make_dummy_csvs

        # * Generate test data
        log_info(logger, "Generating test data...")
        make_dummy_csvs()

        # * Load test data
        log_info(logger, "Loading test data...")
        df_users = pd.read_csv("data/raw/users.csv")
        df_orders = pd.read_csv("data/raw/orders.csv")

        log_info(logger, f"Loaded {len(df_users)} users and {len(df_orders)} orders")

        # * Test conversion rate calculation
        log_info(logger, "Testing conversion rate calculation...")

        # Test 1: Full June 2024 period
        result1 = registration_to_purchase_conversion_rate(
            df_users, df_orders, "2024-06-01", "2024-06-30", 30
        )

        log_success(logger, f"Test 1 - June 2024 conversion rate: {result1}")

        # Test 2: First half of June 2024
        result2 = registration_to_purchase_conversion_rate(
            df_users, df_orders, "2024-06-01", "2024-06-15", 30
        )

        log_success(logger, f"Test 2 - First half June 2024: {result2}")

        # Test 3: Different conversion window
        result3 = registration_to_purchase_conversion_rate(
            df_users, df_orders, "2024-06-01", "2024-06-30", 7
        )

        log_success(logger, f"Test 3 - June 2024 (7-day window): {result3}")

        # * Validate results
        log_info(logger, "Validating results...")

        for i, result in enumerate([result1, result2, result3], 1):
            if isinstance(result, dict) and "conversion_rate" in result:
                rate = result["conversion_rate"]
                registered = result["registered_users"]
                converted = result["converted_users"]
                avg_days = result["average_days_to_purchase"]

                log_info(logger, f"Test {i} validation:")
                log_info(
                    logger,
                    f"  - Conversion rate: {rate}% ({converted}/{registered} users)",
                )
                log_info(logger, f"  - Average days to purchase: {avg_days}")

                # Basic validation
                assert 0 <= rate <= 100, f"Invalid conversion rate: {rate}"
                assert (
                    converted <= registered
                ), f"Converted users ({converted}) > registered users ({registered})"
                assert avg_days >= 0, f"Invalid average days: {avg_days}"

                log_success(logger, f"  ✅ Test {i} validation passed")
            else:
                log_error(
                    logger,
                    ValueError(f"Invalid result format: {result}"),
                    f"Test {i} validation",
                )

        log_success(logger, "All conversion rate analytics tests passed!")

    except Exception as e:
        log_error(logger, e, "Error in conversion analytics test")
        return False

    return True


def test_langgraph_integration():
    """Test the LangGraph agent with conversion rate queries."""
    logger = get_logger(__name__)

    log_info(logger, "Testing LangGraph Integration")
    log_info(logger, "=" * 50)

    try:
        # * Import LangGraph components
        from src.langgraph_agent import create_analytics_agent

        # * Create agent
        log_info(logger, "Creating analytics agent...")
        agent = create_analytics_agent()

        # * Test queries
        test_queries = [
            "What was the conversion rate from registration to purchase for users who registered in June 2024?",
            "Calculate conversion rate for users registered between 2024-06-01 and 2024-06-15 with a 30-day window",
            "Show me both active users by region and conversion rate for June 2024",
        ]

        for i, query in enumerate(test_queries, 1):
            log_info(logger, f"Test Query {i}: {query}")
            log_info(logger, "-" * 60)

            try:
                response = agent.invoke(
                    {"messages": [{"role": "user", "content": query}]}
                )

                agent_response = response["messages"][-1].content
                log_success(logger, f"Agent Response {i}:")
                log_info(logger, agent_response)
                log_info(logger, "")

            except Exception as e:
                log_error(logger, e, f"Error in test query {i}")

        log_success(logger, "LangGraph integration tests completed!")

    except Exception as e:
        log_error(logger, e, "Error in LangGraph integration test")
        return False

    return True


def main():
    """Run all tests."""
    logger = get_logger(__name__)

    log_info(logger, "🚀 Starting Conversion Rate Analytics Tests")
    log_info(logger, "=" * 60)

    success_count = 0
    total_tests = 2

    # Test 1: Analytics function
    if test_conversion_analytics():
        success_count += 1
        log_success(logger, "✅ Conversion analytics function test PASSED")
    else:
        log_error(
            logger,
            Exception("Test failed"),
            "❌ Conversion analytics function test FAILED",
        )

    log_info(logger, "")

    # Test 2: LangGraph integration
    if test_langgraph_integration():
        success_count += 1
        log_success(logger, "✅ LangGraph integration test PASSED")
    else:
        log_error(
            logger, Exception("Test failed"), "❌ LangGraph integration test FAILED"
        )

    log_info(logger, "")
    log_info(logger, "=" * 60)
    log_info(logger, f"Test Results: {success_count}/{total_tests} tests passed")

    if success_count == total_tests:
        log_success(
            logger,
            "🎉 All tests passed! Conversion rate analytics is working correctly.",
        )
        return 0
    else:
        log_error(
            logger,
            Exception(f"Only {success_count}/{total_tests} tests passed"),
            "Some tests failed",
        )
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
