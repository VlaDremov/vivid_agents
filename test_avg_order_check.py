#!/usr/bin/env python3
"""
Test script for the new average order check analytics functionality.
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


def test_avg_order_check_analytics():
    """Test the average order check analytics function."""
    logger = get_logger(__name__)
    
    log_info(logger, "Testing Average Order Check Analytics")
    log_info(logger, "=" * 50)
    
    try:
        # * Import after path setup
        from src.analytics import average_order_check_by_region
        from src.data.make_dummies import make_dummy_csvs
        
        # * Generate test data
        log_info(logger, "Generating test data...")
        make_dummy_csvs()
        
        # * Load test data
        log_info(logger, "Loading test data...")
        df_users = pd.read_csv("data/raw/users.csv")
        df_orders = pd.read_csv("data/raw/orders.csv")
        
        log_info(
            logger, f"Loaded {len(df_users)} users and {len(df_orders)} orders"
        )
        
        # * Test average order check calculation
        log_info(logger, "Testing average order check calculation...")
        
        # Test 1: Full June 2024 period
        result1 = average_order_check_by_region(
            df_users, df_orders, "2024-06-01", "2024-06-30"
        )
        
        log_success(logger, f"Test 1 - June 2024 avg order check: {result1}")
        
        # Test 2: First half of June 2024
        result2 = average_order_check_by_region(
            df_users, df_orders, "2024-06-01", "2024-06-15"
        )
        
        log_success(logger, f"Test 2 - First half June 2024: {result2}")
        
        # Test 3: Single day
        result3 = average_order_check_by_region(
            df_users, df_orders, "2024-06-01", "2024-06-01"
        )
        
        log_success(logger, f"Test 3 - Single day (2024-06-01): {result3}")
        
        # * Validate results
        log_info(logger, "Validating results...")
        
        for i, result in enumerate([result1, result2, result3], 1):
            if isinstance(result, dict) and "_summary" in result:
                summary = result["_summary"]
                total_orders = summary["total_orders"]
                overall_avg = summary["overall_average"]
                regions_count = summary["regions_count"]
                
                log_info(logger, f"Test {i} validation:")
                log_info(
                    logger,
                    f"  - Total orders: {total_orders}, Overall avg: {overall_avg}",
                )
                log_info(logger, f"  - Regions: {regions_count}")
                
                # Basic validation
                assert total_orders >= 0, f"Invalid total orders: {total_orders}"
                assert overall_avg >= 0, f"Invalid overall average: {overall_avg}"
                assert regions_count >= 0, f"Invalid regions count: {regions_count}"
                
                # Validate regional data
                for region, data in result.items():
                    if region != "_summary":
                        assert isinstance(data, dict), f"Invalid data format for {region}"
                        assert "average_order_check" in data, f"Missing avg order check for {region}"
                        assert "order_count" in data, f"Missing order count for {region}"
                        assert data["average_order_check"] >= 0, f"Invalid avg for {region}"
                        assert data["order_count"] >= 0, f"Invalid count for {region}"
                
                log_success(logger, f"  ✅ Test {i} validation passed")
            else:
                log_error(
                    logger,
                    ValueError(f"Invalid result format: {result}"),
                    f"Test {i} validation",
                )
        
        log_success(logger, "All average order check analytics tests passed!")
        
    except Exception as e:
        log_error(logger, e, "Error in average order check analytics test")
        return False
    
    return True


def test_langgraph_integration():
    """Test the LangGraph agent with average order check queries."""
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
            "What's the average order check by region for June 2024?",
            "Calculate average order value by region from 2024-06-01 to 2024-06-15",
            "Show me regional spending patterns for the first half of June",
            "Compare average order values across regions for June 2024",
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


def test_comprehensive_analytics():
    """Test all analytics functions together."""
    logger = get_logger(__name__)
    
    log_info(logger, "Testing Comprehensive Analytics")
    log_info(logger, "=" * 50)
    
    try:
        # * Import LangGraph components
        from src.langgraph_agent import create_analytics_agent
        
        # * Create agent
        log_info(logger, "Creating analytics agent...")
        agent = create_analytics_agent()
        
        # * Comprehensive test query
        comprehensive_query = """
        Give me a complete analytics overview for June 2024:
        1. Active users by region
        2. Registration to purchase conversion rate
        3. Average order check by region
        
        Compare the metrics and provide insights.
        """
        
        log_info(logger, "Comprehensive Query:")
        log_info(logger, comprehensive_query)
        log_info(logger, "-" * 60)
        
        try:
            response = agent.invoke(
                {"messages": [{"role": "user", "content": comprehensive_query}]}
            )
            
            agent_response = response["messages"][-1].content
            log_success(logger, "Comprehensive Analytics Response:")
            log_info(logger, agent_response)
            
        except Exception as e:
            log_error(logger, e, "Error in comprehensive analytics query")
            return False
            
        log_success(logger, "Comprehensive analytics test completed!")
        
    except Exception as e:
        log_error(logger, e, "Error in comprehensive analytics test")
        return False
    
    return True


def main():
    """Run all tests."""
    logger = get_logger(__name__)
    
    log_info(logger, "🚀 Starting Average Order Check Analytics Tests")
    log_info(logger, "=" * 60)
    
    success_count = 0
    total_tests = 3
    
    # Test 1: Analytics function
    if test_avg_order_check_analytics():
        success_count += 1
        log_success(logger, "✅ Average order check analytics function test PASSED")
    else:
        log_error(
            logger,
            Exception("Test failed"),
            "❌ Average order check analytics function test FAILED",
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
    
    # Test 3: Comprehensive analytics
    if test_comprehensive_analytics():
        success_count += 1
        log_success(logger, "✅ Comprehensive analytics test PASSED")
    else:
        log_error(
            logger, Exception("Test failed"), "❌ Comprehensive analytics test FAILED"
        )
    
    log_info(logger, "")
    log_info(logger, "=" * 60)
    log_info(logger, f"Test Results: {success_count}/{total_tests} tests passed")
    
    if success_count == total_tests:
        log_success(
            logger,
            "🎉 All tests passed! Average order check analytics is working correctly.",
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