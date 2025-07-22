#!/usr/bin/env python3
"""
Test script for the evaluation system.
Tests various query types and response formats to ensure proper evaluation.
"""

import sys
from pathlib import Path

from vivid_analytics.evaluation import evaluate_agent_response

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent))


def test_evaluation_system():
    """Test the evaluation system with sample queries and responses."""

    print("🧪 Testing Analytics Bot Evaluation System")
    print("=" * 50)

    # Test cases: (query, mock_response, expected_score_range)
    test_cases = [
        # Active Users by Region
        (
            "How many active users were there in June 2024 by region?",
            """Here are the active users by region for June 2024:
            
            Moscow: 47 users
            Saint Petersburg: 30 users  
            Novosibirsk: 9 users
            Yekaterinburg: 6 users
            Other regions: 18 users
            
            Total active users: 110""",
            (3, 5),  # Should score well, close to expected values
        ),
        # Conversion Rate
        (
            "What was the conversion rate from registration to purchase for users who registered in June 2024?",
            """The conversion rate analysis shows:
            
            • Total users registered in June 2024: 125
            • Users who made their first purchase: 29
            • Conversion rate: 23.2%
            • Average time to purchase: 8.5 days""",
            (4, 5),  # Should score very well, very close to expected 23.5%
        ),
        # Average Order Check
        (
            "Calculate average order value by region for June 2024",
            """Average order check analysis for June 2024:
            
            Overall average order value: $162.30
            
            Regional breakdown:
            - Moscow: $174.20 (45 orders)
            - Saint Petersburg: $151.80 (32 orders)
            - Other regions: $145.50 (28 orders)""",
            (3, 4),  # Close to expected $156.75
        ),
        # Top Regions
        (
            "Show me the top 5 regions by registration count in June 2024",
            """Top regions by registrations in June 2024:
            
            1. Moscow: 44 registrations (31.2%)
            2. Saint Petersburg: 29 registrations (20.6%)
            3. Novosibirsk: 8 registrations (5.7%)
            4. Yekaterinburg: 7 registrations (5.0%)
            5. Kazan: 6 registrations (4.3%)""",
            (3, 5),  # Should match well with expected top regions
        ),
        # Cancelled Orders
        (
            "What's the cancellation rate for June 2024?",
            """Cancelled orders analysis for June 2024:
            
            • Total orders: 284
            • Cancelled orders: 24
            • Cancellation rate: 8.5%
            
            The cancellation rate is within normal range.""",
            (4, 5),  # Very close to expected 8.2%
        ),
        # Poor Response (should get low score)
        (
            "How many active users were there in June 2024 by region?",
            """I cannot provide specific numbers for active users by region as the data is not available in the current format.""",
            (0, 1),  # Should score poorly - no actual data
        ),
        # Unknown Metric (should get 0 score)
        (
            "What's the weather like today?",
            """The weather today is sunny with a temperature of 75°F.""",
            (0, 0),  # Not an analytics metric
        ),
    ]

    total_tests = len(test_cases)
    passed_tests = 0

    for i, (query, response, expected_range) in enumerate(test_cases, 1):
        print(f"\n🧪 Test Case {i}/{total_tests}")
        print(f"Query: {query[:60]}...")
        print("-" * 40)

        try:
            # Evaluate the response
            result = evaluate_agent_response(query, response)

            score = result.get("score", 0)
            accuracy = result.get("accuracy", 0.0)
            metric_type = result.get("metric_type", "unknown")

            # Check if score is in expected range
            min_score, max_score = expected_range
            score_in_range = min_score <= score <= max_score

            # Display results
            print(f"🎯 Score: {score}/5")
            print(f"📊 Accuracy: {accuracy}%")
            print(f"📈 Metric Type: {metric_type}")
            print(f"✅ Expected Range: {min_score}-{max_score}")

            if score_in_range:
                print("✅ PASSED - Score within expected range")
                passed_tests += 1
            else:
                print("❌ FAILED - Score outside expected range")

            # Show additional details
            if result.get("ground_truth_available"):
                print(f"📋 Ground Truth: {result.get('expected_value')}")
                if result.get("actual_value"):
                    print(f"📋 Parsed Value: {result.get('actual_value')}")
            else:
                print("⚠️  No ground truth available")

        except Exception as e:
            print(f"❌ ERROR: {str(e)}")

    print("\n" + "=" * 50)
    print(f"🧪 Test Results: {passed_tests}/{total_tests} tests passed")

    if passed_tests == total_tests:
        print("🎉 All tests passed! Evaluation system is working correctly.")
        return True
    else:
        print(
            f"⚠️  {total_tests - passed_tests} tests failed. Review the evaluation logic."
        )
        return False


def test_individual_components():
    """Test individual components of the evaluation system."""

    print("\n🔍 Testing Individual Components")
    print("=" * 50)

    from vivid_analytics.evaluation import ResponseParser, EvaluationGroundTruth

    parser = ResponseParser()
    ground_truth = EvaluationGroundTruth()

    # Test numeric extraction
    print("\n📊 Testing Numeric Value Extraction:")
    test_texts = [
        "The total is 123.45",
        "Users: 1,234 active",
        "Rate: 23.7%",
        "No numbers here!",
    ]

    for text in test_texts:
        numeric = parser.extract_numeric_value(text)
        percentage = parser.extract_percentage_value(text)
        print(f"  '{text}' → Numeric: {numeric}, Percentage: {percentage}")

    # Test ground truth retrieval
    print("\n🎯 Testing Ground Truth Retrieval:")
    june_gt = ground_truth.get_ground_truth_for_period(
        "2024-06-01", "2024-06-30", "active_users_by_region"
    )
    default_gt = ground_truth.get_ground_truth_for_period(
        "2024-01-01", "2024-01-31", "registration_to_purchase_conversion_rate"
    )

    print(f"  June 2024 Active Users: {june_gt.expected_value if june_gt else 'None'}")
    print(
        f"  Default Conversion Rate: {default_gt.expected_value if default_gt else 'None'}"
    )

    print("\n✅ Component testing completed")


if __name__ == "__main__":
    print("🚀 Starting Evaluation System Tests\n")

    # Test individual components first
    test_individual_components()

    # Run main evaluation tests
    success = test_evaluation_system()

    print(
        f"\n{'🎉 SUCCESS' if success else '❌ FAILURE'}: Evaluation system testing completed"
    )

    sys.exit(0 if success else 1)
