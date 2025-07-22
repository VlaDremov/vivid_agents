import re
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional, Union, List, Tuple
from dataclasses import dataclass
from datetime import timedelta
from .logger_config import (
    get_analytics_logger,
    log_info,
    log_success,
    log_warning,
    log_error,
)
from .langgraph_agent import create_analytics_agent
from .analytics import (
    active_users_by_region,
    registration_to_purchase_conversion_rate,
    average_order_check_by_region,
    users_without_orders_by_region,
    top_regions_by_registrations,
    cancelled_orders_share,
    customer_lifetime_value,
    repeat_customers_percentage,
    visitors_without_purchase,
    registration_dynamic,
)

logger = get_analytics_logger()


@dataclass
class GroundTruthValue:
    """Represents expected values for a metric with tolerance ranges."""

    expected_value: Union[float, int, Dict, List]
    tolerance_percent: float = 10.0  # Default 10% tolerance
    value_type: str = "numeric"  # "numeric", "dict", "list", "percentage"
    description: str = ""


class EvaluationGroundTruth:
    """Manages ground truth values for all analytics metrics."""

    def __init__(self):
        """Initialize ground truth values for all metrics."""
        self.ground_truth = self._initialize_ground_truth()
        log_info(logger, "Ground truth evaluation system initialized")

    def _initialize_ground_truth(self) -> Dict[str, Dict[str, GroundTruthValue]]:
        """Initialize ground truth values by calling analytics functions with fixed parameters."""
        start_date = "2024-06-01"
        end_date = "2024-06-30"

        log_info(
            logger,
            f"Computing ground truth values for period {start_date} to {end_date}",
        )

        # Load data files
        try:
            users_csv_path = "data/raw/users.csv"
            orders_csv_path = "data/raw/orders.csv"

            if not Path(users_csv_path).exists() or not Path(orders_csv_path).exists():
                log_warning(logger, "Data files not found, generating dummy data")
                from .data.make_dummies import make_dummy_csvs

                make_dummy_csvs()

            df_users = pd.read_csv(users_csv_path)
            df_orders = pd.read_csv(orders_csv_path)
            log_info(
                logger, f"Loaded data: {len(df_users)} users, {len(df_orders)} orders"
            )

        except Exception as e:
            log_error(logger, e, "Failed to load data files")
            return {}

        # Calculate actual values using analytics functions
        try:
            # Active users by region
            active_users_result = active_users_by_region(df_users, start_date, end_date)

            # Registration to purchase conversion rate
            conversion_result = registration_to_purchase_conversion_rate(
                df_users, df_orders, start_date, end_date
            )

            # Average order check by region
            avg_order_result = average_order_check_by_region(
                df_users, df_orders, start_date, end_date
            )

            # Users without orders by region
            users_no_orders_result = users_without_orders_by_region(
                df_users, df_orders, start_date, end_date
            )

            # Top regions by registrations
            top_regions_result = top_regions_by_registrations(
                df_users, start_date, end_date, top_k=5
            )

            # Cancelled orders share
            cancelled_orders_result = cancelled_orders_share(
                df_orders, start_date, end_date
            )

            # Customer lifetime value
            clv_result = customer_lifetime_value(
                df_users, df_orders, start_date, end_date
            )

            # Repeat customers percentage
            repeat_customers_result = repeat_customers_percentage(
                df_orders, start_date, end_date
            )

            # Visitors without purchase
            visitors_no_purchase_result = visitors_without_purchase(
                df_users, df_orders, start_date, end_date
            )

            # Registration dynamics
            registration_dynamic_result = registration_dynamic(
                df_users, start_date, end_date
            )

            log_success(logger, "Ground truth values computed successfully")

            return {
                # June 2024 period (computed from actual data)
                f"{start_date}_{end_date}": {
                    "active_users_by_region": GroundTruthValue(
                        expected_value=active_users_result,
                        tolerance_percent=15.0,
                        value_type="numeric",
                        description="Expected total active users for June 2024",
                    ),
                    "registration_to_purchase_conversion_rate": GroundTruthValue(
                        expected_value=conversion_result,
                        tolerance_percent=12.0,
                        value_type="percentage",
                        description="Expected conversion rate from registration to purchase",
                    ),
                    "average_order_check_by_region": GroundTruthValue(
                        expected_value=avg_order_result,
                        tolerance_percent=10.0,
                        value_type="numeric",
                        description="Expected average order check across all regions",
                    ),
                    "users_without_orders_by_region": GroundTruthValue(
                        expected_value=users_no_orders_result,
                        tolerance_percent=15.0,
                        value_type="numeric",
                        description="Expected users who registered but never made orders",
                    ),
                    "top_regions_by_registrations": GroundTruthValue(
                        expected_value=top_regions_result,
                        tolerance_percent=12.0,
                        value_type="list",
                        description="Expected top 5 regions by registration count",
                    ),
                    "cancelled_orders_share": GroundTruthValue(
                        expected_value=cancelled_orders_result,
                        tolerance_percent=20.0,
                        value_type="percentage",
                        description="Expected percentage of cancelled orders",
                    ),
                    "customer_lifetime_value": GroundTruthValue(
                        expected_value=clv_result,
                        tolerance_percent=15.0,
                        value_type="numeric",
                        description="Expected customer lifetime value",
                    ),
                    "repeat_customers_percentage": GroundTruthValue(
                        expected_value=repeat_customers_result,
                        tolerance_percent=12.0,
                        value_type="percentage",
                        description="Expected percentage of customers with multiple orders",
                    ),
                    "visitors_without_purchase": GroundTruthValue(
                        expected_value=visitors_no_purchase_result,
                        tolerance_percent=18.0,
                        value_type="numeric",
                        description="Expected number of visitors who didn't make purchases",
                    ),
                    "registration_dynamic": GroundTruthValue(
                        expected_value=registration_dynamic_result,
                        tolerance_percent=15.0,
                        value_type="dict",
                        description="Expected daily registration counts for June 2024",
                    ),
                },
            }

        except Exception as e:
            log_error(logger, e, "Failed to compute ground truth values")
            return {}

    def get_ground_truth_for_period(
        self, start_date: str, end_date: str, metric_type: str
    ) -> Optional[GroundTruthValue]:
        """Get ground truth value for a specific metric and time period."""
        period_key = f"{start_date}_{end_date}"

        # Only return ground truth for exact period matches
        if (
            period_key in self.ground_truth
            and metric_type in self.ground_truth[period_key]
        ):
            return self.ground_truth[period_key][metric_type]

        # No fallback to default values - this will trigger the model comparison fallback
        log_warning(
            logger,
            f"No ground truth found for metric: {metric_type}, period: {period_key}",
        )
        return None


class ResponseParser:
    """Parses agent responses to extract metric values for comparison."""

    @staticmethod
    def extract_numeric_value(text: str) -> Optional[float]:
        """Extract numeric values from text (handles numbers with commas, decimals, etc.)."""
        log_info(logger, f"Extracting numeric value from: '{text}'")

        # Look for numbers (including decimals and commas) but avoid years
        number_patterns = [
            r"\$(\d{1,3}(?:,\d{3})+(?:\.\d+)?)",  # Currency with commas (e.g., $2,929.46)
            r"\$(\d+(?:\.\d+)?)",  # Currency amounts (e.g., $123.45, $2929.46)
            r"(\d{1,3}(?:,\d{3})+(?:\.\d+)?)",  # Numbers with commas (e.g., 2,929.46)
            r"(?<!(?:19|20|21)\d{2}[^\d])(\d+\.\d+)(?![^\d]*(?:year|yr|month|day))",  # Decimals not followed by time units
            r"(?:were|are|have|total|count|number)[\s\w]*?(\d+)",  # Numbers after quantity words
            r"(\d+)(?:\s+(?:users|customers|orders|people|visitors|registrations))",  # Numbers before count nouns
            r"(?<!(?:19|20|21)\d{2}[^\d])(\d+)(?![^\d]*(?:year|yr|\d{4}|january|february|march|april|may|june|july|august|september|october|november|december))",  # Numbers not part of years or dates
        ]

        for i, pattern in enumerate(number_patterns):
            matches = re.findall(pattern, text)
            log_info(logger, f"Pattern {i+1} '{pattern}' found matches: {matches}")
            if matches:
                # Return the first numeric value found, cleaned up
                value_str = matches[0].replace(",", "")
                try:
                    result = float(value_str)
                    log_info(
                        logger, f"✅ Successfully extracted numeric value: {result}"
                    )
                    return result
                except ValueError:
                    log_warning(logger, f"Could not convert '{value_str}' to float")
                    continue

        log_warning(logger, f"No numeric value found in text: '{text}'")
        return None

    @staticmethod
    def extract_percentage_value(text: str) -> Optional[float]:
        """Extract percentage values from text."""
        # Look for percentage patterns
        percentage_patterns = [
            r"(\d+\.?\d*)%",  # Direct percentage with %
            r"(\d+\.?\d*)\s*percent",  # Word 'percent'
            r"conversion rate[:\s]*(\d+\.?\d*)%?",  # Conversion rate specific
            r"rate[:\s]*(\d+\.?\d*)%?",  # Rate specific
        ]

        for pattern in percentage_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                try:
                    return float(matches[0])
                except ValueError:
                    continue

        return None

    @staticmethod
    def extract_regional_data(text: str) -> Optional[Dict[str, int]]:
        """Extract regional data from text responses."""
        regional_data = {}

        # Common patterns for regional data
        patterns = [
            r"([A-Za-z\s]+):\s*(\d+)",  # "Moscow: 45"
            r"([A-Za-z\s]+)\s*-\s*(\d+)",  # "Moscow - 45"
            r"([A-Za-z\s]+)\s*\|\s*(\d+)",  # "Moscow | 45"
        ]

        for pattern in patterns:
            matches = re.findall(pattern, text)
            for region, count in matches:
                region = region.strip()
                if len(region) > 2:  # Filter out short matches
                    try:
                        regional_data[region] = int(count)
                    except ValueError:
                        continue

        return regional_data if regional_data else None

    @staticmethod
    def extract_top_regions_data(text: str) -> Optional[List[Dict[str, Any]]]:
        """Extract top regions list from text responses."""
        regions_list = []

        # Look for ranked lists
        patterns = [
            r"(\d+)[.)]\s*([A-Za-z\s]+):\s*(\d+)",  # "1. Moscow: 45"
            r"([A-Za-z\s]+):\s*(\d+)\s*\((\d+\.?\d*)%\)",  # "Moscow: 45 (23.5%)"
        ]

        for pattern in patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                if len(match) == 3:
                    try:
                        rank, region, registrations = match
                        regions_list.append(
                            {
                                "rank": int(rank),
                                "region": region.strip(),
                                "registrations": int(registrations),
                            }
                        )
                    except ValueError:
                        continue
                elif len(match) == 3 and pattern.endswith(r"\((\d+\.?\d*)%\)"):
                    try:
                        region, registrations, percentage = match
                        regions_list.append(
                            {
                                "region": region.strip(),
                                "registrations": int(registrations),
                                "percentage": float(percentage),
                            }
                        )
                    except ValueError:
                        continue

        return (
            sorted(
                regions_list,
                key=lambda x: x.get("rank", x.get("registrations", 0)),
                reverse=True,
            )
            if regions_list
            else None
        )

    @staticmethod
    def extract_registration_dynamic_data(text: str) -> Optional[Dict[str, int]]:
        """Extract daily registration counts from text responses."""
        registration_data = {}

        # Common patterns for date-count pairs (preferred format first)
        patterns = [
            r"-\s*([A-Za-z]+\s+\d{1,2}):\s*(\d+)",  # "- June 1: 5" (preferred format)
            r"([A-Za-z]+\s+\d{1,2}):\s*(\d+)",  # "June 1: 5" (simple format)
            r"-\s*\*\*([A-Za-z]+\s+\d{1,2}):\*\*\s*(\d+)",  # "- **June 1:** 5" (bullet with colon inside bold)
            r"\*\*([A-Za-z]+\s+\d{1,2}):\*\*\s*(\d+)",  # "**June 1:** 5" (colon inside bold)
            r"\*\*([A-Za-z]+\s+\d{1,2})\*\*:\s*(\d+)",  # "**June 1**: 5" (colon outside bold)
            r"\*\*(\d{4}-\d{2}-\d{2})\*\*:\s*(\d+)",  # "**2024-06-01**: 5" (markdown bold)
            r"(\d{4}-\d{2}-\d{2}):\s*(\d+)",  # "2024-06-01: 5"
            r"(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{4}):\s*(\d+)",  # "06/01/2024: 5"
        ]

        for i, pattern in enumerate(patterns):
            matches = re.findall(pattern, text, re.IGNORECASE)
            log_info(logger, f"Registration pattern {i+1} '{pattern}' found matches: {matches}")
            for date_str, count in matches:
                date_str = date_str.strip()
                try:
                    # Try to parse the count
                    count_value = int(count)
                    
                    # Convert month names to YYYY-MM-DD format if needed
                    normalized_date = ResponseParser._normalize_date_key(date_str)
                    registration_data[normalized_date] = count_value
                    log_info(logger, f"Successfully parsed: '{date_str}' -> '{normalized_date}' = {count_value}")
                except ValueError:
                    log_warning(logger, f"Failed to parse count '{count}' for date '{date_str}'")
                    continue

        log_info(logger, f"Final registration data: {registration_data}")
        return registration_data if registration_data else None

    @staticmethod
    def _normalize_date_key(date_str: str) -> str:
        """Convert various date formats to YYYY-MM-DD format."""
        # If already in YYYY-MM-DD format, return as-is
        if re.match(r"\d{4}-\d{2}-\d{2}", date_str):
            return date_str
            
        # Handle "Month Day" format (e.g., "June 1")
        month_mapping = {
            'january': '01', 'jan': '01',
            'february': '02', 'feb': '02', 
            'march': '03', 'mar': '03',
            'april': '04', 'apr': '04',
            'may': '05',
            'june': '06', 'jun': '06',
            'july': '07', 'jul': '07',
            'august': '08', 'aug': '08',
            'september': '09', 'sep': '09',
            'october': '10', 'oct': '10', 
            'november': '11', 'nov': '11',
            'december': '12', 'dec': '12'
        }
        
        # Try to parse "Month Day" format
        month_day_match = re.match(r"([a-zA-Z]+)\s+(\d{1,2})", date_str.lower())
        if month_day_match:
            month_name = month_day_match.group(1)
            day = int(month_day_match.group(2))
            
            if month_name in month_mapping:
                # Assume 2024 for now (could be made more flexible)
                return f"2024-{month_mapping[month_name]}-{day:02d}"
        
        # If no conversion possible, return original
        return date_str


class MetricEvaluator:
    """Evaluates agent responses against ground truth values."""

    def __init__(self):
        self.ground_truth = EvaluationGroundTruth()
        self.parser = ResponseParser()
        self.primary_agent = None
        self.fallback_agent = None

    def identify_metric_type(self, query: str) -> Optional[str]:
        """Identify the type of metric from user query."""
        query_lower = query.lower()
        log_info(logger, f"Analyzing query for metric identification: '{query_lower}'")

        metric_keywords = {
            "active_users_by_region": [
                "active users",
                "users by region",
                "regional users",
                "active by region",
                "total users",
                "total active",
            ],
            "registration_to_purchase_conversion_rate": [
                "conversion rate",
                "registration to purchase",
                "conversion",
                "purchase conversion",
            ],
            "average_order_check_by_region": [
                "average order",
                "order check",
                "order value",
                "average check",
                "spending",
                "order amount",
                "average spending",
                "average purchase",
            ],
            "users_without_orders_by_region": [
                "users without orders",
                "non-purchasing users",
                "never ordered",
                "without orders",
                "users registered but never made orders",
                "users registered but never made any orders",
                "users registered but never made any purchases",
                "users registered but never made any purchases",
            ],
            "top_regions_by_registrations": [
                "top regions",
                "best regions",
                "highest registration",
                "most registrations",
                "regions by registration count",
                "top regions by registrations",
            ],
            "cancelled_orders_share": [
                "cancelled orders",
                "cancellation rate",
                "cancelled",
                "order cancellation",
            ],
            "customer_lifetime_value": [
                "lifetime value",
                "clv",
                "ltv",
                "customer lifetime value",
            ],
            "repeat_customers_percentage": [
                "repeat customers",
                "multiple orders",
                "returning customers",
                "repeat",
            ],
            "visitors_without_purchase": [
                "visitors without purchase",
                "non-purchasing visitors",
                "no purchase",
            ],
            "registration_dynamic": [
                "registration dynamics",
                "registration dynamic",  # Add singular form
                "registration trends", 
                "daily registrations",
                "registration over time",
                "registration pattern",
                "registration by day", 
                "daily registration counts",
                "registration timeline",
                "registrations over time",  # Add common variations
                "registration data",
                "registration count",
            ],
        }

        for metric_type, keywords in metric_keywords.items():
            found_keywords = [keyword for keyword in keywords if keyword in query_lower]
            if found_keywords:
                log_info(
                    logger,
                    f"Matched metric '{metric_type}' with keywords: {found_keywords}",
                )
                return metric_type
            else:
                log_info(logger, f"No match for '{metric_type}' (keywords: {keywords})")

        log_warning(logger, f"No metric type identified for query: '{query_lower}'")
        return None

    def extract_dates_from_query(
        self, query: str
    ) -> Tuple[Optional[str], Optional[str]]:
        """Extract start and end dates from query."""
        query_lower = query.lower()

        # Month-year patterns (most common in analytics queries)
        month_patterns = {
            "january 2024": ("2024-01-01", "2024-01-31"),
            "february 2024": ("2024-02-01", "2024-02-29"),
            "march 2024": ("2024-03-01", "2024-03-31"),
            "april 2024": ("2024-04-01", "2024-04-30"),
            "may 2024": ("2024-05-01", "2024-05-31"),
            "june 2024": ("2024-06-01", "2024-06-30"),
            "july 2024": ("2024-07-01", "2024-07-31"),
            "august 2024": ("2024-08-01", "2024-08-31"),
            "september 2024": ("2024-09-01", "2024-09-30"),
            "october 2024": ("2024-10-01", "2024-10-31"),
            "november 2024": ("2024-11-01", "2024-11-30"),
            "december 2024": ("2024-12-01", "2024-12-31"),
            "jan 2024": ("2024-01-01", "2024-01-31"),
            "feb 2024": ("2024-02-01", "2024-02-29"),
            "mar 2024": ("2024-03-01", "2024-03-31"),
            "apr 2024": ("2024-04-01", "2024-04-30"),
            "jul 2024": ("2024-07-01", "2024-07-31"),
            "aug 2024": ("2024-08-01", "2024-08-31"),
            "sep 2024": ("2024-09-01", "2024-09-30"),
            "oct 2024": ("2024-10-01", "2024-10-31"),
            "nov 2024": ("2024-11-01", "2024-11-30"),
            "dec 2024": ("2024-12-01", "2024-12-31"),
        }

        # Check for month-year patterns first
        for month_phrase, (start, end) in month_patterns.items():
            if month_phrase in query_lower:
                return start, end

        # Explicit date range patterns
        explicit_patterns = [
            r"from\s*(\d{4}-\d{2}-\d{2})\s*to\s*(\d{4}-\d{2}-\d{2})",  # from X to Y
            r"between\s*(\d{4}-\d{2}-\d{2})\s*and\s*(\d{4}-\d{2}-\d{2})",  # between X and Y
        ]

        for pattern in explicit_patterns:
            matches = re.findall(pattern, query, re.IGNORECASE)
            if matches:
                return matches[0][0], matches[0][1]

        # Single date patterns (assume month period)
        single_date_patterns = [
            r"(\d{4}-\d{2}-\d{2})",  # YYYY-MM-DD
        ]

        for pattern in single_date_patterns:
            matches = re.findall(pattern, query, re.IGNORECASE)
            if matches and len(matches) >= 2:
                return matches[0], matches[1]
            elif matches and len(matches) == 1:
                # Single date found, assume it's for the whole month
                date_str = matches[0]
                try:
                    from datetime import datetime

                    date_obj = datetime.strptime(date_str, "%Y-%m-%d")
                    # Return first and last day of the month
                    if date_obj.month == 12:
                        next_month = date_obj.replace(
                            year=date_obj.year + 1, month=1, day=1
                        )
                    else:
                        next_month = date_obj.replace(month=date_obj.month + 1, day=1)
                    last_day = next_month.replace(day=1) - timedelta(days=1)
                    return date_obj.replace(day=1).strftime(
                        "%Y-%m-%d"
                    ), last_day.strftime("%Y-%m-%d")
                except Exception:
                    pass

        return None, None

    def compare_values(
        self, actual: Union[float, Dict, List], expected: GroundTruthValue
    ) -> float:
        """Compare actual vs expected values and return accuracy score (0-100)."""
        if expected.value_type == "numeric" or expected.value_type == "percentage":
            if not isinstance(actual, (int, float)):
                return 0.0

            expected_val = float(expected.expected_value)
            tolerance = expected_val * (expected.tolerance_percent / 100.0)

            # Calculate accuracy based on how close actual is to expected
            diff = abs(actual - expected_val)
            if diff <= tolerance:
                # Full accuracy if within tolerance
                accuracy = 100.0 - (diff / tolerance * 20.0)  # Scale within tolerance
                return max(80.0, accuracy)  # Minimum 80% if within tolerance
            else:
                # Diminishing accuracy outside tolerance
                excess_diff = diff - tolerance
                max_acceptable_diff = expected_val * 0.5  # 50% of expected value
                if excess_diff >= max_acceptable_diff:
                    return 0.0
                accuracy = 80.0 * (1.0 - excess_diff / max_acceptable_diff)
                return max(0.0, accuracy)

        elif expected.value_type == "dict":
            if not isinstance(actual, dict) or not isinstance(
                expected.expected_value, dict
            ):
                return 0.0

            total_score = 0.0
            total_keys = 0

            for key, expected_val in expected.expected_value.items():
                total_keys += 1
                if key in actual:
                    # Compare each region's value
                    actual_val = float(actual[key])
                    diff = abs(actual_val - expected_val)

                    # Handle case where expected value is 0 (perfect match required)
                    if expected_val == 0:
                        if actual_val == 0:
                            total_score += 100.0  # Perfect match
                        else:
                            total_score += max(
                                0.0, 50.0 - abs(actual_val) * 10
                            )  # Penalty for non-zero when expecting 0
                    else:
                        tolerance = expected_val * (expected.tolerance_percent / 100.0)

                        if diff <= tolerance:
                            # Avoid division by zero when tolerance is 0
                            if tolerance > 0:
                                key_score = 100.0 - (diff / tolerance * 20.0)
                            else:
                                key_score = 100.0  # Perfect match when tolerance is 0
                            total_score += max(80.0, key_score)
                        else:
                            excess_diff = diff - tolerance
                            max_acceptable = expected_val * 0.5
                            if max_acceptable > 0 and excess_diff < max_acceptable:
                                key_score = 80.0 * (1.0 - excess_diff / max_acceptable)
                                total_score += max(0.0, key_score)
                            # else: 0 points for this key

            return total_score / total_keys if total_keys > 0 else 0.0

        elif expected.value_type == "list":
            if not isinstance(actual, list) or not isinstance(
                expected.expected_value, list
            ):
                return 0.0

            # For lists (like top regions), compare key matching and ordering
            expected_regions = {
                item["region"]: item["registrations"]
                for item in expected.expected_value
            }

            total_score = 0.0
            max_possible = len(expected.expected_value)

            for i, actual_item in enumerate(actual[:max_possible]):
                if isinstance(actual_item, dict) and "region" in actual_item:
                    region = actual_item["region"]
                    if region in expected_regions:
                        # Bonus for correct position
                        position_bonus = max(0, 20 - i * 5)  # Decreases with position

                        # Value accuracy
                        if "registrations" in actual_item:
                            expected_val = expected_regions[region]
                            actual_val = actual_item["registrations"]
                            tolerance = expected_val * (
                                expected.tolerance_percent / 100.0
                            )
                            diff = abs(actual_val - expected_val)

                            if diff <= tolerance:
                                value_score = 80.0 - (diff / tolerance * 20.0)
                            else:
                                value_score = max(0, 40.0 - diff)

                            total_score += value_score + position_bonus
                        else:
                            total_score += (
                                50.0 + position_bonus
                            )  # Partial credit for region match

            return min(100.0, total_score / max_possible) if max_possible > 0 else 0.0

        return 0.0

    def initialize_agents_for_comparison(self):
        """Initialize primary and fallback agents with different models."""
        try:
            if self.primary_agent is None:
                log_info(
                    logger,
                    "Initializing primary agent (gpt-4o-mini) for fallback comparison",
                )
                self.primary_agent = create_analytics_agent("gpt-4o-mini")

            if self.fallback_agent is None:
                log_info(
                    logger,
                    "Initializing fallback agent (o3-mini-2025-01-31) for comparison",
                )
                self.fallback_agent = create_analytics_agent("o3-mini-2025-01-31")

            return True
        except Exception as e:
            log_warning(logger, f"Failed to initialize comparison agents: {str(e)}")
            return False

    def get_model_response(self, agent, query: str) -> Optional[str]:
        """Get response from a specific model agent."""
        try:
            response = agent.invoke({"messages": [{"role": "user", "content": query}]})
            return response["messages"][-1].content
        except Exception as e:
            log_warning(logger, f"Failed to get model response: {str(e)}")
            return None

    def compare_model_responses(
        self, query: str, response1: str, response2: str, metric_type: str
    ) -> Dict[str, Any]:
        """Compare responses from two different models to assess consistency."""
        log_info(logger, f"Comparing model responses for {metric_type}")

        # Parse values from both responses
        if metric_type in [
            "registration_to_purchase_conversion_rate",
            "cancelled_orders_share",
            "repeat_customers_percentage",
        ]:
            value1 = self.parser.extract_percentage_value(response1)
            value2 = self.parser.extract_percentage_value(response2)
            value_type = "percentage"
        elif metric_type in [
            "average_order_check_by_region",
            "customer_lifetime_value",
            "visitors_without_purchase",
            "users_without_orders_by_region",
        ]:
            value1 = self.parser.extract_numeric_value(response1)
            value2 = self.parser.extract_numeric_value(response2)
            value_type = "numeric"
        elif metric_type == "active_users_by_region":
            value1 = self.parser.extract_regional_data(response1)
            value2 = self.parser.extract_regional_data(response2)
            value_type = "regional"
        elif metric_type == "top_regions_by_registrations":
            value1 = self.parser.extract_top_regions_data(response1)
            value2 = self.parser.extract_top_regions_data(response2)
            value_type = "list"
        else:
            # Default to numeric
            value1 = self.parser.extract_numeric_value(response1)
            value2 = self.parser.extract_numeric_value(response2)
            value_type = "numeric"

        if value1 is None or value2 is None:
            log_warning(
                logger, f"Could not extract values for comparison: {value1}, {value2}"
            )
            return {
                "consistency_score": 0.0,
                "comparison_details": "Could not extract comparable values from responses",
                "value1": value1,
                "value2": value2,
                "value_type": value_type,
            }

        # Calculate consistency score based on value type
        consistency_score = 0.0
        comparison_details = ""

        if value_type in ["numeric", "percentage"]:
            # For numeric values, calculate percentage difference
            val1, val2 = float(value1), float(value2)
            if val1 == 0 and val2 == 0:
                consistency_score = 100.0
                comparison_details = "Both models returned identical zero values"
            elif val1 == 0 or val2 == 0:
                consistency_score = 0.0
                comparison_details = f"One model returned zero ({val1} vs {val2})"
            else:
                avg_value = (val1 + val2) / 2
                diff = abs(val1 - val2)
                percentage_diff = (diff / avg_value) * 100

                # Score based on percentage difference
                if percentage_diff <= 5:
                    consistency_score = 100.0
                elif percentage_diff <= 10:
                    consistency_score = 90.0
                elif percentage_diff <= 20:
                    consistency_score = 75.0
                elif percentage_diff <= 30:
                    consistency_score = 50.0
                else:
                    consistency_score = max(0.0, 50.0 - (percentage_diff - 30))

                comparison_details = (
                    f"Values: {val1} vs {val2} (diff: {percentage_diff:.1f}%)"
                )

        elif value_type == "regional":
            # For regional data, compare overlapping regions
            dict1, dict2 = value1, value2
            common_regions = set(dict1.keys()) & set(dict2.keys())

            if not common_regions:
                consistency_score = 0.0
                comparison_details = "No common regions found"
            else:
                region_scores = []
                for region in common_regions:
                    val1, val2 = dict1[region], dict2[region]
                    if val1 == 0 and val2 == 0:
                        region_scores.append(100.0)
                    elif val1 == 0 or val2 == 0:
                        region_scores.append(0.0)
                    else:
                        avg_val = (val1 + val2) / 2
                        diff = abs(val1 - val2)
                        pct_diff = (diff / avg_val) * 100
                        if pct_diff <= 10:
                            region_scores.append(100.0)
                        elif pct_diff <= 20:
                            region_scores.append(80.0)
                        elif pct_diff <= 30:
                            region_scores.append(60.0)
                        else:
                            region_scores.append(max(0.0, 60.0 - (pct_diff - 30)))

                consistency_score = sum(region_scores) / len(region_scores)
                comparison_details = f"Compared {len(common_regions)} regions, avg consistency: {consistency_score:.1f}%"

        elif value_type == "list":
            # For list data (top regions), compare top entries
            list1, list2 = value1, value2
            if not list1 or not list2:
                consistency_score = 0.0
                comparison_details = "One or both lists are empty"
            else:
                # Compare top 3 entries
                top1_regions = {
                    item.get("region")
                    for item in list1[:3]
                    if isinstance(item, dict) and "region" in item
                }
                top2_regions = {
                    item.get("region")
                    for item in list2[:3]
                    if isinstance(item, dict) and "region" in item
                }

                overlap = len(top1_regions & top2_regions)
                total = len(top1_regions | top2_regions)

                if total == 0:
                    consistency_score = 0.0
                    comparison_details = "Could not extract region names"
                else:
                    consistency_score = (overlap / total) * 100
                    comparison_details = (
                        f"Top region overlap: {overlap}/{total} regions match"
                    )

        return {
            "consistency_score": consistency_score,
            "comparison_details": comparison_details,
            "value1": value1,
            "value2": value2,
            "value_type": value_type,
        }

    def fallback_evaluation(self, query: str) -> Dict[str, Any]:
        """Perform fallback evaluation using model-vs-model comparison."""
        log_info(logger, "Performing fallback evaluation using model comparison")

        # Initialize agents if needed
        if not self.initialize_agents_for_comparison():
            return {
                "score": 0,
                "accuracy": 0.0,
                "metric_type": "unknown",
                "evaluation_details": "Failed to initialize comparison agents",
                "ground_truth_available": False,
                "fallback_used": True,
            }

        # Identify metric type
        metric_type = self.identify_metric_type(query)
        if not metric_type:
            return {
                "score": 0,
                "accuracy": 0.0,
                "metric_type": "unknown",
                "evaluation_details": "Could not identify metric type for comparison",
                "ground_truth_available": False,
                "fallback_used": True,
            }

        # Get responses from both models
        response1 = self.get_model_response(self.primary_agent, query)
        response2 = self.get_model_response(self.fallback_agent, query)

        if not response1 or not response2:
            return {
                "score": 0,
                "accuracy": 0.0,
                "metric_type": metric_type,
                "evaluation_details": "Failed to get responses from both models",
                "ground_truth_available": False,
                "fallback_used": True,
            }

        # Compare responses
        comparison_result = self.compare_model_responses(
            query, response1, response2, metric_type
        )

        consistency_score = comparison_result["consistency_score"]
        score = self.accuracy_to_score(consistency_score)

        log_success(
            logger,
            f"Fallback evaluation completed: {metric_type} - Consistency Score: {score}/5 ({consistency_score:.1f}%)",
        )

        return {
            "score": score,
            "accuracy": round(consistency_score, 1),
            "metric_type": metric_type,
            "evaluation_details": f"Model comparison: {comparison_result['comparison_details']}",
            "ground_truth_available": False,
            "fallback_used": True,
            "primary_response": response1,
            "fallback_response": response2,
            "comparison_result": comparison_result,
        }

    def accuracy_to_score(self, accuracy: float) -> int:
        """Convert accuracy percentage (0-100) to score (0-5)."""
        if accuracy >= 90:
            return 5
        elif accuracy >= 75:
            return 4
        elif accuracy >= 60:
            return 3
        elif accuracy >= 40:
            return 2
        elif accuracy >= 20:
            return 1
        else:
            return 0

    def evaluate_response(self, query: str, agent_response: str) -> Dict[str, Any]:
        """Main evaluation function - evaluates agent response against ground truth."""
        log_info(logger, f"Starting evaluation for query: '{query[:100]}...'")
        log_info(logger, f"Agent response: '{agent_response[:200]}...'")

        # Identify metric type
        metric_type = self.identify_metric_type(query)
        log_info(logger, f"Identified metric type: {metric_type}")

        if not metric_type:
            log_warning(logger, "Could not identify metric type from query")
            return {
                "score": 0,
                "accuracy": 0.0,
                "metric_type": "unknown",
                "evaluation_details": "Could not identify metric type from query",
                "ground_truth_available": False,
            }

        # Extract date range
        start_date, end_date = self.extract_dates_from_query(query)
        if not start_date or not end_date:
            start_date, end_date = "2024-06-01", "2024-06-30"  # Default fallback
            log_info(logger, f"Using default date range: {start_date} to {end_date}")
        else:
            log_info(logger, f"Extracted date range: {start_date} to {end_date}")

        # Check if requested dates are outside our ground truth coverage (June 1-30, 2024)
        ground_truth_start = "2024-06-01"
        ground_truth_end = "2024-06-30"

        if start_date != ground_truth_start or end_date != ground_truth_end:
            log_info(
                logger,
                f"Requested period ({start_date} to {end_date}) is outside ground truth coverage "
                f"({ground_truth_start} to {ground_truth_end}), using fallback evaluation",
            )
            return self.fallback_evaluation(query)

        # Get ground truth
        log_info(
            logger,
            f"Looking for ground truth for metric '{metric_type}' in period '{start_date}_{end_date}'",
        )
        ground_truth = self.ground_truth.get_ground_truth_for_period(
            start_date, end_date, metric_type
        )

        if not ground_truth:
            log_warning(
                logger,
                f"No ground truth available for {metric_type} in period {start_date}_{end_date}, using fallback evaluation",
            )
            return self.fallback_evaluation(query)

        log_info(
            logger,
            f"Found ground truth: value={ground_truth.expected_value}, type={ground_truth.value_type}, tolerance={ground_truth.tolerance_percent}%",
        )

        # Parse actual value from response
        log_info(
            logger, f"Parsing {ground_truth.value_type} value from agent response..."
        )
        actual_value = None

        if ground_truth.value_type == "numeric":
            actual_value = self.parser.extract_numeric_value(agent_response)
            log_info(logger, f"Extracted numeric value: {actual_value}")
        elif ground_truth.value_type == "percentage":
            actual_value = self.parser.extract_percentage_value(agent_response)
            log_info(logger, f"Extracted percentage value: {actual_value}")
        elif (
            ground_truth.value_type == "dict" and metric_type == "registration_dynamic"
        ):
            actual_value = self.parser.extract_registration_dynamic_data(agent_response)
            log_info(logger, f"Extracted registration dynamic data: {actual_value}")
        elif ground_truth.value_type == "dict":
            actual_value = self.parser.extract_regional_data(agent_response)
            log_info(logger, f"Extracted regional data: {actual_value}")
        elif ground_truth.value_type == "list":
            actual_value = self.parser.extract_top_regions_data(agent_response)
            log_info(logger, f"Extracted list data: {actual_value}")

        if actual_value is None:
            log_warning(
                logger,
                f"Could not parse {ground_truth.value_type} value from response for {metric_type}",
            )
            log_info(logger, f"Response text analyzed: '{agent_response}'")
            return {
                "score": 0,
                "accuracy": 0.0,
                "metric_type": metric_type,
                "evaluation_details": f"Could not parse {ground_truth.value_type} value from response",
                "ground_truth_available": True,
                "expected_value": ground_truth.expected_value,
            }

        # Compare values
        log_info(
            logger,
            f"Comparing actual value {actual_value} vs expected {ground_truth.expected_value}",
        )
        accuracy = self.compare_values(actual_value, ground_truth)
        score = self.accuracy_to_score(accuracy)

        log_info(logger, f"Calculated accuracy: {accuracy:.1f}%")
        log_info(logger, f"Final score: {score}/5")

        log_success(
            logger,
            f"Evaluation completed: {metric_type} - Score: {score}/5 (Accuracy: {accuracy:.1f}%)",
        )

        return {
            "score": score,
            "accuracy": round(accuracy, 1),
            "metric_type": metric_type,
            "actual_value": actual_value,
            "expected_value": ground_truth.expected_value,
            "tolerance_percent": ground_truth.tolerance_percent,
            "evaluation_details": ground_truth.description,
            "ground_truth_available": True,
        }


# Global evaluator instance
evaluator = MetricEvaluator()


def evaluate_agent_response(query: str, agent_response: str) -> Dict[str, Any]:
    """
    Main function to evaluate an agent response against ground truth.

    Args:
        query: Original user query
        agent_response: Agent's response text

    Returns:
        Dictionary containing evaluation results including score (0-5)
    """
    return evaluator.evaluate_response(query, agent_response)
