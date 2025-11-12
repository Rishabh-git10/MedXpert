#!/bin/bash

# ============================================
# MedXpert Test Runner Script
# ============================================
# Comprehensive test execution with various options

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print banner
echo -e "${BLUE}"
echo "================================================"
echo "    MedXpert Test Suite Runner"
echo "================================================"
echo -e "${NC}"

# Check if pytest is installed
if ! command -v pytest &> /dev/null; then
    echo -e "${RED}ERROR: pytest is not installed${NC}"
    echo "Install with: pip install pytest pytest-cov pytest-mock pytest-xdist"
    exit 1
fi

# Create logs directory if it doesn't exist
mkdir -p tests/logs
mkdir -p htmlcov

# Function to run tests with specific markers
run_test_suite() {
    local name=$1
    local marker=$2
    local options=$3
    
    echo -e "\n${YELLOW}Running ${name}...${NC}"
    if pytest ${options} ${marker} -v; then
        echo -e "${GREEN}✓ ${name} passed${NC}"
        return 0
    else
        echo -e "${RED}✗ ${name} failed${NC}"
        return 1
    fi
}

# Parse command line arguments
FAST_MODE=false
COVERAGE_ONLY=false
SPECIFIC_TEST=""
PARALLEL=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --fast)
            FAST_MODE=true
            shift
            ;;
        --coverage)
            COVERAGE_ONLY=true
            shift
            ;;
        --test)
            SPECIFIC_TEST=$2
            shift 2
            ;;
        --parallel)
            PARALLEL=true
            shift
            ;;
        --help)
            echo "Usage: ./run_tests.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --fast        Run only fast tests (skip slow and integration)"
            echo "  --coverage    Generate coverage report only"
            echo "  --test FILE   Run specific test file"
            echo "  --parallel    Run tests in parallel (requires pytest-xdist)"
            echo "  --help        Show this help message"
            echo ""
            echo "Examples:"
            echo "  ./run_tests.sh                    # Run all tests"
            echo "  ./run_tests.sh --fast             # Run only fast tests"
            echo "  ./run_tests.sh --test test_api.py # Run specific file"
            echo "  ./run_tests.sh --parallel         # Run in parallel"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

# Set up test options
TEST_OPTIONS="--cov=. --cov-report=html --cov-report=term --cov-report=xml"

if [ "$PARALLEL" = true ]; then
    TEST_OPTIONS="$TEST_OPTIONS -n auto"
    echo -e "${BLUE}Running tests in parallel mode${NC}"
fi

# Run specific test file if specified
if [ -n "$SPECIFIC_TEST" ]; then
    echo -e "${BLUE}Running specific test: ${SPECIFIC_TEST}${NC}"
    pytest "${SPECIFIC_TEST}" ${TEST_OPTIONS} -v
    exit $?
fi

# Coverage report only
if [ "$COVERAGE_ONLY" = true ]; then
    echo -e "${BLUE}Generating coverage report...${NC}"
    pytest ${TEST_OPTIONS} --cov-report=html --cov-report=term-missing
    echo -e "${GREEN}Coverage report generated in htmlcov/index.html${NC}"
    exit 0
fi

# Track test results
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0

# Fast mode - skip slow tests
if [ "$FAST_MODE" = true ]; then
    echo -e "${BLUE}Running in FAST mode (skipping slow and integration tests)${NC}"
    
    run_test_suite "Unit Tests" "-m 'not slow and not integration'" "${TEST_OPTIONS}"
    if [ $? -eq 0 ]; then ((PASSED_TESTS++)); else ((FAILED_TESTS++)); fi
    ((TOTAL_TESTS++))
    
else
    # Full test suite
    echo -e "${BLUE}Running FULL test suite${NC}"
    
    # 1. Unit Tests
    run_test_suite "Unit Tests" "-m unit" "${TEST_OPTIONS}"
    if [ $? -eq 0 ]; then ((PASSED_TESTS++)); else ((FAILED_TESTS++)); fi
    ((TOTAL_TESTS++))
    
    # 2. API Tests
    run_test_suite "API Tests" "-m api" "${TEST_OPTIONS}"
    if [ $? -eq 0 ]; then ((PASSED_TESTS++)); else ((FAILED_TESTS++)); fi
    ((TOTAL_TESTS++))
    
    # 3. Model Tests
    run_test_suite "Model Tests" "-m model" "${TEST_OPTIONS}"
    if [ $? -eq 0 ]; then ((PASSED_TESTS++)); else ((FAILED_TESTS++)); fi
    ((TOTAL_TESTS++))
    
    # 4. UI Tests
    run_test_suite "UI Tests" "-m ui" "${TEST_OPTIONS}"
    if [ $? -eq 0 ]; then ((PASSED_TESTS++)); else ((FAILED_TESTS++)); fi
    ((TOTAL_TESTS++))
    
    # 5. Integration Tests
    run_test_suite "Integration Tests" "-m integration" "${TEST_OPTIONS}"
    if [ $? -eq 0 ]; then ((PASSED_TESTS++)); else ((FAILED_TESTS++)); fi
    ((TOTAL_TESTS++))
    
    # 6. Security Tests
    run_test_suite "Security Tests" "-m security" "${TEST_OPTIONS}"
    if [ $? -eq 0 ]; then ((PASSED_TESTS++)); else ((FAILED_TESTS++)); fi
    ((TOTAL_TESTS++))
    
    # 7. Performance Tests (optional - slow)
    echo -e "\n${YELLOW}Performance tests are slow. Run? (y/n)${NC}"
    read -r response
    if [[ "$response" =~ ^[Yy]$ ]]; then
        run_test_suite "Performance Tests" "-m performance" "${TEST_OPTIONS}"
        if [ $? -eq 0 ]; then ((PASSED_TESTS++)); else ((FAILED_TESTS++)); fi
        ((TOTAL_TESTS++))
    fi
fi

# Generate final report
echo -e "\n${BLUE}================================================${NC}"
echo -e "${BLUE}           Test Suite Summary${NC}"
echo -e "${BLUE}================================================${NC}"
echo -e "Total Test Suites: ${TOTAL_TESTS}"
echo -e "${GREEN}Passed: ${PASSED_TESTS}${NC}"
if [ $FAILED_TESTS -gt 0 ]; then
    echo -e "${RED}Failed: ${FAILED_TESTS}${NC}"
else
    echo -e "Failed: ${FAILED_TESTS}"
fi

# Check coverage threshold
echo -e "\n${YELLOW}Checking coverage threshold...${NC}"
coverage report --fail-under=70 2>/dev/null
COVERAGE_CHECK=$?

if [ $COVERAGE_CHECK -eq 0 ]; then
    echo -e "${GREEN}✓ Coverage meets minimum threshold (70%)${NC}"
else
    echo -e "${YELLOW}⚠ Coverage below minimum threshold (70%)${NC}"
fi

# Print report locations
echo -e "\n${BLUE}Report Locations:${NC}"
echo "  HTML Coverage: htmlcov/index.html"
echo "  XML Coverage:  coverage.xml"
echo "  Test Logs:     tests/logs/pytest.log"

# Final status
echo -e "\n${BLUE}================================================${NC}"
if [ $FAILED_TESTS -eq 0 ]; then
    echo -e "${GREEN}✓ ALL TESTS PASSED${NC}"
    echo -e "${BLUE}================================================${NC}"
    exit 0
else
    echo -e "${RED}✗ SOME TESTS FAILED${NC}"
    echo -e "${BLUE}================================================${NC}"
    exit 1
fi