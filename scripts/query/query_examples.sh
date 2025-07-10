#!/bin/bash
# Example queries to demonstrate the system

echo "REM RAG Query Examples"
echo "====================="
echo ""

# Make scripts executable
chmod +x query_random.sh ask.sh

echo "1. Count all node types:"
echo "   ./query_random.sh --count"
echo ""

echo "2. Get a random REM node with neighbors:"
echo "   ./query_random.sh --node-type rem --neighbors"
echo ""

echo "3. Get 3 random synthesis nodes:"
echo "   ./query_random.sh --node-type synthesis --num-samples 3"
echo ""

echo "4. Count only chunk nodes:"
echo "   ./query_random.sh --node-type chunk --count"
echo ""

echo "5. Ask a question with sources:"
echo "   ./ask.sh --show-sources \"What patterns emerged about sovereignty?\""
echo ""

echo "6. Interactive mode:"
echo "   ./ask.sh --interactive"
echo ""

echo "7. Ask with more context (k=20):"
echo "   ./ask.sh --k 20 \"What did we learn about nuclear deterrence?\""
echo ""