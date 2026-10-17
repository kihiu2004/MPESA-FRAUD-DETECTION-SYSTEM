#!/bin/bash
echo "========================================"
echo "M-Pesa Fraud Detection System"
echo "========================================"
echo
echo "Select an option:"
echo
echo "1. Install Dependencies"
echo "2. Run Full Pipeline (Train Models)"
echo "3. Run Dashboard (Web Interface)"
echo "4. Exit"
echo

read -p "Enter your choice (1-4): " choice

case $choice in
    1)
        pip install -r requirements.txt
        ;;
    2)
        python main.py
        ;;
    3)
        streamlit run dashboard/streamlit_app.py
        ;;
    4)
        exit 0
        ;;
    *)
        echo "Invalid choice. Please try again."
        ./run.sh
        ;;
esac
