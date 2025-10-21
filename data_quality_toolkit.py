#!/usr/bin/env python3
"""
Data Quality Toolkit
A comprehensive tool for verifying and repairing crypto trading data
"""

import os
import sys
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def print_banner():
    """Print the toolkit banner"""
    banner = """
╔══════════════════════════════════════════════════════════════╗
║                    DATA QUALITY TOOLKIT                     ║
║                                                              ║
║  A comprehensive tool for crypto trading data verification   ║
║  and repair. This toolkit includes:                        ║
║                                                              ║
║  • Data verification (verify_data.py)                       ║
║  • Data repair (repair_data.py)                             ║
║  • Report generation (generate_report.py)                   ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)

def print_menu():
    """Print the main menu"""
    menu = """
Available Tools:
1. Verify Data Quality     - Check data integrity and completeness
2. Repair Data Issues      - Fix common data quality problems
3. Generate Reports        - Create detailed HTML and JSON reports
4. Run Full Pipeline      - Verify → Repair → Report (recommended)
5. Exit

Choose an option (1-5): """
    return input(menu)

def get_valid_choice():
    """Get a valid menu choice from user"""
    while True:
        try:
            choice = print_menu().strip()
            if choice in ['1', '2', '3', '4', '5']:
                return choice
            else:
                print("Invalid choice. Please select 1-5.")
        except ValueError:
            print("Invalid input. Please enter a number.")
        except KeyboardInterrupt:
            print("\nExiting...")
            return '5'
        except EOFError:
            print("\nExiting...")
            return '5'

def run_verification():
    """Run data verification"""
    logger.info("Running data verification...")
    try:
        import verify_data
        verifier = verify_data.DataVerifier()
        success = verifier.run_full_verification()
        return success
    except Exception as e:
        logger.error(f"Verification failed: {e}")
        return False

def run_repair():
    """Run data repair"""
    logger.info("Running data repair...")
    try:
        import repair_data
        repairer = repair_data.DataRepairer()
        success = repairer.repair_all_data()
        return success
    except Exception as e:
        logger.error(f"Repair failed: {e}")
        return False

def run_report_generation():
    """Run report generation"""
    logger.info("Generating reports...")
    try:
        import generate_report
        generator = generate_report.ReportGenerator()
        success = generator.generate_all_reports()
        return success
    except Exception as e:
        logger.error(f"Report generation failed: {e}")
        return False

def run_full_pipeline():
    """Run the complete pipeline: verify → repair → report"""
    logger.info("Running full data quality pipeline...")
    
    # Ask for confirmation before proceeding with repair
    print("\n" + "="*60)
    print("FULL PIPELINE CONFIRMATION")
    print("="*60)
    print("This will: Verify → Repair → Generate Reports")
    print("⚠️  WARNING: This will modify your data files!")
    print("A backup will be created automatically.")
    confirm = input("Do you want to proceed? (y/N): ")
    if confirm.lower() != 'y':
        logger.info("Full pipeline cancelled by user")
        return False
    
    # Step 1: Verify
    logger.info("Step 1/3: Verifying data...")
    verify_success = run_verification()
    
    if not verify_success:
        logger.warning("Verification found issues. Proceeding with repair...")
    
    # Step 2: Repair
    logger.info("Step 2/3: Repairing data...")
    repair_success = run_repair()
    
    if not repair_success:
        logger.error("Repair failed. Check logs for details.")
        return False
    
    # Step 3: Generate reports
    logger.info("Step 3/3: Generating reports...")
    report_success = run_report_generation()
    
    if not report_success:
        logger.error("Report generation failed. Check logs for details.")
        return False
    
    logger.info("✅ Full pipeline completed successfully!")
    return True

def main():
    """Main function"""
    print_banner()
    
    while True:
        try:
            choice = get_valid_choice()
            
            if choice == '1':
                print("\n" + "="*60)
                print("RUNNING DATA VERIFICATION")
                print("="*60)
                success = run_verification()
                if success:
                    print("✅ Verification completed successfully!")
                else:
                    print("⚠️  Verification found issues that need attention.")
                
            elif choice == '2':
                print("\n" + "="*60)
                print("RUNNING DATA REPAIR")
                print("="*60)
                print("⚠️  WARNING: This will modify your data files!")
                print("A backup will be created automatically.")
                confirm = input("Do you want to proceed? (y/N): ")
                if confirm.lower() == 'y':
                    success = run_repair()
                    if success:
                        print("✅ Data repair completed successfully!")
                    else:
                        print("❌ Data repair failed. Check logs for details.")
                else:
                    print("Repair cancelled.")
                
            elif choice == '3':
                print("\n" + "="*60)
                print("GENERATING REPORTS")
                print("="*60)
                success = run_report_generation()
                if success:
                    print("✅ Reports generated successfully!")
                    print("Check the 'reports' directory for HTML and JSON reports.")
                else:
                    print("❌ Report generation failed. Check logs for details.")
                
            elif choice == '4':
                print("\n" + "="*60)
                print("RUNNING FULL PIPELINE")
                print("="*60)
                success = run_full_pipeline()
                if success:
                    print("✅ Full pipeline completed successfully!")
                else:
                    print("❌ Pipeline failed. Check logs for details.")
                
            elif choice == '5':
                print("Exiting toolkit. Goodbye!")
                break
                
            else:
                print("Invalid choice. Please select 1-5.")
            
            print("\n" + "-"*60)
            
        except KeyboardInterrupt:
            print("\n\nExiting toolkit. Goodbye!")
            break
        except EOFError:
            print("\n\nExiting toolkit. Goodbye!")
            break
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            print(f"An error occurred: {e}")
            print("Please try again or select option 5 to exit.")

if __name__ == "__main__":
    main()
