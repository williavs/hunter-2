"""Test script for description extraction"""

from ai_utils.simple_company_analyzer import extract_description

test_cases = [
    """# GTM Wizards Sales Intelligence Brief - New York Focus

COMPANY BASICS:
GTM Wizards is a B2B sales intelligence platform that helps sales teams identify and close more deals through AI-powered insights and workflow automation.

OFFERINGS:
- AI-powered sales enablement platform
- Prospect personality analysis tools
- Automated sales workflow solutions
- Integration with CRM systems
""",

    """# HubSpot - Marketing, Sales, and Service Software

HubSpot is an all-in-one inbound marketing, sales, and customer service platform that helps companies attract visitors, convert leads, and close customers. Founded in 2006, HubSpot has grown into a leading provider of marketing automation and CRM software for businesses of all sizes.
""",

    """COMPANY BASICS:
Salesforce is a cloud-based customer relationship management (CRM) platform that provides businesses with tools to manage sales, customer service, marketing automation, analytics, and application development.

OFFERINGS:
- Sales Cloud: CRM solution for sales teams
- Service Cloud: Customer service and support platform
- Marketing Cloud: Digital marketing automation
- Commerce Cloud: E-commerce solution
"""
]

print("Testing description extraction with different formats:\n")

for i, test_case in enumerate(test_cases):
    print(f"TEST CASE {i+1}:")
    print("-" * 40)
    print(f"FIRST 100 CHARS: {test_case[:100]}...\n")
    description = extract_description(test_case)
    print(f"EXTRACTED: {description}\n")
    print("-" * 40)
    print() 