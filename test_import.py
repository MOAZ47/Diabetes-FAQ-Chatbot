import sys
import os

print(f"Current Working Directory: {os.getcwd()}")
print("Current sys.path:")
for p in sys.path:
    print(f"  {p}")

try:
    from tools.retrieve_information import RetrieveInformation
    print("\nSuccessfully imported 'RetrieveInformation' from 'tools'!")
except ModuleNotFoundError as e:
    print(f"\nFailed to import 'RetrieveInformation': {e}")
except Exception as e:
    print(f"\nAn unexpected error occurred: {e}")