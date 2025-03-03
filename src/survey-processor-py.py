import csv
import os
import re
from typing import List, Tuple

class SurveyProcessor:
    def __init__(self, csv_filename: str = "survey_responses.csv"):
        self.csv_filename = csv_filename
        self.current_iteration = self._get_last_iteration()
    
    def _get_last_iteration(self) -> int:
        """Read the CSV file if it exists and determine the last iteration number."""
        if not os.path.exists(self.csv_filename):
            return -1  # We'll increment this to 0 for the first entry
        
        try:
            with open(self.csv_filename, 'r', newline='') as f:
                reader = csv.DictReader(f)
                iterations = set(int(row['Iteration']) for row in reader)
                return max(iterations) if iterations else -1
        except Exception as e:
            print(f"Error reading CSV: {e}")
            return -1

    def parse_response(self, response_text: str) -> List[int]:
        """Parse a response text into a list of integers."""
        # Split into lines and clean up
        lines = [line.strip() for line in response_text.strip().split('\n')]
        
        # Validate we have exactly 17 responses
        if len(lines) != 17:
            raise ValueError(f"Expected 17 responses, got {len(lines)}")
        
        responses = []
        for i, line in enumerate(lines, 1):
            # Match pattern: "number) response"
            match = re.match(rf"^{i}\)\s*([1-5])$", line)
            if not match:
                raise ValueError(f"Invalid format in line {i}. Expected '{i}) X' where X is 1-5")
            
            response = int(match.group(1))

            # # Match pattern: "response"
            # response = int(line) 

            if not 1 <= response <= 5:
                raise ValueError(f"Response in line {i} must be between 1 and 5")
            
            responses.append(response)
        
        return responses

    def add_response(self, response_text: str) -> None:
        """Add a new response to the CSV file."""
        try:
            # Parse the response
            responses = self.parse_response(response_text)
            
            # Increment iteration counter
            self.current_iteration += 1
            
            # Prepare the rows
            rows = []
            for question_num, response in enumerate(responses, 1):
                rows.append({
                    '#': question_num,
                    'Iteration': self.current_iteration,
                    'Response': response
                })
            
            # Write to CSV
            file_exists = os.path.exists(self.csv_filename)
            with open(self.csv_filename, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=['#', 'Iteration', 'Response'])
                if not file_exists:
                    writer.writeheader()
                writer.writerows(rows)
            
            print(f"Successfully added response for iteration {self.current_iteration}")
            
        except Exception as e:
            print(f"Error adding response: {e}")
            raise

def main():
    processor = SurveyProcessor("SA_ionic_results.csv")
    
    print("\nSurvey Response Processor")
    print("------------------------")
    print("Format: Enter responses as '1) X' where X is 1-5")
    print("Example:")
    print("1) 5")
    print("2) 3")
    print("...")
    print("17) 4")
    print("\nEnter 'q' on a new line to quit")
    print("------------------------\n")
    
    while True:
        print("\nEnter new response (17 lines):")
        response_text = ""
        
        for _ in range(17):
            line = input()
            if line.lower() == 'q':
                return
            response_text += line + '\n'
        
        # lines = []
        
        # for i in range(17):
        #     line = input(f"{i+1}) ")
        #     if line.lower() == 'q':
        #         return
        #     lines.append(line)
        
        # response_text = '\n'.join(lines)
        
        try:
            processor.add_response(response_text)
        except Exception as e:
            print(f"\nError: {e}")
            print("Please try again.")

if __name__ == "__main__":
    main()
