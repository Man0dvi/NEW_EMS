import pandas as pd

# Load dataset
def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

# Function to handle user queries
def handle_query(query, dataset):
    # Parse query and extract relevant information
    # Implement logic to handle different types of queries
    # Provide appropriate responses based on the dataset
    response = "Sample response based on query: " + query
    return response

# Main function
if __name__ == "__main__":
    file_path = r'C:\Users\mandv\OneDrive\Desktop\EMS\ems_venv\Employee.csv'
    dataset = load_data(file_path)
    
    # Example query
    query = "What was my check-in time yesterday?"
    response = handle_query(query, dataset)
    print(response)
