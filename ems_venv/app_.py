import streamlit as st
import pandas as pd
import sqlite3
from datetime import datetime

# Set up spaCy English language model
# You need to install Spacy and download the English model:
# !pip install spacy
# !python -m spacy download en_core_web_sm
import spacy
nlp = spacy.load("en_core_web_sm")

# Function to connect to SQLite database
def create_connection(db_file):
    conn = None
    try:
        conn = sqlite3.connect(db_file)
        return conn
    except sqlite3.Error as e:
        print(e)
    return conn

# Function to insert login time into check-in table
def insert_login(conn, employee_id):
    cursor = conn.cursor()
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    cursor.execute("INSERT INTO check_in (check_in_time, emp_id) VALUES (?, ?)", (current_time, employee_id))
    conn.commit()

# Function to insert logout time into check-out table
def insert_logout(conn, employee_id):
    cursor = conn.cursor()
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    cursor.execute("INSERT INTO check_out (check_out_time, emp_id) VALUES (?, ?)", (current_time, employee_id))
    conn.commit()

# Function to calculate working hours, attendance, and overtime
def calculate_attendance_and_overtime(conn, employee_id):
    cursor = conn.cursor()
    cursor.execute('''SELECT check_in_time, check_out_time FROM check_in
                      JOIN check_out ON check_in.emp_id = check_out.emp_id
                      WHERE check_in.emp_id = ?''', (employee_id,))
    result = cursor.fetchone()
    
    if result is None:
        attendance = 0
        working_hours = 0
        overtime = 0
    else:
        checkin_time, checkout_time = result
        if checkin_time is not None and checkout_time is not None:
            check_in = datetime.strptime(checkin_time, "%Y-%m-%d %H:%M:%S")
            check_out = datetime.strptime(checkout_time, "%Y-%m-%d %H:%M:%S")
            working_hours = (check_out - check_in).total_seconds() / 3600
            if working_hours >= 6:
                attendance = 10
                overtime = max(0, working_hours - 6)
            else:
                attendance = 5
                overtime = 6 - working_hours
        else:
            attendance = 0
            working_hours = 0
            overtime = 0
    
    return attendance, working_hours, overtime

# Function to get the next available employee ID
def get_next_employee_id(conn, from_check_in=True):
    cursor = conn.cursor()
    if from_check_in:
        cursor.execute("SELECT MAX(emp_id) FROM check_in")
    else:
        cursor.execute("SELECT MAX(emp_id) FROM check_out")
    max_id = cursor.fetchone()[0]
    
    if max_id is None:
        return 1  # If no employees exist yet, start with ID 1
    else:
        return max_id + 1



# Function to update emp_attendance table
def update_emp_attendance(conn, employee_id):
    attendance, working_hours, overtime = calculate_attendance_and_overtime(conn, employee_id)
    cursor = conn.cursor()
    cursor.execute('''INSERT OR REPLACE INTO attendance (employee_id, attendance, working_hours, overtime)
                      VALUES (?, ?, ?, ?)''', (employee_id, attendance, working_hours, overtime))
    conn.commit()

# Function to validate leave application
def validate_leave_application(conn, emp_id, leave_type, start_date, end_date):
    cursor = conn.cursor()
    
    # Get the current month and year
    current_month = datetime.now().strftime('%Y-%m')
    
    # Check if the employee has already taken the maximum allowed leaves for the current month
    cursor.execute("SELECT leaves_taken_this_month FROM leaves WHERE employee_id = ? AND strftime('%Y-%m', start_date) = ?", (emp_id, current_month))
    leaves_taken_result = cursor.fetchone()
    
    if leaves_taken_result is not None:
        leaves_taken_this_month = leaves_taken_result[0]
        # Assuming the maximum allowed leaves per month is 2
        if leaves_taken_this_month < 2:
            # Increment leave_id
            cursor.execute("SELECT MAX(leave_id) FROM leaves")
            max_leave_id = cursor.fetchone()[0]
            if max_leave_id is None:
                leave_id = 1
            else:
                leave_id = max_leave_id + 1
            
            # Insert into leaves table
            cursor.execute("INSERT INTO leaves (leave_id, employee_id, leave_type, start_date, end_date) VALUES (?, ?, ?, ?, ?)",
                           (leave_id, emp_id, leave_type, start_date, end_date))
            conn.commit()
            
            # Insert into leave_application table
            cursor.execute("INSERT INTO leave_application (emp_id, leave_id, status) VALUES (?, ?, ?)",
                           (emp_id, leave_id, "Approved"))
            conn.commit()

            
            return True
        else:
            return False
    else:
        # If no leave requests found for the current month, allow the leave application
        # Increment leave_id
        cursor.execute("SELECT MAX(leave_id) FROM leaves")
        max_leave_id = cursor.fetchone()[0]
        if max_leave_id is None:
            leave_id = 1
        else:
            leave_id = max_leave_id + 1
        
        # Insert into leaves table
        cursor.execute("INSERT INTO leaves (leave_id, employee_id, leave_type, start_date, end_date) VALUES (?, ?, ?, ?, ?)",
                       (leave_id, emp_id, leave_type, start_date, end_date))
        conn.commit()
        
        # Insert into leave_application table
        cursor.execute("INSERT INTO leave_applications (emp_id, leave_id, status) VALUES (?, ?, ?)",
                       (emp_id, leave_id, "Pending"))
        conn.commit()
        
        return True


# Function to get responses based on user input
def get_gemini_response(conn, prompt, emp_id):
    responses = []
    
    # Convert the prompt to lowercase and then check for keywords
    prompt_lower = prompt.lower()
    
    if "check-in" in prompt_lower:
        insert_login(conn, emp_id)
        cursor = conn.cursor()
        cursor.execute("SELECT check_in_time FROM check_in WHERE emp_id = ?", (emp_id,))
        checkin_time_result = cursor.fetchone()
        
        if checkin_time_result is not None:
            checkin_time = checkin_time_result[0]
            responses.append(f"Employee checked in at {checkin_time}")
        else:
            responses.append("No check-in record found for the employee.")
    
    if "check-out" in prompt_lower:
        insert_login(conn, emp_id)
        cursor = conn.cursor()
        cursor.execute("SELECT check_out_time FROM check_out WHERE emp_id = ?", (emp_id,))
        checkout_time_result = cursor.fetchone()
        
        if checkout_time_result is not None:
            checkout_time = checkout_time_result[0]
            responses.append(f"Employee checked in at {checkout_time}")
        else:
            responses.append("No check-in record found for the employee.")

    if "leave" in prompt_lower:
        # Extract leave-related information from the prompt
        doc = nlp(prompt)
        leave_type = None
        start_date = None
        end_date = None
        for token in doc:
            if token.ent_type_ == "DATE" and not start_date:
                start_date = token.text
            elif token.ent_type_ == "DATE" and start_date and not end_date:
                end_date = token.text
            elif token.text.lower() in ["vacation", "sick", "maternity"]:
                leave_type = token.text.lower()
        
        if leave_type and start_date and end_date:
            # Validate leave application
            is_valid_leave = validate_leave_application(conn, emp_id, leave_type, start_date, end_date)
            if is_valid_leave:
                responses.append("Your leave application has been submitted and is pending approval.")
                cursor = conn.cursor()
                cursor.execute("UPDATE leave_applications SET status = 'Approved' WHERE emp_id = ? AND status = 'Pending'", (emp_id,))
                conn.commit()

                responses.append("Your leave application has been approved.")

            else:
                responses.append("Sorry, your leave application cannot be processed at this time.")
        else:
            responses.append("Please provide valid leave details.")
    # Check for keywords related to location
    if "location" in prompt_lower:
        cursor = conn.cursor()
        cursor.execute("SELECT location FROM employee WHERE employee_id = ?", (emp_id,))
        location_result = cursor.fetchone()
        
        if location_result is not None:
            location = location_result[0]
            responses.append(f"Employee's location is: {location}")
        else:
            responses.append(f"Location not found for employee with ID '{emp_id}'.")
    
    # Check for keywords related to working hours and overtime
    if "working hours" in prompt_lower or "overtime" in prompt_lower:
        cursor = conn.cursor()
        cursor.execute("SELECT working_hours, overtime, attendance FROM attendance WHERE employee_id = ?", (emp_id,))
        attendance_result = cursor.fetchone()
        
        if attendance_result is not None:
            working_hours, overtime, attendance = attendance_result
            responses.append(f"Attendance: {attendance}, Working Hours: {working_hours:.2f} hours, Overtime: {overtime:.2f} hours")
        else:
            responses.append("No attendance record found for the employee.")
    
    return responses



# Initialize Streamlit app
st.set_page_config(page_title="Gemini Employee Demo")
st.header("Gemini Application")

# Connect to SQLite database
conn = create_connection('ems.db')

# Create emp_attendance table if not exists
cursor = conn.cursor()

# Check-in and Check-out buttons
check_in_button = st.button("Check-in")
check_out_button = st.button("Check-out")

# Execute actions based on button clicks
# Check-in button logic
if check_in_button:
    employee_id = get_next_employee_id(conn)
    insert_login(conn, employee_id)
    st.write(f"Employee with ID {employee_id} checked-in successfully!")

# Check-out button logic
if check_out_button:
    employee_id = get_next_employee_id(conn, from_check_in=False)
    insert_logout(conn, employee_id)
    update_emp_attendance(conn, employee_id)
    st.write(f"Employee with ID {employee_id} checked-out successfully!")

# Chatbot interface
st.header("Gemini Application")
prompt = st.text_area("Prompt: ", "What was John Doe's check-in time?")
submit = st.button("Generate Insights")

if submit:
    employee_id = get_next_employee_id(conn)
    responses = get_gemini_response(conn, prompt,employee_id)
    st.subheader("Insights Generated:")
    for response in responses:
        st.write(response)

# Close database connection


conn.close()
