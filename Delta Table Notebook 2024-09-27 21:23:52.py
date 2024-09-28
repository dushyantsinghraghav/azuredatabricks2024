# Databricks notebook source
# MAGIC %md
# MAGIC ## Learning about Delta Lake and Delta Tables

# COMMAND ----------

# MAGIC %sql
# MAGIC create database deltatables LOCATION

# COMMAND ----------

# Create a Delta table for an employee
spark.sql("""
CREATE TABLE IF NOT EXISTS employee (
    employee_id INT,
    first_name STRING,
    last_name STRING,
    email STRING,
    phone_number STRING,
    hire_date DATE,
    job_id STRING,
    salary DECIMAL(10, 2),
    manager_id INT,
    department_id INT
) USING DELTA
LOCATION 'dbfs:/mnt/input/employee'
""")

# COMMAND ----------

spark.sql("DESCRIBE employee")

# COMMAND ----------

# MAGIC %md
# MAGIC Log files created with delta table: json (transaction file) , crc file, parque checkpoint file
# MAGIC For every operation there will be a parquest file created

# COMMAND ----------

spark.sql("""
INSERT INTO employee (employee_id, first_name, last_name, email, phone_number, hire_date, job_id, salary, manager_id, department_id)
VALUES (1, 'John', 'Doe', 'john.doe@example.com', '555-1234', '2024-09-27', 'DEV', 75000.00, 101, 10)
""")

# COMMAND ----------

#lets look at the json file data created for the above two transactions
# Check if the directory exists
# Check if the path is a directory or a file
path = "dbfs:/mnt/input/employee/_delta_log/00000000000000000001.json"
if dbutils.fs.ls(path):
    files = dbutils.fs.ls(path)
    if files: 
        display(files)
else:
    print("Path does not exist.")

# COMMAND ----------

spark.sql("""
INSERT INTO employee (employee_id, first_name, last_name, email, phone_number, hire_date, job_id, salary, manager_id, department_id) VALUES
(2, 'Jane', 'Smith', 'jane.smith@example.com', '555-5678', '2024-09-27', 'HR', 65000.00, 102, 20),
(3, 'Alice', 'Johnson', 'alice.johnson@example.com', '555-8765', '2024-09-27', 'FIN', 70000.00, 103, 30),
(4, 'Bob', 'Brown', 'bob.brown@example.com', '555-4321', '2024-09-27', 'MKT', 72000.00, 104, 40),
(5, 'Charlie', 'Davis', 'charlie.davis@example.com', '555-6789', '2024-09-27', 'ENG', 80000.00, 105, 50),
(6, 'Diana', 'Miller', 'diana.miller@example.com', '555-9876', '2024-09-27', 'SALES', 68000.00, 106, 60),
(7, 'Eve', 'Wilson', 'eve.wilson@example.com', '555-5432', '2024-09-27', 'SUP', 71000.00, 107, 70),
(8, 'Frank', 'Moore', 'frank.moore@example.com', '555-6543', '2024-09-27', 'DEV', 75000.00, 108, 80),
(9, 'Grace', 'Taylor', 'grace.taylor@example.com', '555-7654', '2024-09-27', 'HR', 66000.00, 109, 90),
(10, 'Hank', 'Anderson', 'hank.anderson@example.com', '555-8765', '2024-09-27', 'FIN', 73000.00, 110, 100),
(11, 'Ivy', 'Thomas', 'ivy.thomas@example.com', '555-9876', '2024-09-27', 'MKT', 74000.00, 111, 110),
(12, 'Jack', 'Jackson', 'jack.jackson@example.com', '555-0987', '2024-09-27', 'ENG', 82000.00, 112, 120),
(13, 'Karen', 'White', 'karen.white@example.com', '555-1098', '2024-09-27', 'SALES', 69000.00, 113, 130),
(14, 'Leo', 'Harris', 'leo.harris@example.com', '555-2109', '2024-09-27', 'SUP', 72000.00, 114, 140),
(15, 'Mia', 'Martin', 'mia.martin@example.com', '555-3210', '2024-09-27', 'DEV', 76000.00, 115, 150)
""")

# COMMAND ----------

employee_changes = spark.sql("SELECT * FROM employee")
display(employee_changes)

# COMMAND ----------

spark.sql("""
UPDATE employee
SET salary = salary * 1.10
WHERE department_id = 50
""")

# COMMAND ----------

employee_changes_updated = spark.sql("SELECT * FROM employee")
display(employee_changes_updated)

# COMMAND ----------

spark.sql("""
DELETE FROM employee
WHERE id = 14
""")

# COMMAND ----------


