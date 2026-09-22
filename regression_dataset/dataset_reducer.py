import csv
import os

# Configuration
file_path = 'hospital-inpatient-discharges-sparcs-de-identified-2010-1-2.csv'
temp_file_path = 'hospital-inpatient-discharges-sparcs-de-identified-2010-1-reduced.csv'
target_rows = 50000

# Step 1: Count total rows (excluding header) quickly
print('Scanning file to count rows...')
with open(file_path, 'r', encoding='utf-8') as f:
  total_rows = sum(1 for _ in f) - 1  # Subtract 1 for the header line

if total_rows <= target_rows:
  print(
      f'File has {total_rows} rows, which is already less than or equal to'
      f' {target_rows}. No deletion needed.'
  )
else:
  # Calculate step interval to pick evenly spaced rows
  step = total_rows / target_rows
  print(f'Total rows found: {total_rows}. Sampling every {step:.2f} rows.')

  # Step 2: Read from original and write keeping rows to temp file
  with open(file_path, 'r', encoding='utf-8', newline='') as infile, open(
      temp_file_path, 'w', encoding='utf-8', newline=''
  ) as outfile:

    reader = csv.reader(infile)
    writer = csv.writer(outfile)

    # Keep header
    header = next(reader)
    writer.writerow(header)

    current_index = 0
    next_target_index = 0.0
    saved_count = 0

    for row in reader:
      if current_index >= int(next_target_index):
        writer.writerow(row)
        saved_count += 1
        next_target_index += step

      current_index += 1

      # Stop once we hit exactly 5,500 rows
      if saved_count >= target_rows:
        break

  # Step 3: Delete the old massive file and rename the temporary file
  print('Overwriting the original file...')
  os.replace(temp_file_path, file_path)
  print(
      f'Success! {file_path} has been overwritten and now contains only'
      f' {saved_count} rows.'
  )