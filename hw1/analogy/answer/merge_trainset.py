# Define file paths for the two analogy parts
file1 = "../data/input/dev.txt"  # First file
file2 = "../data/reference/dev.out"  # Second file
output_file = "../data/train/dev_combined.txt"  # Output file

# Open the output file for writing
with open(output_file, 'w') as outfile:
    # Open both files simultaneously
    with open(file1, 'r') as f1, open(file2, 'r') as f2:
        # Read and write the header from the first file without modification
        header = f1.readline().strip()
        outfile.write(f"{header}\n")
        
        # Skip the first line in the second file (assuming no corresponding header is needed)
        f2.readline()  # Discard the header of the second file

        # Process the rest of the lines line by line
        for line1, line2 in zip(f1, f2):
            # Strip trailing newlines or spaces
            line1 = line1.strip().lower()
            line2 = line2.strip().lower()
            
            # Combine both lines with a space
            combined_line = f"{line1} {line2}\n"
            
            # Write the combined line to the output file
            outfile.write(combined_line)

print("Files have been successfully merged, skipping the header, into", output_file)
