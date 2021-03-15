import os
import sys
import base64
import subprocess
from pathlib import Path
from collections import defaultdict

# Location of where the xdecode exe is located
xdecode_path = Path('C:/Users/dunca/Documents/JPL/Code/OMG/src/xdecodes/xdecode_v0_8_1') # Functional

# Location of where the emails are downloaded
email_location = f'{Path(__file__).resolve().parents[1]}/littleproby_emails'
email_path = Path(email_location)
email_files = list(email_path.glob('**/*.duncan-omg')) # Gets all emails with the corresponding file type

# Paths to locations to store new files
# Attachments contains the ungrouped encoded attachment data
# sbds contains the grouped decoded attachment data
attachments_path = Path(f'{email_location}/attachments')
sbds_path = Path(f'{email_location}/sbds')

# APEX float full ID numbers
APEX_floats = ['300534060836190', '300534060836350', '300534060832230']
APEX_IDs = ['6190', '6350', '2230']

# Log variable
log = defaultdict(list)

# Parse each email for its attached file data and filename
for i, email_file in enumerate(email_files):
    print(f'{i} emails completed out of {len(email_files)}', end='\r')
    email = open(email_file, 'r')
    email_filename = os.path.basename(email_file)
    filename = ''
    in_encoded_region = False
    encoded_lines = []
    prev_line = ''

    # Parse email line by line
    for line in email:
        # Retrieve attachment from the email if it exists and is an SBD file
        if ('Content-Disposition: attachment' in line and 'filename=' in line) or ('Content-Disposition: attachment' in prev_line and not filename and 'filename=' in line):
            # if the filename="" field is available, isolate the filename
            # by finding "filename=" and selecting the characters that correspond with the filename.
            # The actual filename starts 10 characters after "filename="" starts, and ends 1 charcter from the end
            # Example: filename="300534060836190_001547.sbd"
            #                    ^^^^^^^^^^^^^^^^^^^^^^^^^^
            temp_filename = line[line.find('filename=')+10:-1]
            if ';' in temp_filename or '"' in temp_filename:
                temp_filename = temp_filename.replace(';', '')
                temp_filename = temp_filename.replace('"', '')

            full_id_num = temp_filename[0:temp_filename.find('_')]

            #  Save filename if its ans SBD file and not an APEX float (compare its ID number to the list of APEX float numbers)
            if '.sbd' in temp_filename and full_id_num not in APEX_floats:
                filename = temp_filename
                if full_id_num not in log["alamo_floats"]:
                    log["alamo_floats"].append(full_id_num)
            elif full_id_num in APEX_floats:
                if full_id_num not in log["apex_floats"]:
                    log["apex_floats"].append(full_id_num)
                break
                # print(f'Attachment is for an APEX float (ignoring): {temp_filename}')
            else:
                if temp_filename not in log["failures"]:
                    log["failures"].append(((email_filename, temp_filename), 'Not SBD file'))
                break
                # print(f'Attachment is not an SBD file: {temp_filename}')
        elif ('Content-Disposition: attachment' in prev_line and not filename and 'filename=' not in line):
            if email_filename not in log["failures"]:
                log["failures"].append((email_filename, 'No attachment filename'))
            break
            # print('Attachment has no filename')

        # Check to see if the line is the line announcing the arrival of the data and its encoding
        if 'Content-Transfer-Encoding: base64' in line:
            in_encoded_region = True

        # If we were in the encoded region and have reached the end (signified by '--')
        if in_encoded_region and ('--' in line):
            in_encoded_region = False

        # If we are in the encoded region, and the line is not empty, and the line
        # doesnt contain the string 'base64' or 'X-Attachment' or 'Content-Disposition: attachment', 'filename', and the filename has been found then collect all the encoded lines
        if in_encoded_region and (line != "\n") and ('base64' not in line) and ('X-Attachment' not in line) and ('Content-Disposition: attachment' not in line) and ('filename' not in line) and (filename):
            encoded_lines.append(line)

        prev_line = line

    # print(f'Completed parsing email: {email_file}')

    # If email contained a valid filename
    if filename:
        # print(f'Filename: {filename}')
        # print(f'Encoded length: {len(encoded_lines)}')

        # Pull id number from the filename to separate them into separate directories
        # id number is the 4 numbers before the '_' in the filename
        # Example: 300534060830990_001547.sbd
        #                     ^^^^
        id_num = filename[filename.find('_')-4:filename.find('_')]

        if id_num not in log["ids"]:
            log["ids"].append(id_num)

        # Encoded filename is the original file name with _encoded at the end
        # Example: XXXXXXXXXXXXXX.sbd_encoded
        encoded_filename = f'{attachments_path}/{filename}_encoded'

        # Check to see if the encoded file has already been saved
        if os.path.exists(Path(encoded_filename)):
            log[f"previous_sbd_{id_num}"].append(encoded_filename)
            # print(f'{filename} has already been saved, skipping')
        else:
            log[f"new_sbd_{id_num}"].append(encoded_filename)
            # Writing encoded lines to the encoded_filename file defined above
            encoded_file = open(encoded_filename, 'w')

            for enc_line in encoded_lines:
                encoded_file.write(enc_line)
            
            encoded_file.close()

        # Check if /sbds/{id_num}/ directory exists and make it if not
        sbds_id_folder_path = f'{sbds_path}/{id_num}'
        if not os.path.exists(Path(sbds_id_folder_path)):
            os.makedirs(Path(sbds_id_folder_path))

        # Add filename to the directory
        sbds_id_folder_path += f'/{filename}'

        # Check if file has been decoded (exists in the /sbds/{id_num}/ directory)
        if os.path.exists(Path(sbds_id_folder_path)):
            log[f"previous_decode_{id_num}"].append(filename)
            # print(f'{id_num} already decoded')
        else:
            log[f"new_decode_{id_num}"].append(filename)
            # The encoded attachment file and the decoded attachment file
            encoded_file = open(encoded_filename, 'rb')
            decoded_file = open(sbds_id_folder_path, 'wb')

            # Decode the encoded attachment to base64 and write to the decoded file
            decoded = base64.b64decode(encoded_file.read())
            decoded_file.write(decoded)
            
            encoded_file.close()
            decoded_file.close()

    email.close()

run_xdecode = True
if run_xdecode:
    print('\n=======================================================================================')
    print('Running xdecode:\n')
    # For each ID number in the sbds directory, run xdecode on it and save its output to
    # a JSON file in the directory. JSON file is named {id_num}_decoded.json
    sbds_dirs = os.listdir(sbds_path)
    for id_num in sbds_dirs:
        if id_num in APEX_IDs:
            continue
        id_dir = f'{sbds_path}/{id_num}'
        subprocess.run(f'{xdecode_path} {id_dir} -f json > {id_dir}/{id_num}_decoded.json', shell=True) # output parsable json file of parsed data
        # subprocess.run(f'{xdecode_path} {id_dir} > {id_dir}/{id_num}_decoded_test.txt', shell=True) # output text file of parsed data

print_log = True
if print_log:
    print('\n=======================================================================================')
    print('Printing log:\n')

    # Apex floats found
    print(f'APEX floats found in emails ({len(log["apex_floats"])}): {log["apex_floats"]}')

    # Alamo floats found
    print(f'ALAMO floats found in emails ({len(log["alamo_floats"])}): {log["alamo_floats"]}')

    # Number of previous/new for each probe
    print(f'Probe numbers: ')
    for float_id in log["ids"]:
        print(f'    {float_id}: ')
        print(f'        previous SBDs: {len(log[f"previous_sbd_{float_id}"])}')
        print(f'        new SBDs: {len(log[f"new_sbd_{float_id}"])}')
        print(f'        previous decodes: {len(log[f"previous_decode_{float_id}"])}')
        print(f'        new decodes: {len(log[f"new_decode_{float_id}"])}')

    # Number of failures (of each type)
    no_attachment = []
    not_sbd = []
    other = []
    print('Failures: ')
    for name, fail_type in log["failures"]:
        if fail_type == 'No attachment filename':
            no_attachment.append(name)
        elif fail_type == 'Not SBD file':
            not_sbd.append(name)
        else:
            other.append((name, fail_type))
    print('    No attachment filename: ')
    for name in no_attachment:
        print(f'        {name}')
    print('    Not SBD file: ')
    for email_name, name in not_sbd:
        print(f'        Email: {email_name}')
        print(f'            file: {name}')
    print('    Other: ')
    for name, fail_type in other:
        print(f'        {fail_type}, {name}')
