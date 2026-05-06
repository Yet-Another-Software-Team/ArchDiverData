import requests as re
import csv
import os
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

def check_link(row):
    """
    Checks a single link and returns the row if the link is accessible.
    """
    if len(row) > 1:
        url = row[1]
        try:
            response = re.get(url, timeout=5)
            if response.status_code == 200:
                return row
        except re.exceptions.RequestException:
            pass
    return None

def check_links_parallel(input_file, output_file, max_workers=10):
    """
    Reads a CSV file of links, checks if they are accessible in parallel, and saves the good links to a new CSV file.
    """
    with open(input_file, 'r', newline='', encoding='utf-8') as infile:
        reader = csv.reader(infile)
        header = next(reader)
        rows = list(reader)

    good_rows = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        with tqdm(total=len(rows), desc="Checking links") as pbar:
            future_to_row = {executor.submit(check_link, row): row for row in rows}
            for future in as_completed(future_to_row):
                result = future.result()
                if result:
                    good_rows.append(result)
                pbar.update(1)

    with open(output_file, 'w', newline='', encoding='utf-8') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(header)
        writer.writerows(good_rows)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                    prog='Link Checker',
                    description='Scans links from a CSV file and saves accessible links to a new CSV file.',
                    epilog='arguments: input_file (path to the input CSV file), output_file (path to the output CSV file)')

    parser.add_argument('-i','--input', help='Name of the input CSV file')
    parser.add_argument('-o', '--output', help='Name of the output CSV file')
    args = parser.parse_args()
    input_csv = 'raw links.csv'
    output_csv = 'good_links.csv'

    if args.input:
        input_csv = args.input

    if args.output:
        output_csv = args.output

    # Create output directory if it doesn't exist
    if not os.path.exists('../output'):
        os.makedirs('../output')

    input_csv = os.path.join('..', 'raw_data', input_csv)
    output_csv = os.path.join('..', 'output', output_csv)

    check_links_parallel(input_csv, output_csv)
