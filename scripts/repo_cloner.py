import csv
import os
import subprocess
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_dir_size(path='.'):
    """
    Calculates the total size of a directory.
    """
    total_size = 0
    try:
        with os.scandir(path) as it:
            for entry in it:
                if entry.is_file():
                    total_size += entry.stat().st_size
                elif entry.is_dir():
                    total_size += get_dir_size(entry.path)
    except FileNotFoundError:
        return 0
    except Exception as e:
        logging.error(f"Error calculating size for {path}: {e}")
        return 0
    return total_size

def clone_repo(url, output_dir):
    """
    Clones a repository from a given URL into a specified directory.
    """
    if not url or not url.startswith(('http', 'https')):
        return f"Skipping invalid URL: {url}"
    
    try:
        repo_name = url.split('/')[-1].replace('.git', '')
        repo_path = os.path.join(output_dir, repo_name)
        
        if os.path.exists(repo_path):
            return f"Repository {repo_name} already exists. Skipping."

        logging.info(f"Cloning {url} into {repo_path}")
        subprocess.run(['git', 'clone', '--depth', '1', url, repo_path], check=True, capture_output=True, text=True)
        return f"Successfully cloned {repo_name}"
    except subprocess.CalledProcessError as e:
        return f"Failed to clone {url}. Error: {e.stderr}"
    except Exception as e:
        return f"An unexpected error occurred for {url}. Error: {e}"

def main():
    """
    Main function to parse arguments and clone repositories.
    """
    parser = argparse.ArgumentParser(description="Clone git repositories from a CSV file.")
    parser.add_argument("-file", help="Path to the CSV file containing repository links.")
    parser.add_argument("-output_dir", help="Directory to clone the repositories into.")
    parser.add_argument("-t", "--threads", type=int, default=5, help="Number of parallel threads to use for cloning. Default is 5.")
    parser.add_argument("--max-size", type=int, default=100, help="Maximum directory size in GB. Default is 100.")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    MAX_SIZE_BYTES = args.max_size * 1024 * 1024 * 1024
        
    try:
        with open(args.file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            # Skip header if it exists
            try:
                next(reader)
            except StopIteration:
                logging.info("CSV file is empty.")
                return
            
            urls = [row[1] for row in reader if len(row) > 1]

    except FileNotFoundError:
        logging.error(f"Error: The file {args.file} was not found.")
        return
    except Exception as e:
        logging.error(f"An error occurred while reading the CSV file: {e}")
        return

    with ThreadPoolExecutor(max_workers=args.threads) as executor:
        futures = []
        for url in urls:
            current_size = get_dir_size(args.output_dir)
            if current_size >= MAX_SIZE_BYTES:
                logging.warning(f"Clone directory size ({current_size / (1024**3):.2f} GB) has reached the maximum limit of {args.max_size} GB. Stopping.")
                # Cancel pending futures
                for future in futures:
                    future.cancel()
                break
            
            future = executor.submit(clone_repo, url, args.output_dir)
            futures.append(future)

        for future in as_completed(futures):
            try:
                result = future.result()
                if result:
                    logging.info(result)
            except Exception as exc:
                logging.error(f'A task generated an exception: {exc}')

if __name__ == "__main__":
    main()
