import re
import argparse

def check_train_loss(log_file_path, max_loss):
    with open(log_file_path, 'r') as file:
        data = file.readlines()

    # Regex to find lines with loss values and corresponding steps
    pattern = re.compile(r'^(\d+\.\d+),(\d+),')

    # Find all matches and record the last value before step 999
    last_value_before_999 = None
    for line in data:
        match = pattern.match(line)
        if match:
            loss, step = float(match.group(1)), int(match.group(2))
            if step >= 999:
                break
            last_value_before_999 = loss

    # Check the condition
    if last_value_before_999 is not None and last_value_before_999 <= max_loss:
        print(f"Success: The train loss before step 999 is {last_value_before_999}, which is within the acceptable limit of {max_loss}.")
    else:
        print(f"Failure: The train loss before step 999 is {last_value_before_999}, which exceeds the limit of {max_loss}.")
        exit(1)  # Causes the GitHub Action to fail if the condition is not met

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check the train loss from a log file.")
    parser.add_argument("log_file_path", type=str, help="Path to the log file.")
    parser.add_argument("max_loss", type=float, help="Maximum acceptable train loss.")
    args = parser.parse_args()

    check_train_loss(args.log_file_path, args.max_loss)