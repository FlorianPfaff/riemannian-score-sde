import re
import argparse

def check_train_loss(log_file_path, max_loss):
    with open(log_file_path, 'r') as file:
        data = file.readlines()

    # Regex to find lines with loss values and corresponding steps
    loss_pattern = re.compile(r'^(\d+\.\d+),(\d+),')
    # Regex to find Z value in the last row
    z_pattern = re.compile(r'^.*,.*,(.*\d+)$')

    # Find all matches and record the last value before step 999
    last_value_before_999 = None
    for line in data:
        loss_match = loss_pattern.match(line)
        if loss_match:
            loss, step = float(loss_match.group(1)), int(loss_match.group(2))
            if step >= 999:
                break
            last_value_before_999 = loss

        z_match = z_pattern.match(line)
        if z_match:
            last_z_value = float(z_match.group(1))

    # Check the condition for train loss
    if last_value_before_999 is not None and last_value_before_999 <= max_loss:
        print(f"Success: The train loss before step 999 is {last_value_before_999}, which is within the acceptable limit of {max_loss}.")
    else:
        print(f"Failure: The train loss before step 999 is {last_value_before_999}, which exceeds the limit of {max_loss}.")
        exit(1)  # Causes the GitHub Action to fail if the condition is not met

    # Check the condition for Z value
    if last_z_value is not None and abs(last_z_value - 1.0) <= 0.02:
        print("Success: The Z value in the last row deviates at most 0.02 from 1.")
    else:
        print(f"Failure: The Z value in the last row deviates more than 0.02 from 1. It is {last_z_value}.")
        exit(1)  # Causes the GitHub Action to fail if the condition is not met


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check the train loss from a log file.")
    parser.add_argument("log_file_path", type=str, help="Path to the log file.")
    parser.add_argument("max_loss", type=float, help="Maximum acceptable train loss.")
    args = parser.parse_args()

    check_train_loss(args.log_file_path, args.max_loss)