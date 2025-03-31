import json


def calculate_pass_metrics(jsonl_file):
    pass_1_count = 0
    pass_5_count = 0
    total_count = 0

    with open(jsonl_file, "r") as file:
        for line in file:
            try:
                data = json.loads(line)
                results = data.get("results", [])

                # Check if 'results' is a list
                if not isinstance(results, list):
                    continue

                total_count += 1

                # Calculate pass@1
                pass_1 = results[0] if len(results) > 0 else False
                if pass_1:
                    pass_1_count += 1

                # Calculate pass@5
                pass_5 = any(results[:5]) if len(results) >= 1 else False
                if pass_5:
                    pass_5_count += 1

            except json.JSONDecodeError as e:
                print(f"Error decoding JSON line: {e}")
                continue

    return total_count, pass_1_count, pass_5_count


def main():
    jsonl_file = "Result/Qwen2.5-Coder-32B-Instruct-output-HumanEval.jsonl"

    total, pass_1, pass_5 = calculate_pass_metrics(jsonl_file)

    if total == 0:
        print("No valid examples found.")
        return

    print(f"Total examples: {total}")
    print(f"Pass@1: {pass_1} ({pass_1 / total:.2%})")
    print(f"Pass@5: {pass_5} ({pass_5 / total:.2%})")


if __name__ == "__main__":
    main()
