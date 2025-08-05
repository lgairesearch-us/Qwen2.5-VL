import pandas as pd
from tqdm import tqdm
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
import json
from auto_eval.parse_trajectory import parse_trajectory
import base64
import io
import traceback
import ast

def format_input(intent, tool_calls):
    user_query = f"The user query: {intent}\nTask progress (You have done the following operations on the current device):"
    for i, tool_call in enumerate(tool_calls):
        # user_query += f" Step {i+1}: {json.dumps(tool_call)};"
        user_query += f" Step {i+1}: {tool_call};"
    return user_query

def process_trajectory(instance, image_dir, target_dims, max_pixels):

    try:
        task_id = instance['task_id']
        html_path = os.path.join(instance['result_dir'], f'render_{task_id}.html')
        trajectory_data = parse_trajectory(html_path)
        trajectory = trajectory_data['trajectory']

        past_actions = []
        training_examples = []

        for step_idx, step in enumerate(trajectory):

            input = format_input(instance['intent'], past_actions)

            raw_action = step['Predict Action']
            raw_action = raw_action.split("<tool_call>")[0]

            try:
                end_idx = raw_action.rindex('}') + 1
                planner_action = raw_action[:end_idx]
            except:
                planner_action = raw_action

            try:
                planner_action_dict = json.loads(planner_action)
                next_action = planner_action_dict['next_action']
            except:
                try:
                    planner_action_dict = ast.literal_eval(planner_action)
                    next_action = planner_action_dict['next_action']
                except:
                    try:
                        next_action = planner_action.split('"next_action":')[1].split("\n")[0].strip()
                    except:
                        print('Error parsing next action')
                        # print(planner_action)
                        # print(raw_action)
                        # print(step['Predict Action'])
                        continue

            past_actions.append(next_action)

            current_image_base64 = step['Image Observation']
            output_path = os.path.join(image_dir, f'{task_id}_{step_idx}.png')
            # Decode base64 image and save as PNG
            try:
                # Remove data URL prefix if present
                if "base64," in current_image_base64:
                    current_image_base64 = current_image_base64.split("base64,")[1]

                # Decode the base64 string
                image_data = base64.b64decode(current_image_base64)

                # Create an image from the binary data
                with open(output_path, 'wb') as f:
                    f.write(image_data)
            except Exception as e:
                print(f"Error saving image for task {task_id}, step {step_idx}: {e}")
                continue

            training_example = {
                "image": output_path,
                "conversations": [
                    {"from": "human", "value": f"<image>\n{input}"},
                    {"from": "gpt", "value": planner_action}
                ]
            }

            training_examples.append(training_example)

        return training_examples
    except Exception as e:
        print(f"Error processing trajectory: {e}")
        traceback.print_exc()
        return []

def read_json(json_path):
    # Check if the JSON file exists
    if os.path.exists(json_path):
        # Read the JSON file
        with open(json_path, 'r', encoding='utf-8') as json_file:
            json_content = json.load(json_file)
        return json_content
    return None


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='Prepare WebVoyager data for Qwen model training')
    parser.add_argument('--output_dir', type=str, help='Directory to save processed data')
    parser.add_argument('--csv_path', type=str, help='Path to the data')
    parser.add_argument('--target_width', type=int, default=1288,
                        help='Target width for processed images')
    parser.add_argument('--target_height', type=int, default=2044,
                        help='Target height for processed images')
    parser.add_argument('--max_pixels', type=int, default=3000000,
                        help='Maximum number of pixels in the image (width * height)')
    return parser.parse_args()

args = parse_args()

output_dir = args.output_dir
image_dir = os.path.join(output_dir, 'images')
os.makedirs(image_dir, exist_ok=True)

target_dims = [args.target_width, args.target_height]
training_instances = []

df = pd.read_csv(args.csv_path)
# filtered_df = df[df['auto_eval_res'] == 1]
filtered_df = df
data = filtered_df.to_dict('records')
print(f"Loaded {len(data)} instances with auto_eval_res == 1")

with ProcessPoolExecutor() as executor:
    process_func = partial(process_trajectory, image_dir=image_dir, target_dims=target_dims, max_pixels=args.max_pixels)
    futures = [executor.submit(process_func, instance) for instance in data]

    for future in tqdm(as_completed(futures), total=len(futures)):
        result = future.result()
        training_instances.extend(result)

# Write training instances to JSON file
output_json_path = os.path.join(output_dir, 'train.json')
with open(output_json_path, 'w') as f:
    json.dump(training_instances, f, indent=2)

print(f"Saved {len(training_instances)} training instances to {output_json_path}")
